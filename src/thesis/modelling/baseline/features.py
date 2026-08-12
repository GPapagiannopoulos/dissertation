"""Helper functions for the implementation of the XGBoost tree."""

from collections.abc import Sequence
from pathlib import Path

import polars as pl

EXCLUDED_CODES: tuple[str, ...] = ("MEDS_BIRTH", "MEDS_DEATH")
"""Codes no landmark may see, for two different reasons.

`MEDS_DEATH` is leakage: it is day-resolution, so a death recorded at 00:00 on the
discharge date precedes any landmark later that day, and "this patient is recorded
dead" predicts a bad outcome almost perfectly. Measured on three shards, it is
visible to 103 landmarks. `MEDS_BIRTH` is the opposite problem -- it is carried by
every subject exactly once, so it is constant and cannot split anything.

Dropping both also restores parity with the transformer: neither code carries an
OMOP concept, so stage 5.1's tokeniser already discards them and the sequences
never contain either.
"""

CREATININE_CODES: tuple[str, ...] = ("LOINC/2160-0", "LOINC/38483-4")
"""Serum and blood creatinine, the two codes KDIGO's definition rests on.

Urine creatinine, 24h clearance and the protein ratio are deliberately excluded --
they are different measurements on a different scale, and averaging them into one
baseline would make the delta meaningless.
"""

CONTEXT_PREFIX = "CTX/"
"""Namespace for derived features, keeping them sortable apart from real codes."""


def build_spine(labels: pl.LazyFrame, split: pl.LazyFrame) -> pl.LazyFrame:
    """Builds the spine for the model.

    Args:
        labels (pl.LazyFrame): The dataset labelled and with prediction
            grids
        split (pl.LazyFrame): The subject_ids split into training/validation/
            testing

    Returns:
        pl.LazyFrame: A LazyFrame representing the spine, specifically
            times when predictions where made for each subject.
    """
    return (
        labels.join(split, on="subject_id", how="left")
        .sort("subject_id", "visit_id", "prediction_time")
        .with_row_index("landmark_id")
    )


def visible_events(events: pl.LazyFrame, drop: list[str]) -> pl.LazyFrame:
    """Selects events to be included in the model.

    Args:
        events: The dataset labelled and with prediction grids
        drop (list[str]): A list of codes to exclude from the model

    Returns:
        pl.LazyFrame: A LazyFrame containing only events that the model is
            allowed to see
    """
    return events.select("subject_id", "code", "time", "numeric_value").filter(
        pl.col("code").is_in(drop).not_() & pl.col("time").is_not_null()
    )


def running_counts(events: pl.LazyFrame) -> pl.LazyFrame:
    """Returns a running count of visible events per subject and event type."""
    return events.sort("subject_id", "code", "time").with_columns(
        pl.col("time").cum_count().over("subject_id", "code").alias("n_obs")
    )


def candidate_pairs(spine: pl.LazyFrame, events: pl.LazyFrame) -> pl.LazyFrame:
    """Joins every landmark with every code its subject ever carries."""
    return spine.select("landmark_id", "subject_id", "prediction_time").join(
        events.select("subject_id", "code").unique(), on="subject_id", how="inner"
    )


def last_observation(pairs: pl.LazyFrame, events: pl.LazyFrame) -> pl.LazyFrame:
    """Joins landmarks with all the codes that happened during it."""
    return (
        pairs.sort("prediction_time")
        .join_asof(
            events.sort("time"),
            left_on="prediction_time",
            right_on="time",
            by=["subject_id", "code"],
            strategy="backward",
        )
        .drop_nulls("time")
        .with_columns(
            (
                (pl.col("prediction_time") - pl.col("time")).dt.total_minutes() / 60
            ).alias("hours_since")
        )
        .select(
            "landmark_id",
            "code",
            pl.col("numeric_value").alias("last_value"),
            "hours_since",
            "n_obs",
        )
    )


def build_context(
    spine: pl.LazyFrame,
    events: pl.LazyFrame,
    *,
    creatinine: Sequence[str] = CREATININE_CODES,
    baseline_hours: int = 24,
) -> pl.LazyFrame:
    """Derives the handful of features the last-observation pass cannot express.

    Emitted in the **long schema** as pseudo-codes under `CTX/`, so they ride the
    same CSR path as everything else rather than needing a dense block hstacked on.
    Only `last_value` is populated; `hours_since` and `n_obs` are null and so cost no
    matrix entry at all.

    Three of the five exist because AKI is *defined* on a creatinine trajectory, and
    a last observation cannot express a trajectory. The admission baseline is the
    minimum in the first `baseline_hours` of the stay, matching the KDIGO arm of the
    EDA work rather than a rolling seven-day floor.

    This is the one function that reads the **raw** shard rather than
    `visible_events`, because age is measured from `MEDS_BIRTH` -- the same trap
    stage 5.2 hit, where tokenising first leaves every age null.

    Only `admittime` and `prediction_time` are read off the spine. `diagtime`,
    `dischtime`, `horizon_time`, `horizon_hours` and `died_in_hospital` all sit in
    the same frame and are all **future** information; `diagtime` in particular is
    the moment the AKI was diagnosed and would be a perfect leak.

    Args:
        spine (pl.LazyFrame): The landmarks, from `build_spine`.
        events (pl.LazyFrame): One raw shard, before `visible_events`.
        creatinine (Sequence[str]): Codes counting as a serum creatinine.
        baseline_hours (int): The admission window the baseline is drawn from.

    Returns:
        pl.LazyFrame: Long-schema rows -- `landmark_id`, `code`, `last_value`,
            `hours_since`, `n_obs` -- carrying one row per landmark per context
            feature that could be computed.
    """
    births = (
        events.filter(pl.col("code") == "MEDS_BIRTH")
        .group_by("subject_id")
        .agg(pl.col("time").min().alias("birth_time"))
    )
    stream = (
        events.filter(
            pl.col("code").is_in(creatinine) & pl.col("numeric_value").is_not_null()
        )
        .select("subject_id", "time", "numeric_value")
        .sort("time")
    )

    admissions = spine.select("subject_id", "visit_id", "admittime").unique()
    baseline = (
        admissions.join(stream, on="subject_id")
        .filter(
            (pl.col("time") >= pl.col("admittime"))
            & (pl.col("time") < pl.col("admittime") + pl.duration(hours=baseline_hours))
        )
        .group_by("subject_id", "visit_id")
        .agg(pl.col("numeric_value").min().alias("baseline"))
    )

    wide = (
        spine.select(
            "landmark_id", "subject_id", "visit_id", "admittime", "prediction_time"
        )
        .sort("prediction_time")
        .join_asof(
            stream,
            left_on="prediction_time",
            right_on="time",
            by="subject_id",
            strategy="backward",
        )
        .rename({"numeric_value": "latest"})
        .join(births, on="subject_id", how="left")
        .join(baseline, on=["subject_id", "visit_id"], how="left")
        .select(
            "landmark_id",
            age_years=(
                pl.col("prediction_time") - pl.col("birth_time")
            ).dt.total_minutes()
            / (60 * 24 * 365.25),
            hours_since_admission=(
                pl.col("prediction_time") - pl.col("admittime")
            ).dt.total_minutes()
            / 60,
            creatinine_baseline=pl.col("baseline"),
            creatinine_delta=pl.col("latest") - pl.col("baseline"),
            # A zero baseline is not physiological, but one bad row would otherwise
            # emit an infinity that survives every quantile boundary.
            creatinine_ratio=pl.when(pl.col("baseline") > 0)
            .then(pl.col("latest") / pl.col("baseline"))
            .otherwise(None),
        )
    )

    return (
        wide.unpivot(index="landmark_id", variable_name="code", value_name="last_value")
        .drop_nulls("last_value")
        .select(
            "landmark_id",
            code=CONTEXT_PREFIX + pl.col("code"),
            last_value=pl.col("last_value").cast(pl.Float32),
            hours_since=pl.lit(None, dtype=pl.Float64),
            n_obs=pl.lit(None, dtype=pl.UInt32),
        )
    )


def run_build_features(
    events: Path,
    labels: Path,
    split: Path,
    dest: Path,
    *,
    drop: Sequence[str] = EXCLUDED_CODES,
) -> Path:
    """Materialises the long feature table, shard by shard.

    Shards are processed one at a time, as in every stage since 2.6: peak memory
    stays at one shard, and a crash at shard 137 leaves 137 readable outputs. Stage
    2.6's shards are subject-disjoint, so a landmark's whole history lives in the one
    shard its subject does and no landmark is ever split across two outputs.

    `landmark_id` is assigned **once, globally**, before the loop -- unlike stage
    5.2's `sequence_id`, which numbers densely per call and needs a running offset.
    Numbering here is a single sorted `with_row_index` over all 2,965,363 landmarks,
    so the shards could be built in parallel without reworking anything.

    The spine is written per shard rather than once, because the matrix builder needs
    a shard's roster of landmarks -- including any carrying no visible code at all,
    which vanish from the long table. Scanning `spine/*.parquet` recovers the global
    view for free.

    Args:
        events (Path): Stage 2.6's shard folder, i.e. <normalized>/data.
        labels (Path): The landmark labels parquet from stage 4.
        split (Path): The subject split parquet from stage 3.
        dest (Path): The folder to create, which must not already exist.
        drop (Sequence[str]): Codes no landmark may see. Defaults to
            `EXCLUDED_CODES`.

    Returns:
        Path: The folder written, holding `spine/` and `features/`.

    Raises:
        FileExistsError: If dest already exists.
        FileNotFoundError: If events holds no parquet, or an input file is missing.
        ValueError: If any landmark's subject is absent from the split, which would
            put labelled rows in no fold at all.
    """
    if dest.exists():
        raise FileExistsError(
            f"Destination {dest} already exists; refusing to overwrite. "
            f"Remove it or choose a new path."
        )
    shards = sorted(events.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"Found no parquet shards in {events}. Point events at stage 2.6's "
            f"data/ folder."
        )
    for path in (labels, split):
        if not path.is_file():
            raise FileNotFoundError(f"Expected a file at {path}.")

    spine = build_spine(pl.scan_parquet(labels), pl.scan_parquet(split)).collect()
    orphaned = int(spine["fold"].is_null().sum())
    if orphaned:
        raise ValueError(
            f"{orphaned} landmarks belong to a subject absent from {split}, so they "
            f"would sit in no fold. Rebuild the split, or the labels, against the "
            f"same dataset."
        )
    print(f"spine holds {spine.height} landmarks over {len(shards)} shards")

    spine_dir = dest / "spine"
    feature_dir = dest / "features"
    spine_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)

    for position, shard in enumerate(shards, start=1):
        raw = pl.scan_parquet(shard)
        counted = running_counts(visible_events(raw, drop))
        subjects = counted.select("subject_id").unique()
        slice_ = spine.lazy().join(subjects, on="subject_id", how="semi")

        # The asof leaves hash-probe order behind; sorting is what makes the
        # artifact reproducible, as in stages 2.6, 4 and 5.2.
        long = (
            pl.concat(
                [
                    last_observation(candidate_pairs(slice_, counted), counted),
                    # Reads `raw`, not `counted` -- age needs MEDS_BIRTH, which
                    # `visible_events` has already dropped.
                    build_context(slice_, raw),
                ]
            )
            .sort("landmark_id", "code")
            .collect()
        )
        roster = slice_.sort("landmark_id").collect()

        roster.write_parquet(spine_dir / shard.name)
        long.write_parquet(feature_dir / shard.name)

        print(
            f"  [{position}/{len(shards)}] {shard.name}: "
            f"{roster.height} landmarks, {long.height} pairs",
            flush=True,
        )

    return dest
