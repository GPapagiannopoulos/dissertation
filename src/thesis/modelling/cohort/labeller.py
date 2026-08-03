"""Module housing the custom labeller and its helper functions: Stage 4.

MOTOR natively uses femr to apply a time horizon to the labels and
handle censoring. The module responsible for this works by looping over
subjects, making it inefficient and unfit for our purposes (making predictions
per admission rather than patient). Hence, we develop our own labeller
that vectorizes grid generation per-admission, and let MOTOR handle the
mapping of subjects to labels.
"""

from pathlib import Path
from typing import Final

import polars as pl

_INPATIENT_VISIT_CODES: Final[list[str]] = ["Visit/IP", "Visit/ERIP"]
_COMMUNITY_ACQUIRED_CUTOFF_HOURS: Final[int] = 48
_PREDICTION_GRID_DELTA_HOURS: Final[str] = "12h"
_PREDICTION_HORIZON_HOURS: Final[str] = "48h"


def build_admission_windows(events: pl.LazyFrame) -> pl.LazyFrame:
    """Extracts the admission window information for the normalized db.

    Args:
        events: a LazyFrame containing all normalized events

    Returns:
        pl.LazyFrame: a frame containing the admission and discharge window ::

            subject_id              (Int64)                 as given
            visit_id                (Int64)                 as given
            visit_code              (String)                e.g. Visit/IP | Visit/ERIP
            admittime               (Datetime('us'))        renamed
            dischtime               (Datetime('us'))        renamed
    """
    return events.filter(pl.col("code").str.starts_with("Visit/")).select(
        pl.col("subject_id"),
        pl.col("visit_id"),
        pl.col("code").alias("visit_code"),
        pl.col("time").alias("admittime"),
        pl.col("end").alias("dischtime"),
    )


def filter_inpatient_admissions(windows: pl.LazyFrame) -> pl.LazyFrame:
    """Filters the admission window only for inpatient admissions."""
    return windows.filter(pl.col("visit_code").is_in(_INPATIENT_VISIT_CODES))


def filter_false_admissions(windows: pl.LazyFrame) -> pl.LazyFrame:
    """Filters out admissions where the LOS is non-positive."""
    return windows.filter(pl.col("admittime") < pl.col("dischtime"))


def join_diagnosis_time(windows: pl.LazyFrame, labels: pl.LazyFrame) -> pl.LazyFrame:
    """Joins the labels with the time of diagnosis to the admission window.

    Not all visits have a timestamp for a diagnosis of interest. Uses a left
    join to preserve this information as a null value for diagtime.

    Args:
        windows (pl.LazyFrame): a LazyFrame containing the admission windows
        labels (pl.LazyFrame): a LazyFrame containing the output of the
            'identify_surviving_admissions.py' script

    Returns:
        pl.LazyFrame: a LazyFrame containing a timestamp in pl.Datetime['us']
            with the moment of diagnosis for a specific visit, null otherwise.
    """
    return windows.join(
        labels.select(
            "subject_id",
            "visit_id",
            pl.col("timestamp").cast(pl.Datetime("us")).alias("diagtime"),
        ),
        on=["subject_id", "visit_id"],
        how="left",
    )


def build_landmark_grid(
    windows_with_onset: pl.LazyFrame,
    delta_hours: str = _PREDICTION_GRID_DELTA_HOURS,
) -> pl.LazyFrame:
    """Generates the prediction landmark grid for each visit.

    Args:
        windows_with_onset (pl.LazyFrame): a LazyFrame containing the admission windows
            for each visit_id and the time of diagnosis
        delta_hours (str): a string representing the number of hours
            between prediction times.

    Returns:
        pl.LazyFrame: a LazyFrame containing one record for each prediction
            landmark, starting 48h post admission, and ending at discharge or
            diagnosis confirmation exclusive, whichever comes first. Drops visits
            without any admission landmarks in the generated window.
    """
    return (
        windows_with_onset.with_columns(
            pl.datetime_ranges(
                pl.col("admittime")
                + pl.duration(hours=_COMMUNITY_ACQUIRED_CUTOFF_HOURS),
                pl.min_horizontal(pl.col("dischtime"), pl.col("diagtime")),
                interval=delta_hours,
                closed="left",
            ).alias("prediction_time")
        )
        .filter(pl.col("prediction_time").list.len() > 0)
        .explode("prediction_time")
    )


def apply_time_horizons(
    grid: pl.LazyFrame, horizon: str = _PREDICTION_HORIZON_HOURS
) -> pl.LazyFrame:
    """Applies the time horizon to the grid.

    Creates a new boolean column indicating whether a target diagnosis was
    made within the designated time horizon.
    """
    horizon_unit = horizon[-1]
    horizon_value = int(horizon[:-1])
    match horizon_unit:
        case "w":
            horizon_duration = pl.duration(weeks=horizon_value)
        case "d":
            horizon_duration = pl.duration(days=horizon_value)
        case "h":
            horizon_duration = pl.duration(hours=horizon_value)
        case _:
            raise ValueError(
                f"Unrecognized time horizon unit: {horizon_unit}\n"
                f"Valid units are: 'w', 'd', 'h'"
            )

    grid_with_horizon = grid.with_columns(
        pl.min_horizontal(
            pl.col("prediction_time") + horizon_duration,
            (pl.col("dischtime")),
        ).alias("horizon_time")
    ).with_columns(
        horizon_hours=(
            pl.col("horizon_time") - pl.col("prediction_time")
        ).dt.total_seconds()
        / 3600
    )

    boolean_value = pl.col("diagtime").is_not_null() & (
        pl.col("diagtime") <= pl.col("horizon_time")
    )
    return grid_with_horizon.with_columns(
        boolean_value=boolean_value,
    )


def run_build_labels(
    events: Path,
    labels: Path,
    dest: Path,
    *,
    delta_hours: str = _PREDICTION_GRID_DELTA_HOURS,
    prediction_horizon: str = _PREDICTION_HORIZON_HOURS,
) -> Path:
    """Lays a labelled prediction landmark grid over every inpatient admission.

    Args:
        events: Path to the normalised shard folder, stage 2.6's data/
        labels: Path to the surviving positive diagnoses parquet
        dest: Path to the parquet to write
        delta_hours: spacing between landmarks, as a Polars duration string
        prediction_horizon: how far each landmark forecasts, same format

    Returns:
        Path: dest, holding one row per landmark

    Raises:
        FileExistsError: if dest already exists
        FileNotFoundError: if labels is missing, or events holds no parquet
        ValueError: if an admission in labels kept no events through stage 2.6,
            leaving nothing to featurize it from
    """
    if dest.exists():
        raise FileExistsError(
            f"Destination {dest} already exists; refusing to overwrite. "
            f"Remove it or choose a new path."
        )

    if not labels.is_file():
        raise FileNotFoundError(f"Expected file at {labels}.")
    labels_lf = pl.scan_parquet(labels)
    stranded = (
        labels_lf.filter(pl.col("n_surviving_events") == 0)
        .select(pl.len())
        .collect()
        .item()
    )
    if stranded:
        raise ValueError(
            f"{stranded} labelled admissions kept no events through stage 2.6 and "
            f"cannot be featurized. Rebuild {labels} against the normalised shards."
        )

    shards = sorted(events.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"Found no parquet shards in {events}. Point events at the normalised "
            f"dataset's data/ folder."
        )
    events_lf = pl.scan_parquet(shards)

    windows = build_admission_windows(events_lf)
    inpatient_admission_windows = filter_inpatient_admissions(windows)
    valid_admission_windows = filter_false_admissions(inpatient_admission_windows)
    window_w_diagnosis_time = join_diagnosis_time(valid_admission_windows, labels_lf)
    prediction_grid = build_landmark_grid(window_w_diagnosis_time, delta_hours)
    time_horizons = apply_time_horizons(prediction_grid, prediction_horizon)

    dest.parent.mkdir(parents=True, exist_ok=True)
    # The left join returns hash-probe order, so the grid comes out shuffled by
    # subject. Sorting makes the artifact reproducible and groups each subject's
    # landmarks the way femr reads them.
    time_horizons.sort("subject_id", "prediction_time").sink_parquet(dest)

    written = pl.scan_parquet(dest).select(
        n_landmarks=pl.len(),
        n_positive=pl.col("boolean_value").sum(),
        n_admissions=pl.col("visit_id").n_unique(),
        n_subjects=pl.col("subject_id").n_unique(),
    )
    summary = written.collect().row(0, named=True)
    rate = (
        100 * summary["n_positive"] / summary["n_landmarks"]
        if summary["n_landmarks"]
        else 0.0
    )
    print(
        f"{summary['n_landmarks']} landmarks over {summary['n_admissions']} admissions "
        f"and {summary['n_subjects']} subjects, of which "
        f"{summary['n_positive']} positive ({rate:.2f}%)"
    )

    return dest
