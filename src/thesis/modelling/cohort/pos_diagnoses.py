"""Module for extracting patients with a positive AKI diagnosis as per criteria."""

from pathlib import Path

import polars as pl


def identify_surviving_aki_admissions(
    labels: pl.LazyFrame, events: pl.LazyFrame
) -> pl.LazyFrame:
    """Identifies the admissions with valid tokens post normalization.

    The process of normalizing MIMIC-IV data resulted into dropping of
    events that could not be mapped to any valid MOTOR token. For positive
    diagnosis events to be useful they need surviving events.
    """
    events_per_admission = events.group_by("visit_id").len("n_visits")

    return (
        labels.join(events_per_admission, on="visit_id", how="left")
        .with_columns(n_surviving_events=pl.coalesce(pl.col("n_visits"), 0.0))
        .drop("n_visits")
    )


def run_identify_surviving_aki_admissions(
    labels: Path, events: Path, dest: Path
) -> Path:
    """Annotates every positive label with the events its admission kept.

    Events carrying no visit_id (MEDS_BIRTH, demographics) are attributable to no
    admission and so are counted against none of them.

    Args:
        labels: Path to the positive diagnosis labels parquet
        events: Path to the normalised shard folder
        dest: Path to the parquet to write

    Returns:
        Path: dest, holding the labels plus n_surviving_events

    Raises:
        FileExistsError: if dest already exists
        FileNotFoundError: if labels is missing, or events holds no parquet
    """
    if dest.exists():
        raise FileExistsError(
            f"Destination {dest} already exists; refusing to overwrite. "
            f"Remove it or choose a new path."
        )
    if not labels.is_file():
        raise FileNotFoundError(f"Expected a file at {labels}.")
    shards = sorted(events.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"Found no parquet shards in {events}. Point events at the normalised "
            f"dataset's data/ folder."
        )

    annotated = identify_surviving_aki_admissions(
        pl.scan_parquet(labels),
        pl.scan_parquet(shards).select("visit_id"),
    ).collect(engine="streaming")

    dest.parent.mkdir(parents=True, exist_ok=True)
    annotated.write_parquet(dest)

    stranded = annotated.filter(pl.col("n_surviving_events") == 0).height
    print(f"{annotated.height} positive admissions, of which {stranded} kept no events")

    return dest
