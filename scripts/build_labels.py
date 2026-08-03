"""One-off driver for Stage 4 of the MOTOR ETL: lay the labelled landmark grid.

Run from repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/build_labels.py

Reads the admission windows and the deaths from the normalised shards and the
AKI onsets from the surviving-positives labels, then writes one row per
prediction landmark.

Writes roughly 3M rows at the default 12h spacing and 48h horizon. Existing
destination paths are refused.
"""

import argparse
from pathlib import Path

import polars as pl

from thesis.modelling.cohort.labeller import (
    _PREDICTION_GRID_DELTA_HOURS,
    _PREDICTION_HORIZON_HOURS,
    run_build_labels,
)

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "meds_output" / "normalized" / "data"
LABELS = ROOT / "meds_output" / "labels" / "surviving_aki_admissions.parquet"
DEST = ROOT / "meds_output" / "labels" / "landmark_labels.parquet"


def _parse_args() -> argparse.Namespace:
    """Parses the paths and the grid parameters, all defaulted to the design."""
    parser = argparse.ArgumentParser(
        description="Lay a labelled prediction grid over every inpatient admission."
    )
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--delta", type=str, default=_PREDICTION_GRID_DELTA_HOURS)
    parser.add_argument("--horizon", type=str, default=_PREDICTION_HORIZON_HOURS)

    return parser.parse_args()


def main() -> None:
    """Resolves the paths and writes the grid."""
    args = _parse_args()

    print(f"events  {args.events}")
    print(f"labels  {args.labels}")
    print(f"dest    {args.dest}")
    print(f"delta   {args.delta}")
    print(f"horizon {args.horizon}")

    dest = run_build_labels(
        args.events,
        args.labels,
        args.dest,
        delta_hours=args.delta,
        prediction_horizon=args.horizon,
    )

    grid = pl.scan_parquet(dest)
    print(f"Done. Written to {dest}")
    print(
        grid.group_by("boolean_value")
        .agg(
            n_landmarks=pl.len(),
            n_admissions=pl.col("visit_id").n_unique(),
            median_horizon_h=pl.col("horizon_hours").median(),
        )
        .sort("boolean_value")
        .collect()
    )


if __name__ == "__main__":
    main()
