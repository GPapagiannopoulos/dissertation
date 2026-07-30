r"""One-off driver for annotating the positive labels with post-normalisation survival.

Run from the repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/identify_surviving_admissions.py

Reads the positive diagnosis labels and the normalised shards, and writes the labels
back out with an n_surviving_events column. An existing dest is refused.
"""

import argparse
from pathlib import Path

import polars as pl

from thesis.modelling.cohort.pos_diagnoses import run_identify_surviving_aki_admissions

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "meds_output" / "labels" / "diagnosis_labels.parquet"
EVENTS = ROOT / "meds_output" / "normalized" / "data"
DEST = ROOT / "meds_output" / "labels" / "surviving_aki_admissions.parquet"


def _parse_args() -> argparse.Namespace:
    """Parses the paths, all of which default to the repo's own layout."""
    parser = argparse.ArgumentParser(
        description="Annotate the positive labels with the events their admission kept."
    )
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--dest", type=Path, default=DEST)

    return parser.parse_args()


def main() -> None:
    """Resolves the paths and annotates every positive label."""
    args = _parse_args()

    print(f"labels {args.labels}")
    print(f"events {args.events}")
    print(f"dest   {args.dest}")

    dest = run_identify_surviving_aki_admissions(args.labels, args.events, args.dest)
    print(f"Done. Annotated labels written to {dest}")
    print(pl.scan_parquet(dest).select("n_surviving_events").collect().describe())


if __name__ == "__main__":
    main()
