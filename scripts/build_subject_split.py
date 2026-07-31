"""One-off driver for Stage 3 of the MOTOR ETL: split subjects into train/val/test.

Run from repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/build_subject_split.py

Reads the subject_ids from the meds_reader database, the admission counts
from the normalized shards, and the positives from the labels.

Writes one row per subject. Existing destination paths are refused.
"""

import argparse
from pathlib import Path

import polars as pl

from thesis.modelling.cohort.split import FOLD_SEED, run_build_subject_split

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "meds_output" / "reader_db_v2"
ADMISSIONS = ROOT / "meds_output" / "normalized" / "data"
LABELS = ROOT / "meds_output" / "labels" / "surviving_aki_admissions.parquet"
DEST = ROOT / "meds_output" / "labels" / "subject_split.parquet"


def _parse_args() -> argparse.Namespace:
    """Parses the paths, all of which default to the repo's layout."""
    parser = argparse.ArgumentParser(
        description="Assign every subject in the database a fold."
    )
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--admissions", type=Path, default=ADMISSIONS)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--seed", type=int, default=FOLD_SEED)

    return parser.parse_args()


def main() -> None:
    """Resolves the paths and writes the split."""
    args = _parse_args()

    print(f"db         {args.db}")
    print(f"admissions {args.admissions}")
    print(f"labels     {args.labels}")
    print(f"dest       {args.dest}")
    print(f"seed       {args.seed}")

    dest = run_build_subject_split(
        args.db, args.dest, args.admissions, args.labels, seed=args.seed
    )
    folds = pl.read_parquet(dest)
    print(f"Done. {folds.height} subjects written to {dest}")
    print(
        folds.pivot(
            on="fold", index="stratum", values="subject_id", aggregate_function="len"
        ).sort("stratum")
    )


if __name__ == "__main__":
    main()
