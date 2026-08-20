r"""One-off driver for stage 7a: MEDS events -> the XGBoost baseline's long features.

Run from the repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/etl/build_baseline_features.py

Reads stage 2.6's normalised shards, stage 4's landmark labels and stage 3's subject
split, and writes one long feature table plus one spine slice per shard. An existing
dest is refused.

The baseline is deliberately fed the **same events the transformer sees** -- stage
2.6's output, at its measured 33% coverage -- so the comparison isolates the
architecture rather than data access. What it does not share is the tokenisation:
where stage 5.1 turns a creatinine of 2.4 mg/dL into a bin, this keeps the float,
which is what makes a KDIGO-shaped delta computable at all.
"""

import argparse
from pathlib import Path

from thesis.modelling.baseline.features import run_build_features

ROOT = Path(__file__).resolve().parents[2]
EVENTS = ROOT / "meds_output" / "normalized" / "data"
LABELS = ROOT / "meds_output" / "labels" / "landmark_labels.parquet"
SPLIT = ROOT / "meds_output" / "labels" / "subject_split.parquet"
DEST = ROOT / "meds_output" / "baseline"


def _parse_args() -> argparse.Namespace:
    """Parses the paths, all defaulting to the repo's own layout."""
    parser = argparse.ArgumentParser(
        description="Build the XGBoost baseline's long feature table."
    )
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--dest", type=Path, default=DEST)

    return parser.parse_args()


def main() -> None:
    """Resolves the paths and builds the features for every shard."""
    args = _parse_args()

    print(f"events {args.events}")
    print(f"labels {args.labels}")
    print(f"split  {args.split}")
    print(f"dest   {args.dest}")

    written = run_build_features(args.events, args.labels, args.split, args.dest)
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
