r"""One-off driver for stage 2.6 of the MOTOR ETL: MEDS codes -> MOTOR tokens.

Run from the repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/apply_concept_map.py

Reads the stage 1 shards and concept_map.parquet, and writes a fresh MEDS dataset root
holding the rewritten shards plus its own metadata. An existing dest is refused.
"""

import argparse
from pathlib import Path

from thesis.modelling.etl_pipeline.normalise import run_apply_concept_map

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "meds_output" / "base" / "data"
DEST = ROOT / "meds_output" / "normalized"
CONCEPT_MAP = ROOT / "meds_output" / "concept_map.parquet"
ATHENA = ROOT / "athena"


def _parse_args() -> argparse.Namespace:
    """Parses the paths, all of which default to the repo's own layout."""
    parser = argparse.ArgumentParser(
        description="Rewrite the MEDS shards onto the codes MOTOR holds."
    )
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--concept-map", type=Path, default=CONCEPT_MAP)
    parser.add_argument("--athena", type=Path, default=ATHENA)

    return parser.parse_args()


def main() -> None:
    """Resolves the paths and applies the concept map to every shard."""
    args = _parse_args()

    print(f"events      {args.events}")
    print(f"concept map {args.concept_map}")
    print(f"athena      {args.athena}")
    print(f"dest        {args.dest}")

    dest = run_apply_concept_map(
        args.events,
        args.dest,
        concept_map=args.concept_map,
        concept=args.athena / "CONCEPT.csv",
    )
    print(f"Done. Normalised dataset written to {dest}")


if __name__ == "__main__":
    main()
