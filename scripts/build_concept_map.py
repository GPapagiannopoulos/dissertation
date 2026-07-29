r"""One-off driver for stage 2.5 of the MOTOR ETL: MEDS codes -> MOTOR tokens.

Run from the repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/build_concept_map.py

Reads the stage 1 shards, the Athena vocabulary exports and MOTOR's dictionary, and
writes concept_map.parquet. The job takes seconds and holds no state, so an existing
dest is overwritten rather than refused.
"""

import argparse
from pathlib import Path

import msgpack
import polars as pl

from thesis.modelling.etl_pipeline.codes import MANUAL_CONCEPT_MAP
from thesis.modelling.etl_pipeline.coverage import (
    build_concept_map,
    code_inventory,
    load_motor_vocab,
    scan_athena,
)

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "meds_output" / "base" / "data" / "*.parquet"
CODE_METADATA = ROOT / "meds_output" / "base" / "metadata" / "codes.parquet"
ATHENA = ROOT / "athena"
DICTIONARY = ROOT / "motor_model" / "dictionary"
MODEL_CONFIG = ROOT / "motor_model" / "model" / "config.msgpack"
DEST = ROOT / "meds_output" / "concept_map.parquet"


def _read_vocab_size(path: Path) -> int:
    """Reads vocab_size from the model's own config, so the two cannot disagree."""
    with open(path, "rb") as f:
        config = msgpack.load(f, use_list=False, strict_map_key=False)
    return config["transformer"]["vocab_size"]


def _parse_args() -> argparse.Namespace:
    """Parses the paths and the one knob that may need to vary."""
    parser = argparse.ArgumentParser(
        description="Map the codes the MEDS shards emit onto MOTOR's vocabulary."
    )
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--code-metadata", type=Path, default=CODE_METADATA)
    parser.add_argument("--athena", type=Path, default=ATHENA)
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY)
    parser.add_argument("--model-config", type=Path, default=MODEL_CONFIG)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Overrides the size the model config declares.",
    )

    return parser.parse_args()


def main() -> None:
    """Builds the concept map and writes it to parquet."""
    args = _parse_args()
    vocab_size = args.vocab_size or _read_vocab_size(args.model_config)

    print(f"events        {args.events}")
    print(f"code metadata {args.code_metadata}")
    print(f"athena        {args.athena}")
    print(f"dictionary    {args.dictionary}")
    print(f"dest          {args.dest}")
    print(f"vocab_size={vocab_size}")

    vocab = load_motor_vocab(args.dictionary, vocab_size=vocab_size)
    inventory = code_inventory(pl.scan_parquet(args.events)).collect(engine="streaming")
    concept_map = build_concept_map(
        inventory.lazy(),
        vocab,
        pl.scan_parquet(args.code_metadata),
        scan_athena(args.athena / "CONCEPT.csv"),
        scan_athena(args.athena / "CONCEPT_RELATIONSHIP.csv"),
        MANUAL_CONCEPT_MAP,
    ).collect()

    events = inventory["count"].sum()
    covered = concept_map["count"].sum()
    print(f"\n{inventory.height} distinct codes, {events} events")
    print(f"{concept_map.height} codes mapped, {covered} events covered", end=" ")
    print(f"({covered / events:.1%})")

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    concept_map.write_parquet(args.dest)
    print(f"Done. Concept map written to {args.dest}")


if __name__ == "__main__":
    main()
