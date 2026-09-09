r"""One-off driver for stage 5.1 of the MOTOR pipeline: MEDS events -> sequences.

Run from the repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/etl/build_sequences.py
"""

import argparse
from pathlib import Path

from thesis.modelling.finetune.sequences import run_build_sequences

ROOT = Path(__file__).resolve().parents[2]
EVENTS = ROOT / "meds_output" / "normalized" / "data"
LABELS = ROOT / "meds_output" / "labels" / "landmark_labels.parquet"
DEST = ROOT / "meds_output" / "sequences"
DICTIONARY = ROOT / "motor_model" / "dictionary"

# 65,536 is what transformer.vocab_size declares in motor_model/model/config.msgpack
VOCAB_SIZE = 65_536

LENGTH = 1024
STRIDE = 512


def _parse_args() -> argparse.Namespace:
    """Parses the paths and widths, all defaulting to the repo's own layout."""
    parser = argparse.ArgumentParser(
        description="Tokenise the MEDS shards into MOTOR sequences."
    )
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--length", type=int, default=LENGTH)
    parser.add_argument("--stride", type=int, default=STRIDE)

    return parser.parse_args()


def main() -> None:
    """Resolves the paths and builds the sequences for every shard."""
    args = _parse_args()

    print(f"events     {args.events}")
    print(f"labels     {args.labels}")
    print(f"dictionary {args.dictionary}")
    print(f"dest       {args.dest}")
    print(f"length {args.length}, stride {args.stride}, vocab {args.vocab_size}")

    dest = run_build_sequences(
        args.events,
        args.labels,
        args.dest,
        dictionary=args.dictionary,
        vocab_size=args.vocab_size,
        length=args.length,
        stride=args.stride,
    )
    print(f"Done. Sequences written to {dest}")


if __name__ == "__main__":
    main()
