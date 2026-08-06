r"""One-off driver for stage 5.1 of the MOTOR pipeline: MEDS events -> sequences.

Run from the repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/build_sequences.py

Reads stage 2.6's normalised shards, the landmark labels and MOTOR's dictionary, and
writes a folder holding one tokenised, positioned, aged sequence file per shard plus
the labels pinned onto their positions. An existing dest is refused.

The events are cut to the subjects that carry a landmark and to the events at or
before each subject's last one; everything after it is causally unreachable. The
ancestor expansion is deliberately NOT applied here -- it is a per-token lookup the
collate joins at batch time, and materialising it would multiply the artifact.
"""

import argparse
from pathlib import Path

from thesis.modelling.motor.sequences import run_build_sequences

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "meds_output" / "normalized" / "data"
LABELS = ROOT / "meds_output" / "labels" / "landmark_labels.parquet"
DEST = ROOT / "meds_output" / "sequences"
DICTIONARY = ROOT / "motor_model" / "dictionary"

# 65,536 is what transformer.vocab_size declares in motor_model/model/config.msgpack;
# the embedding table has exactly this many rows and a token index IS a row of it.
VOCAB_SIZE = 65_536

# MOTOR attends 496 positions back, inclusive. A stride of half the length therefore
# gives every position in a chunk past the first a full window, so no landmark is
# scored from a truncated history.
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
