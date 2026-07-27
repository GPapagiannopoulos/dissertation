"""One-off driver for calling the db creation functions."""

import argparse
import sys
from pathlib import Path

from thesis.modelling.etl_pipeline.meds_reader_db import run_meds_reader_convert

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "meds_output" / "base"
DEST = ROOT / "meds_output" / "reader_db"


def _parse_args() -> argparse.Namespace:
    """Parses the tuning knobs that may need to vary."""
    parser = argparse.ArgumentParser(
        description="Convert the MEDS long-format shards into a MEDS reader db."
    )

    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument(
        "--dest", type=Path, default=DEST, help="Output root; must not already exist."
    )
    parser.add_argument("--src", type=Path, default=SRC)

    return parser.parse_args()


def main() -> None:
    """Launches the db creation functions."""
    args = _parse_args()
    executable = Path(sys.executable).with_name("meds_reader_convert")

    print(f"src        {args.src}")
    print(f"dest       {args.dest}")
    print(f"executable {executable}")
    print(f"threads={args.num_threads}")

    run_meds_reader_convert(
        args.src, args.dest, executable=executable, num_threads=args.num_threads
    )
    print(f"Done. MEDS db built at {args.dest}")


if __name__ == "__main__":
    main()
