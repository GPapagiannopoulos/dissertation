r"""One-off driver for stage 2 of the MOTOR ETL: MEDS -> meds_reader database.

Run from the repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/build_meds_reader_db.py

The meds_reader_convert console script is taken from the same virtualenv as the
interpreter running this file, so activating the venv (or fixing PATH) is not
required. src defaults to the output of stage 1; dest must not already exist,
so delete it before re-running.

Log the run, since the cli reports progress only on stdout::

    .venv-modelling/bin/python scripts/build_meds_reader_db.py --num-threads 8 \\
        2>&1 | tee ~/reader_db_$(date +%F_%H%M).log
"""

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

    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Threads the cli uses; one child process is launched regardless.",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEST, help="Output root; must not already exist."
    )
    parser.add_argument(
        "--src", type=Path, default=SRC, help="MEDS dataset root; must contain data/."
    )

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
