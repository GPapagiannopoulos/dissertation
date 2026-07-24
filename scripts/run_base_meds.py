r"""One-off driver for stage 1 of the MOTOR ETL: the MIMIC-IV -> MEDS conversion.

Run from the repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/run_base_meds.py

The meds_etl_mimic console script is taken from the same virtualenv as the
interpreter running this file, so activating the venv (or fixing PATH) is not
required. Output is written to a fresh directory under the project root; there
is no overwrite by design, so delete it before re-running.

Because this is a multi-hour, memory-hungry job, run it detached and capped::

    tmux new -s meds
    systemd-run --user --scope -p MemoryMax=6G \\
        .venv-modelling/bin/python scripts/run_base_meds.py \\
        2>&1 | tee ~/meds_etl_$(date +%F_%H%M).log
"""

import argparse
import sys
from pathlib import Path

from thesis.modelling.etl_pipeline.base_meds import run_base_meds

ROOT = Path(__file__).resolve().parents[1]
# mimic_data holds the 2.2/ symlink -> mimic-iv-3.1, i.e. the parent the ETL wants.
SRC = ROOT / "mimic_data"
DEST = ROOT / "meds_output" / "base"


def _parse_args() -> argparse.Namespace:
    """Parses the tuning knobs a one-off run may need to vary."""
    parser = argparse.ArgumentParser(
        description="Run the MIMIC-IV -> MEDS conversion (ETL stage 1)."
    )
    parser.add_argument("--num-shards", type=int, default=200)
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Sort-stage processes; lower this if the run trips its memory cap.",
    )
    parser.add_argument("--backend", choices=["cpp", "polars"], default="cpp")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEST,
        help="Output root; must not already exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Resolves the venv-local executable and launches the conversion."""
    args = _parse_args()
    # Take the console script from the venv running this file, not from PATH.
    executable = Path(sys.executable).with_name("meds_etl_mimic")

    print(f"src        {SRC}")
    print(f"dest       {args.dest}")
    print(f"executable {executable}")
    print(f"shards={args.num_shards} proc={args.num_proc} backend={args.backend}")

    run_base_meds(
        SRC,
        args.dest,
        executable=executable,
        num_shards=args.num_shards,
        num_proc=args.num_proc,
        backend=args.backend,
    )
    print(f"Done. MEDS output at {args.dest}")


if __name__ == "__main__":
    main()
