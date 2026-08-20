r"""Stage 7b driver: fit the XGBoost baseline on the AKI landmark task.

Run from the repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/train/train_baseline_aki.py \
        --dest motor_output/runs/xgb-seed0

Reads the long feature table stage 7a wrote, builds the training and validation
matrices through a shard-at-a-time `ExtMemQuantileDMatrix` whose binned pages spill to
`<dest>/cache`, and early-stops on validation AUPRC. An existing dest is refused.

The matrices do not fit in this host's 14 GB, so run it under a memory cap -- the
cgroup is what turns an over-run into a dead job rather than a dead desktop:

    systemd-run --user --unit=xgb-aki --working-directory="$PWD" \
        -p MemoryMax=9500M -p MemorySwapMax=1G -p OOMScoreAdjust=800 \
        .venv-modelling/bin/python scripts/train/train_baseline_aki.py \
            --dest motor_output/runs/xgb-seed0 --nthread 4 --max-bin 64
"""

import argparse
from pathlib import Path

from thesis.modelling.baseline.model import run_train_baseline

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "meds_output" / "baseline"
DEST = ROOT / "motor_output" / "runs" / "xgb-seed0"


def _parse_args() -> argparse.Namespace:
    """Reads the run's configuration off the command line."""
    parser = argparse.ArgumentParser(description="Fit the XGBoost AKI baseline.")
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nthread", type=int, default=None)
    parser.add_argument("--max-bin", type=int, default=256)

    return parser.parse_args()


def main() -> None:
    """Resolves the paths and fits the booster."""
    args = _parse_args()

    print(f"features {args.features}")
    print(f"dest     {args.dest}")

    written = run_train_baseline(
        args.features,
        args.dest,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        nthread=args.nthread,
        max_bin=args.max_bin,
    )
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
