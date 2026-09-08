r"""Stage 7c driver: score the fine-tuned MOTOR encoder against the XGBoost baseline.

Run from the repo root with the modelling environment's interpreter:

    .venv-modelling/bin/python scripts/evaluate/comparison/compare_models.py \
        --booster motor_output/runs/xgb-seed0 \
        --checkpoint motor_output/runs/aki-seed0/best.pt

Both models are scored on the **whole** validation fold and through the same
`binary_metrics`, so the only difference between the two columns is the architecture.
The training loop's own AUPRC was measured on a 150-batch subsample and is not
comparable to anything; this is what replaces it.
"""

import argparse
from pathlib import Path

from thesis.modelling.evaluation.compare import run_comparison

ROOT = Path(__file__).resolve().parents[3]
FEATURES = ROOT / "meds_output" / "baseline"
BOOSTER = ROOT / "motor_output" / "runs" / "xgb-seed0"
CHECKPOINT = ROOT / "motor_output" / "runs" / "aki-seed0" / "best.pt"
SEQUENCES = ROOT / "meds_output" / "sequences"
SPLIT = ROOT / "meds_output" / "labels" / "subject_split.parquet"
ORACLE = ROOT / "motor_output" / "oracle_fp32.npz"
DICTIONARY = ROOT / "motor_model" / "dictionary"
DEST = ROOT / "motor_output" / "comparison"


def _parse_args() -> argparse.Namespace:
    """Reads the comparison's configuration off the command line."""
    parser = argparse.ArgumentParser(description="Compare MOTOR against XGBoost.")
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--booster", type=Path, default=BOOSTER)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--sequences", type=Path, default=SEQUENCES)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--oracle", type=Path, default=ORACLE)
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--fold", type=str, default="validation")
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    """Resolves the paths and runs both models over the fold."""
    args = _parse_args()

    print(f"features   {args.features}")
    print(f"booster    {args.booster}")
    print(f"checkpoint {args.checkpoint}")
    print(f"fold       {args.fold}")

    run_comparison(
        args.booster,
        args.features,
        args.checkpoint,
        args.sequences,
        args.split,
        args.oracle,
        args.dictionary,
        args.dest,
        fold=args.fold,
        resamples=args.resamples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
