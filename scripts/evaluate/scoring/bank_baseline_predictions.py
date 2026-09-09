r"""Bank one fitted booster's per-landmark predictions for a fold.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/evaluate/scoring/bank_baseline_predictions.py \
        --booster motor_output/runs/ng-xgb-seed0-600 \
        --fold test \
        --dest motor_output/comparison/newgrid/xgboost600_test_predictions.npz

Writes the same `(scores, targets, subjects, times)` bundle the transformer arm banks,
so any later comparison, decision curve or ensemble is npz arithmetic with no model
loaded.

Check the round window it prints. `predict_fold` scores at the booster's
`best_iteration`, which is documented not to survive a save/load round-trip reliably;
if it is absent the model is scored at every round instead, and two folds scored at
two different round counts are incomparable without anything raising.
"""

import argparse
from pathlib import Path

import numpy as np

from thesis.modelling.evaluation.compare import score_baseline

ROOT = Path(__file__).resolve().parents[3]
FEATURES = ROOT / "meds_output" / "baseline"
BOOSTER = ROOT / "motor_output" / "runs" / "ng-xgb-seed0-600"


def _parse_args() -> argparse.Namespace:
    """Reads the booster, the fold and where the bundle goes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--booster", type=Path, default=BOOSTER)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--fold", default="validation")
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument(
        "--expect-auprc",
        type=float,
        default=None,
        help="fail unless the fold scores this AUPRC to four decimals. Use it when "
        "re-banking a fold whose value is already known, to prove the round window "
        "matches the one the existing artifacts were produced at",
    )
    return parser.parse_args()


def main() -> None:
    """Scores the fold and writes the bundle."""
    args = _parse_args()
    if args.dest.exists():
        raise SystemExit(f"{args.dest} exists; refusing to overwrite a banked bundle.")

    print(f"booster  {args.booster}")
    print(f"features {args.features}")
    print(f"fold     {args.fold}", flush=True)

    metrics, bundle = score_baseline(
        args.booster,
        args.features,
        fold=args.fold,
        resamples=args.resamples,
        seed=args.seed,
    )

    print(
        f"\nauprc {metrics['auprc']:.5f} "
        f"[{metrics['auprc_lo']:.5f}, {metrics['auprc_hi']:.5f}] | "
        f"auroc {metrics['auroc']:.5f} | brier {metrics['brier']:.6f} | "
        f"ece {metrics['ece']:.5f} | p@1% {metrics['precision_at_1pct']:.5f}"
    )
    print(
        f"n {int(metrics['n']):,} | positives {int(metrics['n_positive']):,} | "
        f"subjects {int(metrics['n_subjects']):,} | "
        f"base rate {metrics['base_rate']:.5%}"
    )

    if args.expect_auprc is not None:
        gap = abs(metrics["auprc"] - args.expect_auprc)
        if round(gap, 4) != 0.0:
            raise SystemExit(
                f"Expected AUPRC {args.expect_auprc:.5f} on {args.fold}, got "
                f"{metrics['auprc']:.5f} (off by {gap:.5f}). The booster is being "
                f"scored at a different round window than the existing artifacts "
                f"were, so the folds would not be comparable. Do not proceed."
            )
        print(f"\nmatches the expected {args.expect_auprc:.5f} -- round window agrees")

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.dest, *bundle)
    print(f"wrote {args.dest}")


if __name__ == "__main__":
    main()
