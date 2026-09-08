"""Pair the LoRA ensemble against the XGBoost baseline on one fold.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python \
        scripts/evaluate/comparison/pair_ensemble_vs_baseline.py \
        --fold testing \
        --dest motor_output/comparison/newgrid/ensemble_vs_xgboost600_test.json

This is the headline comparison of the project, and it is banked-npz arithmetic: both
arms were scored once and their per-landmark predictions saved, so this needs no model,
no GPU and no inference pass.

Every metric is differenced on the SAME draw of subjects. Two one-model intervals
overlap freely even when one model wins on nearly every resample, because they carry
the variance of the cohort itself; pairing cancels that, and what survives is the
difference between the models. The 12-hourly grid puts ~23 correlated predictions
inside one admission, so the draw is over SUBJECTS -- a landmark-level bootstrap would
report an interval several times too narrow.

Members and the arm rosters come from `ensemble.roster`, so this cannot drift from the
decision-curve or diversity analyses.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from thesis.modelling.ensemble.diversity import align_members, load_member
from thesis.modelling.ensemble.roster import (
    ROOT,
    baseline_bundle,
    drop_contested,
    lora_ensemble,
)
from thesis.modelling.evaluation.intervals import _draw_rows, _subject_groups
from thesis.modelling.evaluation.metrics import binary_metrics

REPORTED = ("auprc", "auroc", "brier", "ece", "precision_at_1pct")

# Brier and ECE are ERRORS, so a negative difference favours the ensemble while a
# positive one favours the tree -- the opposite of every other metric here. Reporting
# one "share of draws favouring the ensemble" across all five would invert the reading
# on exactly the two metrics the calibration claim rests on.
LOWER_IS_BETTER = frozenset({"brier", "ece"})


def _parse_args() -> argparse.Namespace:
    """Reads the fold and the resampling budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", default="validation")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Scores both arms, then bands every reported difference."""
    args = _parse_args()

    members = lora_ensemble(args.fold)
    baseline = baseline_bundle(args.fold)
    if not baseline.is_file():
        raise SystemExit(
            f"{baseline} is missing; bank it with "
            f"scripts/evaluate/scoring/bank_baseline_predictions.py --fold {args.fold}."
        )

    loaded = [load_member(path) for path in [*members, baseline]]
    stacked, targets, subjects = align_members(drop_contested(loaded))
    targets = targets.astype(float)
    groups = _subject_groups(subjects)

    # the baseline was appended last, so everything before it is the ensemble
    ensemble = stacked[:-1].mean(axis=0)
    tree = stacked[-1]

    print(
        f"fold {args.fold}: {len(members)} members, {targets.size:,} landmarks, "
        f"{len(groups):,} subjects, prevalence {targets.mean():.4%}\n"
    )

    left, right = binary_metrics(ensemble, targets), binary_metrics(tree, targets)
    report = {
        "fold": args.fold,
        "n_members": len(members),
        "members": [str(path.relative_to(ROOT)) for path in members],
        "baseline": str(baseline.relative_to(ROOT)),
        "resamples": args.resamples,
        "seed": args.seed,
        "ensemble": {key: float(value) for key, value in left.items()},
        "xgboost600": {key: float(value) for key, value in right.items()},
    }

    print(f"{'metric':<20} {'ensemble':>10} {'xgboost600':>11} {'difference':>12}")
    for key in REPORTED:
        print(f"{key:<20} {left[key]:>10.5f} {right[key]:>11.5f}")

    rng = np.random.default_rng(args.seed)
    draws: dict[str, list[float]] = {key: [] for key in REPORTED}
    for index in range(args.resamples):
        rows = _draw_rows(groups, rng)
        a = binary_metrics(ensemble[rows], targets[rows])
        b = binary_metrics(tree[rows], targets[rows])
        for key in REPORTED:
            draws[key].append(a[key] - b[key])
        if (index + 1) % 250 == 0:
            print(f"  draw {index + 1}/{args.resamples}", flush=True)

    report["paired"] = {}
    print(f"\npaired 95% intervals, {args.resamples} subject resamples")
    for key in REPORTED:
        values = np.array(draws[key])
        observed = left[key] - right[key]
        low, high = np.quantile(values, [0.025, 0.975])
        clears = "*" if low > 0 or high < 0 else " "
        wins = (values < 0) if key in LOWER_IS_BETTER else (values > 0)
        report["paired"][key] = {
            "value": float(observed),
            "lo": float(low),
            "hi": float(high),
            "p_positive": float((values > 0).mean()),
            "p_favours_ensemble": float(wins.mean()),
            "lower_is_better": key in LOWER_IS_BETTER,
        }
        print(
            f" {clears} {key:<20} {observed:+.5f} [{low:+.5f}, {high:+.5f}]  "
            f"{wins.mean():>6.1%} of draws favour the ensemble"
        )
    print("\n  * marks an interval that excludes zero")

    if args.dest:
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        args.dest.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.dest}")


if __name__ == "__main__":
    main()
