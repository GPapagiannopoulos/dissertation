"""Per-arm subject-level bootstrap intervals on the TEST fold.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/evaluate/comparison/test_fold_intervals.py

Writes `motor_output/comparison/newgrid/test_fold_intervals.json`, which
`build_results_summary.py` reads to render its test-fold table.

Every metric is collected on each draw, so five intervals cost one pass rather than
five. The RNG construction and draw order match `bootstrap_interval` exactly, so a
single-metric rerun through the library reproduces these numbers.

Selects nothing: every checkpoint stem comes from `roster.by_loss_stem`, which reads
the validation ladder. Needs no GPU. About 15 minutes per arm at 2,000 resamples.
"""

import json
import time
from pathlib import Path

import numpy as np

from thesis.modelling.ensemble import roster as R
from thesis.modelling.ensemble.diversity import align_members, load_member
from thesis.modelling.evaluation import intervals as iv
from thesis.modelling.evaluation.metrics import binary_metrics

FOLD, RESAMPLES, SEED, ALPHA = "testing", 2000, 0, 0.05
METRICS = ("auprc", "auroc", "brier", "ece", "precision_at_1pct")
DEST = R.NEWGRID / "test_fold_intervals.json"


def arms() -> dict[str, list[Path]]:
    """Every test-fold arm, as the banked bundles that make it up.

    Returns:
        dict[str, list[Path]]: Arm label to its members.
    """
    lora23 = R.lora_ensemble(FOLD)
    lora_last = [
        dict(R.checkpoint_bundles(R.RUNS / run, R.SELECTION[FOLD]))["last"]
        for run in R.LORA_RUNS
    ]
    mono_seeds = [
        p
        for run in R.MONOLITHIC_RUNS
        for p in R.monolithic_bundles(run, R.MONOLITHIC_WINDOW, FOLD)
    ]
    mono_snap = R.monolithic_bundles(R.MONOLITHIC_HEADLINE, R.MONOLITHIC_WINDOW, FOLD)
    mono_one = [
        dict(R.checkpoint_bundles(R.RUNS / R.MONOLITHIC_HEADLINE, R.SELECTION[FOLD]))[
            R.by_loss_stem(R.MONOLITHIC_HEADLINE)
        ]
    ]
    return {
        "LoRA ensemble, 23 (by-loss + last)": lora23,
        "LoRA ensemble, 12 (all last)": lora_last,
        "MOTOR monolithic, 3 seeds x pre-collapse": mono_seeds,
        "MOTOR monolithic, 1 run x pre-collapse": mono_snap,
        "MOTOR monolithic, single checkpoint": mono_one,
        "XGBoost, 600 rounds": [R.baseline_bundle(FOLD)],
    }


def scores_of(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Averages one arm's members onto a common row order.

    Args:
        paths (list[Path]): The arm's banked bundles.

    Returns:
        tuple: The arm's scores, the shared targets, and the shared subjects.
    """
    members = R.drop_contested([load_member(p) for p in paths])
    if len(members) == 1:
        members = members * 2
    S, y, subj = align_members(members)
    return S.mean(axis=0), y.astype(float), subj


def interval(
    scores: np.ndarray, targets: np.ndarray, subjects: np.ndarray
) -> dict[str, dict[str, float]]:
    """Subject-level bootstrap intervals for every metric, in one pass.

    Args:
        scores (np.ndarray): The arm's predicted probabilities.
        targets (np.ndarray): The binary labels.
        subjects (np.ndarray): The subject each row belongs to.

    Returns:
        dict[str, dict[str, float]]: Metric to its bounds and surviving draw count.
    """
    groups = iv._subject_groups(subjects)
    rng = np.random.default_rng(SEED)
    draws = {m: [] for m in METRICS}
    for _ in range(RESAMPLES):
        _r = iv._draw_rows(groups, rng)
        got = binary_metrics(scores[_r], targets[_r])
        for m in METRICS:
            if not np.isnan(got[m]):
                draws[m].append(got[m])
    return {
        m: {
            "lo": float(np.quantile(draws[m], ALPHA / 2)),
            "hi": float(np.quantile(draws[m], 1 - ALPHA / 2)),
            "n_draws": len(draws[m]),
        }
        for m in METRICS
    }


def main() -> None:
    """Scores every arm and writes the report."""
    out: dict[str, object] = {
        "fold": FOLD,
        "resamples": RESAMPLES,
        "seed": SEED,
        "alpha": ALPHA,
        "arms": {},
    }
    for label, paths in arms().items():
        t = time.perf_counter()
        s, y, subj = scores_of(paths)
        point = binary_metrics(s, y)
        band = interval(s, y, subj)
        out["arms"][label] = {
            "n_members": len(paths),
            "n_labels": int(y.size),
            "n_subjects": int(np.unique(subj).size),
            "point": point,
            "interval": band,
        }
        print(
            f"{label:44} AUPRC {point['auprc']:.5f} "
            f"[{band['auprc']['lo']:.5f}, {band['auprc']['hi']:.5f}] "
            f"({time.perf_counter() - t:.0f}s)",
            flush=True,
        )
    DEST.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {DEST}")


if __name__ == "__main__":
    main()
