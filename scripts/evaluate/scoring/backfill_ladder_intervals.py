r"""Add the missing metric intervals to a run's banked checkpoint ladder.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python \
        scripts/evaluate/scoring/backfill_ladder_intervals.py \
        --runs motor_output/runs/ng-* \
        --selection selection_test
"""

import argparse
import json
from pathlib import Path

import numpy as np

from thesis.modelling.evaluation.intervals import _draw_rows, _subject_groups
from thesis.modelling.evaluation.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[3]
REPORTED = ("auprc", "auroc", "brier", "ece", "precision_at_1pct")
# the band already in the file, and so the one that pins the draw construction
WITNESS = "auprc"
# four decimals: the bands are quantiles of a few hundred draws, and float noise in
# the last bits of a quantile is not a construction mismatch
TOLERANCE = 1e-4


def _parse_args() -> argparse.Namespace:
    """Reads the run folders, which selection directory, and the draw budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--selection", default="selection_test")
    parser.add_argument("--resamples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def default_resamples() -> int:
    """Retrieves the resamples `score_checkpoints.py` banked with."""
    import importlib.util

    path = ROOT / "scripts" / "evaluate" / "scoring" / "score_checkpoints.py"
    spec = importlib.util.spec_from_file_location("score_checkpoints", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.INTERVAL_RESAMPLES)


def all_intervals(
    scores: np.ndarray,
    targets: np.ndarray,
    subjects: np.ndarray,
    *,
    resamples: int,
    seed: int,
    alpha: float,
) -> dict[str, tuple[float, float]]:
    """Calculate the interval of all reported metrics on shared subjects.

    Args:
        scores (np.ndarray): Predicted probabilities.
        targets (np.ndarray): Binary labels.
        subjects (np.ndarray): The subject each row belongs to.
        resamples (int): How many bootstrap draws.
        seed (int): Seeds the draws.
        alpha (float): Two-sided width, so 0.05 gives a 95% interval.

    Returns:
        dict[str, tuple[float, float]]: Lower and upper bound per metric.
    """
    groups = _subject_groups(subjects)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {metric: [] for metric in REPORTED}
    for _ in range(resamples):
        rows = _draw_rows(groups, rng)
        got = binary_metrics(scores[rows], targets[rows])
        for metric in REPORTED:
            # NaN draws are dropped, not zero-filled, exactly as bootstrap_interval
            if not np.isnan(got[metric]):
                draws[metric].append(got[metric])
    return {
        metric: (
            float(np.quantile(values, alpha / 2)),
            float(np.quantile(values, 1 - alpha / 2)),
        )
        for metric, values in draws.items()
    }


def check_witness(row: dict, bands: dict[str, tuple[float, float]]) -> None:
    """Validates calculation by comparing the recomputed AUPRC with the banked one.

    Args:
        row (dict): One ladder row, carrying `auprc_lo` and `auprc_hi`.
        bands (dict[str, tuple[float, float]]): The freshly computed intervals.

    Raises:
        ValueError: If the two differ by more than `TOLERANCE`, which means the draws
            here are not `bootstrap_interval`'s and no band may be written.
    """
    if f"{WITNESS}_lo" not in row:
        return
    for bound, fresh in zip(("lo", "hi"), bands[WITNESS], strict=True):
        banked = row[f"{WITNESS}_{bound}"]
        if abs(banked - fresh) > TOLERANCE:
            raise ValueError(
                f"{row['checkpoint']}: recomputed {WITNESS}_{bound} {fresh:.6f} "
                f"against the banked {banked:.6f}. The draw construction differs, "
                f"so the backfilled bands would not match the existing ones."
            )


def backfill(run: Path, selection: str, **draw: float) -> int:
    """Rewrites one run's ranking file with every metric's interval.

    Args:
        run (Path): A run folder under `motor_output/runs`.
        selection (str): Which selection directory, e.g. `selection_test`.
        **draw (float): `resamples`, `seed` and `alpha`, passed to `all_intervals`.

    Returns:
        int: How many rows were filled; zero if the run has no ladder or is done.
    """
    ranking = run / selection / "checkpoint_ranking.json"
    if not ranking.is_file():
        return 0
    rows = json.loads(ranking.read_text())
    pending = [row for row in rows if f"{REPORTED[1]}_lo" not in row]
    if not pending:
        return 0

    for row in pending:
        bundle = run / selection / f"{Path(row['checkpoint']).stem}_predictions.npz"
        with np.load(bundle) as loaded:
            scores, targets, subjects = (loaded[key] for key in loaded.files[:3])
        bands = all_intervals(scores, targets.astype(float), subjects, **draw)
        check_witness(row, bands)
        for metric, (low, high) in bands.items():
            row[f"{metric}_lo"], row[f"{metric}_hi"] = low, high
        print(f"  {run.name}/{row['checkpoint']}", flush=True)

    ranking.write_text(json.dumps(rows, indent=1))
    return len(pending)


def main() -> None:
    """Backfills every named run, reporting what each one needed."""
    args = _parse_args()
    resamples = args.resamples or default_resamples()
    print(f"backfilling {len(REPORTED)} intervals at {resamples:,} draws", flush=True)

    filled = 0
    for run in sorted(args.runs):
        filled += backfill(
            run,
            args.selection,
            resamples=resamples,
            seed=args.seed,
            alpha=args.alpha,
        )
    print(f"\nfilled {filled} ladder rows")


if __name__ == "__main__":
    main()
