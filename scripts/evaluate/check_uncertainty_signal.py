"""Does epistemic uncertainty carry anything the predicted risk does not?

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/evaluate/check_uncertainty_signal.py \
        --runs motor_output/runs/lora-cfg-* motor_output/runs/lora-sched-seed1 \
        --stems step_014000 last \
        --dest motor_output/comparison/uncertainty_control.json

This is the control that decides whether the uncertainty work is worth doing. It seeks
to answer:

1. How much of the epistemic column is a function of the prediction alone?
2. At a fixed predicted risk, does the residual predict anything?

At ten bands the halves of the top band differ in predicted risk by 0.059 and the
apparent effect is roughly four times the real one; at two hundred the leak falls to
0.0002. The reported gap is the average over bands, with a subject-level bootstrap
around it, split into the high-risk region and the rest because the effect is not
uniform.

Needs no GPU: it reads the prediction bundles `score_checkpoints.py` already banked.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from thesis.modelling.ensemble.diversity import (
    align_members,
    checkpoint_bundles,
    load_member,
)
from thesis.modelling.ensemble.uncertainty import decompose, explained_by_bins

# the subject-level draw already exists for the XGBoost comparison; a second copy here
# is exactly the metric drift the project keeps one of everything to avoid
from thesis.modelling.evaluation.intervals import _draw_rows, _subject_groups

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ("mean", "total", "aleatoric", "epistemic", "variance")


def _parse_args() -> argparse.Namespace:
    """Reads the member set, the binning and the resampling budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--stems",
        nargs="+",
        default=["step_014000", "last"],
        help="which checkpoints each run contributes",
    )
    parser.add_argument(
        "--bands", type=int, default=200, help="bands of predicted risk for the gap"
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.1,
        help="the high-risk share reported separately",
    )
    parser.add_argument("--deciles", type=int, default=10, help="bands in the table")
    parser.add_argument("--bins", type=int, default=100, help="bins for explained")
    parser.add_argument(
        "--resamples",
        type=int,
        default=2000,
        help="subject draws; 0 skips the interval",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dest", type=Path, default=None)
    return parser.parse_args()


def member_paths(runs: list[Path], stems: list[str]) -> list[Path]:
    """One bundle per run per stem, falling back to `last` for a short schedule."""
    paths = []
    for run in runs:
        available = dict(checkpoint_bundles(run))
        for stem in stems:
            paths.append(available.get(stem) or available["last"])
    return paths


def band_gaps(
    rows: np.ndarray,
    mean: np.ndarray,
    epistemic: np.ndarray,
    residual: np.ndarray,
    *,
    bands: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-band difference between the contested and the agreed half.

    Args:
        rows (np.ndarray): The row indices to use, so a bootstrap draw can pass its own.
        mean (np.ndarray): The ensemble's prediction.
        epistemic (np.ndarray): The epistemic column.
        residual (np.ndarray): `target - prediction`, positive where under-predicted.
        bands (int): How many equal-mass bands of prediction to cut.

    Returns:
        tuple[np.ndarray, np.ndarray]: The residual gap per band, and the leftover
            prediction gap per band, which should be near zero.
    """
    ordered = rows[np.argsort(mean[rows], kind="stable")]
    gaps, leaks = [], []
    for band in np.array_split(ordered, bands):
        values = epistemic[band]
        cut = np.median(values)
        high, low = band[values > cut], band[values <= cut]
        if high.size and low.size:
            gaps.append(residual[high].mean() - residual[low].mean())
            leaks.append(mean[high].mean() - mean[low].mean())
    return np.array(gaps), np.array(leaks)


def regions(gaps: np.ndarray, top: int) -> dict[str, float]:
    """The pooled gap over every band, the highest-risk bands, and the rest."""
    return {
        "all": float(gaps.mean()),
        "high_risk": float(gaps[-top:].mean()),
        "rest": float(gaps[:-top].mean()),
    }


def band_table(
    epistemic: np.ndarray, mean: np.ndarray, targets: np.ndarray, *, deciles: int
) -> list[dict[str, float]]:
    """The readable version of the same split, one row per band."""
    order = np.argsort(mean, kind="stable")
    rows = []
    for index, band in enumerate(np.array_split(order, deciles)):
        values = epistemic[band]
        cut = np.median(values)
        high, low = band[values > cut], band[values <= cut]
        if not high.size or not low.size:
            continue
        rows.append(
            {
                "band": index,
                "n": int(band.size),
                "prediction_high": float(mean[high].mean()),
                "prediction_low": float(mean[low].mean()),
                "epistemic_mean": float(values.mean()),
                "rate_high": float(targets[high].mean()),
                "rate_low": float(targets[low].mean()),
            }
        )
    return rows


def main() -> None:
    """Decomposes the ensemble and reports whether the epistemic column is redundant."""
    args = _parse_args()

    paths = member_paths(args.runs, args.stems)
    print(f"{len(paths)} members")
    scores, targets, subjects = align_members([load_member(path) for path in paths])
    parts = decompose(scores)
    groups = _subject_groups(subjects)
    print(
        f"{targets.size:,} landmarks, {len(groups):,} subjects, "
        f"base rate {targets.mean():.4%}\n",
        flush=True,
    )

    for name in COLUMNS:
        column = getattr(parts, name)
        print(
            f"  {name:10s} mean {column.mean():.6f}  sd {column.std():.6f}  "
            f"min {column.min():.6f}  max {column.max():.6f}"
        )

    fraction = float((parts.epistemic / np.maximum(parts.total, 1e-12)).mean())
    explained = explained_by_bins(parts.epistemic, parts.mean, bins=args.bins)
    pearson = float(np.corrcoef(parts.mean, parts.epistemic)[0, 1])
    spearman = float(spearmanr(parts.mean, parts.epistemic).statistic)

    print(f"\nepistemic / total, mean over landmarks: {fraction:.4f}")
    print(
        f"correlation with the prediction: pearson {pearson:.4f}  "
        f"spearman {spearman:.4f}"
    )
    print(
        f"variance of epistemic explained by a {args.bins}-bin function of the "
        f"prediction: {explained:.4f}"
    )
    print(f"  -> {1 - explained:.4f} of it is NOT a function of the prediction")

    table = band_table(parts.epistemic, parts.mean, targets, deciles=args.deciles)
    print(
        f"\n{args.deciles} bands of predicted risk, split at the median epistemic:\n"
        f"{'band':>4} {'n':>8} {'pred hi':>8} {'pred lo':>8} {'epi mean':>9} "
        f"{'rate hi':>8} {'rate lo':>8}"
    )
    for row in table:
        print(
            f"{row['band']:>4} {row['n']:>8,} {row['prediction_high']:>8.4f} "
            f"{row['prediction_low']:>8.4f} {row['epistemic_mean']:>9.5f} "
            f"{row['rate_high']:>8.4f} {row['rate_low']:>8.4f}"
        )

    residual = targets - parts.mean
    top = max(1, round(args.top_fraction * args.bands))
    gaps, leaks = band_gaps(
        np.arange(targets.size),
        parts.mean,
        parts.epistemic,
        residual,
        bands=args.bands,
    )
    observed = regions(gaps, top)
    leak = regions(leaks, top)
    print(f"\nresidual gap over {args.bands} bands, contested half minus agreed half")
    for key, value in observed.items():
        print(f"  {key:10s} {value:+.5f}   (leftover prediction gap {leak[key]:+.6f})")

    report = {
        "members": [str(path) for path in paths],
        "n_labels": int(targets.size),
        "n_subjects": len(groups),
        "bands": args.bands,
        "columns": {
            name: {
                "mean": float(getattr(parts, name).mean()),
                "sd": float(getattr(parts, name).std()),
            }
            for name in COLUMNS
        },
        "epistemic_fraction_of_total": fraction,
        "pearson_with_prediction": pearson,
        "spearman_with_prediction": spearman,
        "explained_by_prediction": explained,
        "band_table": table,
        "residual_gap": {key: {"value": value} for key, value in observed.items()},
        "prediction_gap": leak,
    }

    if args.resamples:
        rng = np.random.default_rng(args.seed)
        draws: dict[str, list[float]] = {key: [] for key in observed}
        for index in range(args.resamples):
            drawn, _ = band_gaps(
                _draw_rows(groups, rng),
                parts.mean,
                parts.epistemic,
                residual,
                bands=args.bands,
            )
            for key, value in regions(drawn, top).items():
                draws[key].append(value)
            if (index + 1) % 200 == 0:
                print(f"  draw {index + 1}/{args.resamples}", flush=True)

        print(f"\npaired 95% intervals, {args.resamples} subject resamples")
        for key, value in observed.items():
            values = np.array(draws[key])
            low, high = np.quantile(values, [0.025, 0.975])
            report["residual_gap"][key] |= {
                "lo": float(low),
                "hi": float(high),
                "p_positive": float((values > 0).mean()),
            }
            print(
                f"  {key:10s} {value:+.5f}  [{low:+.5f}, {high:+.5f}]  "
                f"positive in {(values > 0).mean():.1%} of draws"
            )

    if args.dest:
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        args.dest.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.dest}")


if __name__ == "__main__":
    main()
