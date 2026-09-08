"""Renders the two-panel ensemble-size saturation figure as PNG for LaTeX.

Panel A is ensemble AUPRC against the number of training runs, panel B is net
benefit at a chosen threshold per 1,000 landmarks against the same axis. Both
have a min-max band that shows the range of values for that ensemble size.

Writes to `motor_output/figures/saturation_test/` by default.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/evaluate/plot_saturation.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as pyplot  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from thesis.modelling.figures.decision_curves import PALETTE  # noqa: E402
from thesis.modelling.figures.saturation import (  # noqa: E402
    PER,
    panel_note,
    saturation_axes,
)

NEWGRID = ROOT / "motor_output" / "comparison" / "newgrid"


def _parse_args() -> argparse.Namespace:
    """Command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explore", type=Path, default=NEWGRID / "explore_test.json")
    parser.add_argument(
        "--curves", type=Path, default=NEWGRID / "decision_curves_test.json"
    )
    parser.add_argument(
        "--members",
        choices=("by_loss", "by_loss_plus_last"),
        default="by_loss",
        help="Which panel-A sweep to draw. Panel B is always two members per run.",
    )
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / "motor_output" / "figures" / "saturation_test",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def net_benefit_rows(report: dict, threshold: float) -> list[dict]:
    """Panel B's rows, read at one threshold off the banked subset curves.

    Args:
        report (dict): The parsed decision-curve report.
        threshold (float): Which reported threshold to read.

    Returns:
        list[dict]: One `n_runs` / `mean` / `low` / `high` entry per size.

    Raises:
        ValueError: If the threshold is not one the report banked.
    """
    key = f"{threshold:g}"
    rows = []
    for size, entry in report["subset_curves"].items():
        at = entry["at_reported"]
        match = at.get(key) or at.get(f"{threshold:.2f}") or at.get(str(threshold))
        if match is None:
            raise ValueError(
                f"threshold {threshold} is not banked; the report holds "
                f"{', '.join(sorted(at))}."
            )
        rows.append(
            {
                "n_runs": int(size),
                "mean": match["mean"],
                "low": match["low"],
                "high": match["high"],
                "n_subsets": entry["n_subsets"],
            }
        )
    return rows


def main() -> None:
    """Composes the figure and writes it."""
    args = _parse_args()
    explore = json.loads(args.explore.read_text())
    curves = json.loads(args.curves.read_text())

    rows_a = explore["saturation"][args.members]
    rows_b = net_benefit_rows(curves, args.threshold)
    args.dest.mkdir(parents=True, exist_ok=True)

    figure, (left, right) = pyplot.subplots(1, 2, figsize=(10.0, 4.2))
    saturation_axes(
        left,
        rows_a,
        mean_key="mean_auprc",
        low_key="min_auprc",
        high_key="max_auprc",
        colour=PALETTE[1],
    )
    left.set_ylabel("ensemble AUPRC")
    left.set_title("A. discrimination", loc="left", fontsize=10)
    per_run = 1 if args.members == "by_loss" else 2
    panel_note(
        left,
        f"{per_run} checkpoint{'s' if per_run > 1 else ''} per run\n"
        f"{min(r['n_subsets'] for r in rows_a)}–{max(r['n_subsets'] for r in rows_a)}"
        " subsets per point",
    )

    saturation_axes(
        right,
        rows_b,
        mean_key="mean",
        low_key="low",
        high_key="high",
        scale=PER,
        colour=PALETTE[0],
    )
    right.set_ylabel(f"net benefit per 1,000, at t = {args.threshold:g}")
    right.set_title("B. clinical utility", loc="left", fontsize=10)
    panel_note(
        right,
        "2 checkpoints per run\n"
        f"{min(r['n_subsets'] for r in rows_b)}–{max(r['n_subsets'] for r in rows_b)}"
        " subsets per point",
    )

    fold = curves.get("fold", "unknown")
    figure.suptitle(
        f"Ensemble size against budget — {fold} fold, "
        f"{curves['n_labels']:,} landmarks. "
        "Band is the min–max across which runs were included, not across patients.",
        fontsize=8.5,
        y=0.02,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    out = args.dest / f"saturation_{args.members}_t{args.threshold:g}.png"
    figure.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
