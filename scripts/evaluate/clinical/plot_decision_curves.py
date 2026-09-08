"""Render the decision curve figures for the report.

Run from the repo root with the modelling interpreter, after `decision_curves.py`
has written its report:

    .venv-modelling/bin/python scripts/evaluate/clinical/plot_decision_curves.py \
        --report motor_output/comparison/decision_curves.json \
        --dest-dir motor_output/figures

Writes PNG at 600 dpi, which LaTeX's graphicx takes directly. A vector format
would stay sharp at any magnification and would be smaller for line art, but raster
was asked for, and 600 dpi is the density at which line plots stop showing stair
steps in print.

This driver only loads, composes and saves. Every number it draws was computed and
asserted by `ensemble/decision_curve.py`, and the composition lives in
`ensemble/graphs/decision_curves.py` -- so a figure can be wrong about how it renders,
never about what it renders.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from thesis.modelling.figures.decision_curves import (  # noqa: E402
    differences_figure,
    panels_figure,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = ROOT / "motor_output" / "comparison" / "decision_curves.json"


def _parse_args() -> argparse.Namespace:
    """Reads the report to render and where the images should go."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--dest-dir", type=Path, default=ROOT / "motor_output" / "figures"
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--format", default="png", help="any raster format matplotlib writes"
    )
    parser.add_argument(
        "--band",
        type=float,
        nargs=2,
        default=(0.05, 0.15),
        help="the clinically defensible threshold window to shade",
    )
    return parser.parse_args()


def main() -> None:
    """Builds both figures and writes them beside each other."""
    args = _parse_args()
    if not args.report.is_file():
        raise SystemExit(
            f"{args.report} is missing; run "
            "scripts/evaluate/clinical/decision_curves.py first."
        )

    report = json.loads(args.report.read_text())
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "decision_curves": panels_figure(report, band=tuple(args.band)),
    }
    if report.get("differences"):
        figures["decision_curve_differences"] = differences_figure(report)
    else:
        print("report holds no differences block; skipping that figure")

    for name, figure in figures.items():
        path = args.dest_dir / f"{name}.{args.format}"
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
        size = path.stat().st_size / 1024
        print(f"wrote {path}  ({size:,.0f} KiB, {args.dpi} dpi)")


if __name__ == "__main__":
    main()
