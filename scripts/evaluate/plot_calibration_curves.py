"""Render the calibration figure for the report.

Run from the repo root with the modelling interpreter, after `calibration_curves.py`
has written its report:

    .venv-modelling/bin/python scripts/evaluate/plot_calibration_curves.py \
        --report motor_output/comparison/newgrid/calibration_test.json \
        --dest-dir motor_output/figures

Writes PNG at 600 dpi, matching `plot_decision_curves.py`.

`--arms` exists because six overlapping reliability curves are unreadable. The default
draws the three the report's calibration claim is about; pass more to inspect, fewer
to publish.

This driver only loads, composes and saves. Every number it draws was computed and
asserted by `evaluation/metrics.py`, and the composition lives in
`modelling/figures/calibration.py` -- so a figure can be wrong about how it renders,
never about what it renders.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from thesis.modelling.figures.calibration import calibration_figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NEWGRID = ROOT / "motor_output" / "comparison" / "newgrid"
DEFAULT_REPORT = NEWGRID / "calibration_test.json"

# the three arms the calibration claim compares: the ensemble, the tree it is measured
# against, and the monolithic arm it replaces
DEFAULT_ARMS = ("xgboost", "lora_last2", "monolithic_seeds")


def _parse_args() -> argparse.Namespace:
    """Reads the report to render and where the image should go."""
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
        "--arms",
        nargs="+",
        default=DEFAULT_ARMS,
        help="which arms to draw; pass 'all' for every arm in the report",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="output stem, defaulting to the report's fold",
    )
    return parser.parse_args()


def main() -> None:
    """Builds the figure and writes it beside the decision curves."""
    args = _parse_args()
    if not args.report.is_file():
        raise SystemExit(
            f"{args.report} is missing; run "
            f"scripts/evaluate/calibration_curves.py first."
        )

    report = json.loads(args.report.read_text())
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    arms = None if tuple(args.arms) == ("all",) else tuple(args.arms)
    figure = calibration_figure(report, arms=arms)

    stem = args.name or f"calibration_{report.get('fold', 'fold')}"
    path = args.dest_dir / f"{stem}.{args.format}"
    figure.savefig(path, dpi=args.dpi, bbox_inches="tight")
    size = path.stat().st_size / 1024
    print(f"wrote {path}  ({size:,.0f} KiB, {args.dpi} dpi)")


if __name__ == "__main__":
    main()
