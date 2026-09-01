"""Reliability curves for every arm, with subject-level bands.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/evaluate/calibration_curves.py \
        --fold testing \
        --dest motor_output/comparison/newgrid/calibration_test.json

ECE is one number, and one number is weak evidence for a calibration claim: two arms
can miss by the same total while one is confidently wrong at the top of the ranking
and the other is diffusely wrong among rows nobody acts on. This resolves the same
quantity by bin, so the claim can name WHERE each arm misses.

The curve is the decomposition of the reported ECE, not a second opinion on it. Bins
are cut equal-COUNT on each arm's own scores, exactly as
`expected_calibration_error` cuts them, and the driver asserts that summing the curve
reproduces the arm's reported ECE. Equal-width bins would be the conventional
presentation and are wrong here: at 3.4% prevalence they put over 99% of rows in the
first bin.

Bin edges are cut ONCE on the full fold and held fixed across bootstrap draws. A draw
that recuts its own quantiles moves the bins as well as the rates, and the band then
measures bin drift rather than uncertainty about calibration.

Members and the arm rosters come from `ensemble.roster`, so this cannot drift from the
decision-curve or pairing analyses. Needs no GPU: it reads banked predictions.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from thesis.modelling.ensemble.diversity import align_members, load_member
from thesis.modelling.ensemble.roster import (
    MONOLITHIC_WINDOW,
    ROOT,
    arm_definitions,
    drop_contested,
)
from thesis.modelling.evaluation.intervals import _draw_rows, _subject_groups
from thesis.modelling.evaluation.metrics import (
    calibration_edges,
    expected_calibration_error,
    flexible_calibration_curve,
)

COMPARISON = ROOT / "motor_output" / "comparison"

# the arm whose alert threshold marks the figures. Calibration below it is never acted
# on, so an arm may miss badly there and still deploy well
BUDGET_ARM = "lora_last2"
ALERT_BUDGET = 0.01


def _parse_args() -> argparse.Namespace:
    """Reads the fold, the bin count and the resampling budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="equal-count bins per arm. Ten is what the reported ECE uses; more "
        "bins resolve the top of the ranking better and are noisier per bin",
    )
    parser.add_argument(
        "--fold",
        default="validation",
        help="which fold's banked bundles to read; selection stays on validation",
    )
    parser.add_argument(
        "--monolithic-window",
        type=int,
        nargs=2,
        default=MONOLITHIC_WINDOW,
        metavar=("FIRST", "LAST"),
        help="inclusive step window the monolithic snapshot arms average over",
    )
    parser.add_argument(
        "--resamples", type=int, default=2000, help="subject draws; 0 skips bands"
    )
    parser.add_argument(
        "--flexible-knots",
        type=int,
        default=5,
        help="spline knots for the flexible (Van Calster moderate) curve; 0 skips it",
    )
    parser.add_argument(
        "--flexible-grid",
        type=int,
        default=100,
        help="points the smooth curve is drawn at",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Builds one curve per arm and, unless disabled, its bootstrap band."""
    args = _parse_args()

    definitions = arm_definitions(tuple(args.monolithic_window), args.fold)
    flat = [path for paths in definitions.values() for path in paths]
    stacked, targets, subjects = align_members(
        drop_contested([load_member(path) for path in flat])
    )
    targets = targets.astype(float)
    groups = _subject_groups(subjects)

    arms, cursor = {}, 0
    for name, paths in definitions.items():
        arms[name] = stacked[cursor : cursor + len(paths)].mean(axis=0)
        cursor += len(paths)

    prevalence = float(targets.mean())
    print(
        f"{targets.size:,} landmarks, {len(groups):,} subjects, "
        f"prevalence {prevalence:.4%}, {args.bins} equal-count bins\n"
    )

    # each arm is binned on ITS OWN quantiles, because that is what its reported ECE
    # is cut on; a shared grid would make the curves line up on the page while no
    # longer summing to any number the project reports
    indices, edges, report_arms = {}, {}, {}
    for name, scores in arms.items():
        bounds = calibration_edges(scores, bins=args.bins)
        edges[name] = bounds
        indices[name] = np.digitize(scores, bounds[1:-1], right=True)

    def curve(rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Predicted and observed rates per bin, over all rows or one draw."""
        built = {}
        for name, scores in arms.items():
            index = indices[name] if rows is None else indices[name][rows]
            drawn_scores = scores if rows is None else scores[rows]
            drawn_targets = targets if rows is None else targets[rows]
            width = edges[name].size - 1
            count = np.bincount(index, minlength=width).astype(float)
            filled = count > 0
            safe = np.where(filled, count, 1.0)
            built[name] = (
                np.where(
                    filled,
                    np.bincount(index, weights=drawn_scores, minlength=width) / safe,
                    np.nan,
                ),
                np.where(
                    filled,
                    np.bincount(index, weights=drawn_targets, minlength=width) / safe,
                    np.nan,
                ),
                count,
            )
        return built

    observed = curve()
    budget_threshold = float(np.quantile(arms[BUDGET_ARM], 1.0 - ALERT_BUDGET))

    print(f"{'arm':22s} {'bins':>4} {'ece':>8} {'curve':>8}  worst bin")
    for name, (predicted, rate, count) in observed.items():
        # the curve IS the ECE decomposition; if this disagrees the two are binning
        # differently and neither the figure nor the number can be trusted
        from_curve = float(np.nansum(count * np.abs(predicted - rate)) / count.sum())
        reported = expected_calibration_error(arms[name], targets, bins=args.bins)
        if not np.isclose(from_curve, reported, atol=1e-9):
            raise SystemExit(
                f"{name}: the curve sums to ECE {from_curve:.9f} against the reported "
                f"{reported:.9f}. The bins disagree; do not report either."
            )
        worst = int(np.nanargmax(np.abs(rate - predicted)))
        report_arms[name] = {
            "members": [str(path) for path in definitions[name]],
            "n_members": len(definitions[name]),
            "edges": edges[name].tolist(),
            "mean_score": predicted.tolist(),
            "observed_rate": rate.tolist(),
            "count": count.astype(int).tolist(),
            "ece": reported,
        }
        print(
            f"{name:22s} {count.size:>4} {reported:>8.5f} {from_curve:>8.5f}  "
            f"bin {worst + 1}: predicted {predicted[worst]:.4f}, "
            f"observed {rate[worst]:.4f}"
        )

    flexible_grid: dict[str, np.ndarray] = {}
    if args.flexible_knots:
        print(
            f"\nflexible calibration curve, {args.flexible_knots}-knot spline in the "
            "logit (Van Calster's MODERATE level)"
        )
        print(f"{'arm':22s} {'intercept':>10} {'slope':>8}  reading")
        for name, scores in arms.items():
            fitted = flexible_calibration_curve(
                scores, targets, knots=args.flexible_knots, grid=args.flexible_grid
            )
            flexible_grid[name] = fitted.grid_score
            report_arms[name]["flexible_grid"] = fitted.grid_score.tolist()
            report_arms[name]["flexible_rate"] = fitted.fitted_rate.tolist()
            report_arms[name]["calibration_intercept"] = fitted.intercept
            report_arms[name]["calibration_slope"] = fitted.slope
            reading = (
                "too extreme"
                if fitted.slope < 0.95
                else "too flat"
                if fitted.slope > 1.05
                else "weak-calibrated"
            )
            print(
                f"{name:22s} {fitted.intercept:>+10.4f} {fitted.slope:>8.4f}  {reading}"
            )

    if args.resamples:
        rng = np.random.default_rng(args.seed)
        draws = {name: [] for name in arms}
        flexible_draws: dict[str, list[np.ndarray]] = {name: [] for name in arms}
        for step in range(args.resamples):
            rows = _draw_rows(groups, rng)
            drawn = curve(rows)
            for name, (_, rate, _) in drawn.items():
                draws[name].append(rate)
            for name, scores in arms.items() if flexible_grid else ():
                # the grid is the FULL-FOLD one, held fixed: a draw that re-quantiles
                # its own abscissae reports each curve at different x and the band
                # then measures where the grid moved, not where the curve did
                flexible_draws[name].append(
                    flexible_calibration_curve(
                        scores[rows],
                        targets[rows],
                        knots=args.flexible_knots,
                        grid_scores=flexible_grid[name],
                    ).fitted_rate
                )
            if (step + 1) % 200 == 0:
                print(f"  draw {step + 1}/{args.resamples}", flush=True)

        print(f"\nobserved rate, 95% band over {args.resamples} subject resamples")
        for name in arms:
            values = np.stack(draws[name])
            low, high = np.nanquantile(values, [0.025, 0.975], axis=0)
            report_arms[name]["observed_lo"] = low.tolist()
            report_arms[name]["observed_hi"] = high.tolist()
            predicted = np.asarray(report_arms[name]["mean_score"])
            rate = np.asarray(report_arms[name]["observed_rate"])
            # a bin whose band excludes its own predicted probability is miscalibrated
            # by more than the cohort's own sampling noise explains
            misses = int(np.sum((predicted < low) | (predicted > high)))
            print(
                f"  {name:22s} {misses}/{predicted.size} bins exclude their "
                f"own predicted rate"
            )
            if flexible_grid:
                smooth = np.stack(flexible_draws[name])
                flow, fhigh = np.nanquantile(smooth, [0.025, 0.975], axis=0)
                report_arms[name]["flexible_lo"] = flow.tolist()
                report_arms[name]["flexible_hi"] = fhigh.tolist()
                grid = flexible_grid[name]
                off = int(np.sum((grid < flow) | (grid > fhigh)))
                print(
                    f"  {name:22s} {off}/{grid.size} grid points where the smooth "
                    f"band excludes the diagonal"
                )

    report = {
        "fold": args.fold,
        "bins": args.bins,
        "n_labels": int(targets.size),
        "n_subjects": len(groups),
        "prevalence": prevalence,
        "resamples": args.resamples,
        "seed": args.seed,
        "monolithic_window": list(args.monolithic_window),
        "flexible_knots": args.flexible_knots,
        "alert_budget": ALERT_BUDGET,
        "budget_arm": BUDGET_ARM,
        "budget_threshold": budget_threshold,
        "arms": report_arms,
    }
    if args.dest:
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        args.dest.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.dest}")


if __name__ == "__main__":
    main()
