"""Risk and coverage inside an alert budget, using member disagreement to defer.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/evaluate/selective_prediction.py \
        --dest motor_output/comparison/newgrid/selective_prediction.json

A ward acts on a fixed number of alerts a day. This asks whether the ensemble's
internal disagreement is a better reason to withhold one of them than the score the
ensemble already produced -- and it is deliberately built so that the answer can be
no.

Three design choices come straight from what the uncertainty work measured, and each
one exists to stop this analysis flattering itself:

* Deferring globally is not tested, because ambiguity tracks the prediction at
  Spearman 0.954 and deferring the fold's most contested 10% deletes the alert list
  along with 41.6% of its positives. Deferral happens inside the alert list only.
* The headline control is the score's own ordering. Inside an alert list, deferring
  the lowest scores is the same thing as deferring the landmarks nearest the operating
  threshold, so the two controls the design named are one control -- and it needs no
  ensemble, which is the bar a 23-member ensemble has to clear to be worth running.
* The oracle is measured first. Stage 9's fitted correction had a real signal and a
  ceiling smaller than the cost of estimating it, and the hour that cost was spent
  before anyone measured the ceiling.

Needs no GPU: it reads the prediction bundles `score_checkpoints.py` already banked.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from thesis.modelling.ensemble.diversity import align_members, load_member
from thesis.modelling.ensemble.roster import ROOT, drop_contested, lora_ensemble
from thesis.modelling.ensemble.selective import (
    control_signals,
    disagreement_signals,
    rank_by_score,
    selective_curve,
)
from thesis.modelling.ensemble.uncertainty import explained_by_bins
from thesis.modelling.evaluation.intervals import _draw_rows, _subject_groups

# the budgets a ward might plausibly staff, so the headline 1% is not doing the work
BUDGETS = (0.005, 0.01, 0.02, 0.05)

# the coverage the paired interval is taken at: a fifth of the alert list withheld
HEADLINE_COVERAGE = 0.8

# what a real signal has to beat, and the ceiling it is a fraction of
CONTROL = "low_score"
ORACLE = "oracle"
RANDOM = "random"


def _parse_args() -> argparse.Namespace:
    """Reads the budgets, the coverage grid and the resampling budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budgets", type=float, nargs="+", default=BUDGETS, help="alert budgets"
    )
    parser.add_argument(
        "--coverages",
        type=float,
        nargs="+",
        default=(1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5),
        help="fractions of the alert list still acted on",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=200,
        help="risk bands for the within-band ranking; coarse bands leak the score",
    )
    parser.add_argument(
        "--resamples", type=int, default=2000, help="subject draws; 0 skips intervals"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fold",
        choices=("validation", "testing"),
        default="validation",
        help="which fold's banked bundles to read; `testing` reads selection_test",
    )
    parser.add_argument("--dest", type=Path, default=None)
    return parser.parse_args()


def precision_at(
    scores: np.ndarray,
    targets: np.ndarray,
    signal: np.ndarray,
    *,
    budget: float,
    coverage: float,
    backfill: bool,
    ranking: np.ndarray | None = None,
) -> float:
    """Precision on the acted-on set at one budget and one coverage."""
    curve = selective_curve(
        scores,
        targets,
        signal,
        budget=budget,
        coverages=np.array([coverage]),
        backfill=backfill,
        ranking=ranking,
    )
    return float(curve.precision[0])


def main() -> None:
    """Measures every signal at every budget, then bands the headline difference."""
    args = _parse_args()

    paths = lora_ensemble(args.fold)
    members, targets, subjects = align_members(
        drop_contested([load_member(path) for path in paths])
    )
    targets = targets.astype(float)
    scores = members.mean(axis=0)
    groups = _subject_groups(subjects)
    coverages = np.asarray(args.coverages, dtype=float)

    print(
        f"{args.fold} fold: {len(paths)} members, {targets.size:,} landmarks, "
        f"{len(groups):,} subjects, prevalence {targets.mean():.4%}\n"
    )

    signals = disagreement_signals(members, bins=args.bins)
    signals.update(control_signals(scores, targets, seed=args.seed))

    # the confound the whole design exists to route around, quantified per signal: how
    # much of each column a 100-bin step function of the prediction already accounts
    # for. A signal near 1.0 is the prediction wearing a different name.
    explained = {
        name: float(explained_by_bins(column, scores))
        for name, column in signals.items()
        if name != ORACLE
    }
    print("how much of each signal the prediction alone explains")
    for name, value in sorted(explained.items(), key=lambda pair: -pair[1]):
        print(f"  {name:20s} {value:.4f}")

    report = {
        "n_members": len(paths),
        "members": [str(path.relative_to(ROOT)) for path in paths],
        "n_labels": int(targets.size),
        "n_subjects": len(groups),
        "prevalence": float(targets.mean()),
        "budgets": list(args.budgets),
        "coverages": coverages.tolist(),
        "bins": args.bins,
        "explained_by_prediction": explained,
        "curves": {},
    }

    for backfill in (False, True):
        mode = "backfill" if backfill else "no_backfill"
        report["curves"][mode] = {}
        for budget in args.budgets:
            block = {}
            for name, column in signals.items():
                curve = selective_curve(
                    scores,
                    targets,
                    column,
                    budget=budget,
                    coverages=coverages,
                    backfill=backfill,
                )
                block[name] = {
                    "precision": curve.precision.tolist(),
                    "recall": curve.recall.tolist(),
                    "n_acted": curve.n_acted.tolist(),
                    "deferred_prevalence": curve.deferred_prevalence.tolist(),
                }
            report["curves"][mode][f"{budget}"] = block

            title = (
                "acted-on count held at the budget, deferred alerts backfilled"
                if backfill
                else "deferred alerts simply dropped"
            )
            print(f"\nbudget {budget:.1%} -- {title}")
            print(
                f"{'signal':20s} "
                + "  ".join(f"{c:>7.0%}" for c in coverages)
                + "   defer@80%"
            )
            for name, block_row in block.items():
                body = "  ".join(f"{p:>7.4f}" for p in block_row["precision"])
                at = np.argmin(np.abs(coverages - HEADLINE_COVERAGE))
                print(
                    f"{name:20s} {body}   {block_row['deferred_prevalence'][at]:>9.4f}"
                )

    if not args.resamples:
        _write(report, args.dest)
        return

    # the question is whether a signal beats the control, so the interval is on their
    # DIFFERENCE over one draw of subjects -- two one-signal intervals overlap freely
    # even when one wins on nearly every draw
    rng = np.random.default_rng(args.seed)
    tested = [name for name in signals if name not in (CONTROL, ORACLE)]
    modes = {"no_backfill": False, "backfill": True}
    cells = [
        (mode, budget, name)
        for mode in modes
        for budget in args.budgets
        for name in tested
    ]
    draws: dict[tuple[str, float, str], list[float]] = {cell: [] for cell in cells}

    print(
        f"\npaired 95% intervals against `{CONTROL}` at "
        f"{HEADLINE_COVERAGE:.0%} coverage, {args.resamples} subject resamples"
    )
    for index in range(args.resamples):
        rows = _draw_rows(groups, rng)
        drawn_scores, drawn_targets = scores[rows], targets[rows]
        # the score ranking is shared by every signal and every budget on this draw;
        # recomputing it per cell would sort the fold 48 times for no new information
        ranking = rank_by_score(drawn_scores)
        drawn = {name: column[rows] for name, column in signals.items()}

        for mode, backfill in modes.items():
            for budget in args.budgets:
                reference = precision_at(
                    drawn_scores,
                    drawn_targets,
                    drawn[CONTROL],
                    budget=budget,
                    coverage=HEADLINE_COVERAGE,
                    backfill=backfill,
                    ranking=ranking,
                )
                for name in tested:
                    draws[mode, budget, name].append(
                        precision_at(
                            drawn_scores,
                            drawn_targets,
                            drawn[name],
                            budget=budget,
                            coverage=HEADLINE_COVERAGE,
                            backfill=backfill,
                            ranking=ranking,
                        )
                        - reference
                    )
        if (index + 1) % 200 == 0:
            print(f"  draw {index + 1}/{args.resamples}", flush=True)

    full = rank_by_score(scores)
    report["paired"] = {mode: {} for mode in modes}
    for mode, backfill in modes.items():
        for budget in args.budgets:
            observed_control = precision_at(
                scores,
                targets,
                signals[CONTROL],
                budget=budget,
                coverage=HEADLINE_COVERAGE,
                backfill=backfill,
                ranking=full,
            )
            for name in tested:
                values = np.array(draws[mode, budget, name])
                observed = (
                    precision_at(
                        scores,
                        targets,
                        signals[name],
                        budget=budget,
                        coverage=HEADLINE_COVERAGE,
                        backfill=backfill,
                        ranking=full,
                    )
                    - observed_control
                )
                low, high = np.quantile(values, [0.025, 0.975])
                report["paired"][mode][f"{budget}:{name}"] = {
                    "value": float(observed),
                    "lo": float(low),
                    "hi": float(high),
                    "p_positive": float((values > 0).mean()),
                }
                print(
                    f"  {mode:12s} {budget:>6.1%} {name:20s} {observed:+.5f} "
                    f"[{low:+.5f}, {high:+.5f}]  {(values > 0).mean():>6.1%} of draws"
                )

    # "no better than the score" and "no better than a coin" are different claims, and
    # the second is the sharper one. Both differences were measured on the SAME draw
    # against the same reference, so subtracting the stored columns IS the paired
    # difference between the two signals -- no second bootstrap is needed.
    report["paired_vs_random"] = {mode: {} for mode in modes}
    print(f"\npaired 95% intervals against `{RANDOM}`, same draws")
    for mode in modes:
        for budget in args.budgets:
            baseline = np.array(draws[mode, budget, RANDOM])
            for name in tested:
                if name == RANDOM:
                    continue
                values = np.array(draws[mode, budget, name]) - baseline
                observed = (
                    report["paired"][mode][f"{budget}:{name}"]["value"]
                    - report["paired"][mode][f"{budget}:{RANDOM}"]["value"]
                )
                low, high = np.quantile(values, [0.025, 0.975])
                report["paired_vs_random"][mode][f"{budget}:{name}"] = {
                    "value": float(observed),
                    "lo": float(low),
                    "hi": float(high),
                    "p_positive": float((values > 0).mean()),
                }
                print(
                    f"  {mode:12s} {budget:>6.1%} {name:20s} {observed:+.5f} "
                    f"[{low:+.5f}, {high:+.5f}]  {(values > 0).mean():>6.1%} of draws"
                )

    _write(report, args.dest)


def _write(report: dict, dest: Path | None) -> None:
    """Saves the report, if one was asked for."""
    if dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
