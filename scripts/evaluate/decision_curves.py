"""Decision curve analysis over every arm, plus the ensemble-size curve.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/evaluate/decision_curves.py \
        --dest motor_output/comparison/decision_curves.json

Net benefit assesses whether at the threshold a clinician would actually act on,
is this model worth consulting at all?

This script generates two panels:

A) arm comparison: a single monolithic fine-tune, an ensemble of three
  monolithic fine-tunes, one monolithic run's own checkpoints, the LoRA ensemble under
  both checkpoint rules, and XGBoost, against alerting on everyone and on nobody.
B) ensemble size sweep: each member is one training run contributing the checkpoint
  the frozen rule selects and its `last`. Every subset of each size is enumerated and
  averaged to avoid selection bias. A spread of ensemble performance is given as an
  argument for deployment.

Everything here is measured on the landmark grid rebuilt 2026-08-21, which runs from
`admittime` rather than from the 48h cutoff. The old-grid roster this file carried
until 2026-08-26 named runs that no longer describe the data.

The threshold window is fixed in advance of the intervention, using clinical judgement.
Acting on predicted AKI means checking creatinine more often, reviewing
nephrotoxic drugs, holding contrast, and offering more IV fluids. These are low risk
interventions, and clinicians have a low barrier to offer them. On that basis a band of
5%-15% was selected.

Needs no GPU: it reads the prediction bundles `score_checkpoints.py` already banked.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from thesis.modelling.ensemble.decision_curve import (
    alert_rate,
    fraction_beating,
    net_benefit,
    subset_curves,
    treat_all_net_benefit,
    worst_subset,
)
from thesis.modelling.ensemble.diversity import align_members, load_member

# the roster lives in one module so this analysis and the selective-prediction one
# cannot drift into measuring different ensembles
from thesis.modelling.ensemble.roster import (
    LORA_RUNS,
    MONOLITHIC_RUNS,
    MONOLITHIC_WINDOW,
    ROOT,
    arm_definitions,
    drop_contested,
    lora_bundles,
)

# the subject-level draw already exists for the XGBoost comparison; a second copy here
# is exactly the metric drift the project keeps one of everything to avoid
from thesis.modelling.evaluation.intervals import _draw_rows, _subject_groups
from thesis.modelling.evaluation.metrics import binary_metrics

COMPARISON = ROOT / "motor_output" / "comparison"

REPORTED_THRESHOLDS = (0.05, 0.10, 0.15)

# the panel B size matching the monolithic arm's three training runs
BUDGET_MATCHED_SIZE = 3


def _parse_args() -> argparse.Namespace:
    """Reads the threshold grid and the resampling budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=float, default=0.01, help="lowest threshold")
    parser.add_argument("--high", type=float, default=0.30, help="highest threshold")
    parser.add_argument("--steps", type=int, default=59, help="points on the grid")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=(1, 3, 5, 7, 10, 12),
        help="ensemble sizes for panel B, counted in training runs",
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
        "--pair",
        action="append",
        default=[],
        metavar="LEFT:RIGHT",
        help="bootstrap the difference between two arms; repeatable",
    )
    parser.add_argument(
        "--resamples", type=int, default=2000, help="subject draws; 0 skips intervals"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fold",
        default="validation",
        help="which fold's banked bundles to read; selection stays on validation",
    )
    parser.add_argument("--dest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Builds both panels and, unless disabled, the paired difference bands."""
    args = _parse_args()
    # rounded so the reported thresholds land exactly on the grid; linspace puts
    # 0.09999999999999998 where 0.10 belongs, and a searchsorted then silently reads
    # the next grid point while the label still formats as "10%"
    thresholds = np.round(np.linspace(args.low, args.high, args.steps), 6)

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
        f"prevalence {prevalence:.4%}\n"
    )

    report = {
        "thresholds": thresholds.tolist(),
        "n_labels": int(targets.size),
        "n_subjects": len(groups),
        "prevalence": prevalence,
        "reported_thresholds": list(REPORTED_THRESHOLDS),
        # recorded because panel B is sensitive to it: at (14000, 18000) the monolithic
        # arm holds two checkpoints per seed and 92% of LoRA trios beat it at 10%; at
        # (6000, 18000) it holds four and only 54% do. The wider window is the
        # generous treatment of the comparator, so it is the default.
        "fold": args.fold,
        "monolithic_window": list(args.monolithic_window),
        "lora_runs": LORA_RUNS,
        "monolithic_runs": MONOLITHIC_RUNS,
        "arms": {},
        "references": {
            "treat_all": treat_all_net_benefit(targets, thresholds).tolist(),
            "treat_none": np.zeros_like(thresholds).tolist(),
        },
    }

    # panel B is computed first: it names the worst three-run ensemble, which then
    # joins the arms so it can take an ordinary paired interval like everything else
    run_members = np.stack(
        [
            stacked[[flat.index(path) for path in lora_bundles(run, args.fold)]].mean(
                axis=0
            )
            for run in LORA_RUNS
        ]
    )
    marks = np.argmin(
        np.abs(thresholds[:, None] - np.array(REPORTED_THRESHOLDS)), axis=0
    )
    if not np.allclose(thresholds[marks], REPORTED_THRESHOLDS):
        raise SystemExit(
            f"The reported thresholds {REPORTED_THRESHOLDS} are not on the grid; "
            f"nearest are {thresholds[marks]}. Widen --steps or move the window."
        )
    budget = int(marks[1])

    report["subset_curves"] = {}
    panel_b = {
        size: subset_curves(run_members, targets, thresholds, size=size)
        for size in args.sizes
    }

    trio = panel_b.get(BUDGET_MATCHED_SIZE)
    if trio is not None:
        chosen = worst_subset(trio, budget)
        definitions["lora_worst_trio"] = [
            path
            for index in chosen
            for path in lora_bundles(LORA_RUNS[index], args.fold)
        ]
        arms["lora_worst_trio"] = run_members[list(chosen)].mean(axis=0)
        report["worst_trio"] = {
            "runs": [LORA_RUNS[index] for index in chosen],
            "ranked_at_threshold": float(thresholds[budget]),
        }

    header = "  ".join(f"NB@{t:.0%}" for t in REPORTED_THRESHOLDS)
    print(f"{'arm':22s} {'n':>3} {'auprc':>7} {'ece':>7}  {header}   alert@10%")
    for name, scores in arms.items():
        curve = net_benefit(scores, targets, thresholds)
        at = net_benefit(scores, targets, np.array(REPORTED_THRESHOLDS))
        rates = alert_rate(scores, REPORTED_THRESHOLDS)
        metrics = binary_metrics(scores, targets)
        report["arms"][name] = {
            "members": [str(path) for path in definitions[name]],
            "n_members": len(definitions[name]),
            "net_benefit": curve.tolist(),
            "alert_rate": alert_rate(scores, thresholds).tolist(),
            "at_reported": {
                f"{t}": float(v) for t, v in zip(REPORTED_THRESHOLDS, at, strict=True)
            },
            "auprc": float(metrics["auprc"]),
            "ece": float(metrics["ece"]),
        }
        body = "  ".join(f"{value:>7.5f}" for value in at)
        print(
            f"{name:22s} {len(definitions[name]):>3} {metrics['auprc']:>7.5f} "
            f"{metrics['ece']:>7.5f}  {body}   {rates[1]:>8.2%}"
        )

    reference = treat_all_net_benefit(targets, np.array(REPORTED_THRESHOLDS))
    print(
        f"{'treat everyone':22s} {'-':>3} {'-':>7} {'-':>7}  "
        + "  ".join(f"{value:>7.5f}" for value in reference)
    )

    print(
        f"\nensemble size, over every subset of the {len(LORA_RUNS)} LoRA runs "
        f"(the spread is over WHICH RUNS, not which patients)"
    )
    print(
        f"{'runs':>5} {'subsets':>8}  "
        + "  ".join(f"{'NB@' + format(t, '.0%'):>25}" for t in REPORTED_THRESHOLDS)
    )
    for size, curves in panel_b.items():
        report["subset_curves"][str(size)] = {
            "n_subsets": curves.n_subsets,
            "mean": curves.mean.tolist(),
            "low": curves.low.tolist(),
            "high": curves.high.tolist(),
            "at_reported": {
                f"{thresholds[i]}": {
                    "mean": float(curves.mean[i]),
                    "low": float(curves.low[i]),
                    "high": float(curves.high[i]),
                }
                for i in marks
            },
        }
        body = "  ".join(
            f"{curves.mean[i]:.5f} [{curves.low[i]:.5f},{curves.high[i]:.5f}]"
            for i in marks
        )
        print(f"{size:>5} {curves.n_subsets:>8}  {body}")

    # the member-choice question, answered by counting rather than by resampling
    if trio is not None:
        reference_curve = net_benefit(arms["monolithic_seeds"], targets, thresholds)
        beating = fraction_beating(trio.curves, reference_curve)
        report["trios_beating_monolithic_seeds"] = beating.tolist()
        print(
            f"\nof the {trio.n_subsets} possible three-run LoRA ensembles, how many "
            f"beat the three-run monolithic ensemble:"
        )
        for i in marks:
            print(
                f"  at {thresholds[i]:.0%}: "
                f"{int(round(beating[i] * trio.n_subsets)):>3}/{trio.n_subsets} "
                f"({beating[i]:.1%})"
            )
        print(f"  worst trio: {', '.join(report['worst_trio']['runs'])}")

    pairs = [tuple(spec.split(":", 1)) for spec in args.pair] or [
        ("lora_last2", "monolithic_seeds"),
        ("lora_last2", "monolithic_single"),
        ("lora_last2", "xgboost"),
        ("lora_last2", "lora_all_last"),
        ("lora_worst_trio", "monolithic_seeds"),
    ]
    for left, right in pairs:
        for name in (left, right):
            if name not in arms:
                raise SystemExit(f"--pair names unknown arm {name!r}.")

    if args.resamples:
        rng = np.random.default_rng(args.seed)
        draws = {pair: [] for pair in pairs}
        for index in range(args.resamples):
            rows = _draw_rows(groups, rng)
            drawn = {
                name: net_benefit(scores[rows], targets[rows], thresholds)
                for name, scores in arms.items()
            }
            for pair in pairs:
                draws[pair].append(drawn[pair[0]] - drawn[pair[1]])
            if (index + 1) % 200 == 0:
                print(f"  draw {index + 1}/{args.resamples}", flush=True)

        report["differences"] = {}
        print(f"\npaired 95% bands, {args.resamples} subject resamples")
        for left, right in pairs:
            values = np.stack(draws[left, right])
            observed = net_benefit(arms[left], targets, thresholds) - net_benefit(
                arms[right], targets, thresholds
            )
            low, high = np.quantile(values, [0.025, 0.975], axis=0)
            report["differences"][f"{left}-{right}"] = {
                "value": observed.tolist(),
                "lo": low.tolist(),
                "hi": high.tolist(),
                "p_positive": (values > 0).mean(axis=0).tolist(),
            }
            body = "  ".join(
                f"{thresholds[i]:.0%}: {observed[i]:+.5f} "
                f"[{low[i]:+.5f}, {high[i]:+.5f}]"
                for i in marks
            )
            print(f"  {left} - {right}\n    {body}")

    if args.dest:
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        args.dest.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.dest}")


if __name__ == "__main__":
    main()
