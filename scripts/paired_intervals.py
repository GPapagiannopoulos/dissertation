"""Paired subject-level confidence intervals between named arms.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/paired_intervals.py \
        --runs motor_output/runs/lora-cfg-* motor_output/runs/lora-sched-seed1 \
        --arm-stems last2=step_014000,last --arm-stems alllast=last \
        --arm mono=motor_output/comparison/v3_step_010000_predictions.npz \
        --pair last2:alllast --pair last2:mono \
        --dest motor_output/comparison/paired_snapshot.json

An arm is the equal-weight mean of its bundles, so a single npz is a single model and
a list is an ensemble. Every arm is scored on the SAME draw of subjects, which is the
whole mechanism: a draw holding easy patients is easy for all of them, so the cohort's
own variance cancels and what survives is the difference between the arms. Two
one-model intervals cannot do this -- they overlap freely even when one arm wins on
every resample.

Subjects are resampled, never rows: the 12-hourly grid puts ~9 correlated landmarks
inside one admission, so a row-level bootstrap reports an interval several times too
narrow. Needs no GPU -- it reads the prediction bundles `score_checkpoints.py` banked.

`--arm-stems` builds a UNIFORM arm -- the same checkpoint stems from every run -- and
its fallback to `last` only fires when a run never wrote that stem. The by-loss roster
is not uniform (nine config runs take `step_014000`, `ff-r8` takes `last`), so that arm
has to be spelled out with `--arm` and explicit paths. The headline
ensemble-vs-monolithic figure was produced that way, by the earlier one-off at
`motor_output/comparison/paired_ensemble_vs_monolithic.py`, which this driver
generalises.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

# the subject-level draw already exists for the XGBoost comparison; a second copy
# here is exactly the metric drift the project keeps one of everything to avoid
from thesis.modelling.baseline.compare import _draw_rows, _subject_groups
from thesis.modelling.ensemble.diversity import (
    align_members,
    checkpoint_bundles,
    load_member,
)
from thesis.modelling.motor.training import binary_metrics

ROOT = Path(__file__).resolve().parents[1]
REPORTED = ("auprc", "auroc", "brier", "ece", "precision_at_1pct")


def _parse_args() -> argparse.Namespace:
    """Reads the arms, the pairs to difference, and the resampling budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", default=[])
    parser.add_argument(
        "--arm-stems",
        action="append",
        default=[],
        metavar="NAME=STEM[,STEM...]",
        help="an arm taking those checkpoints from every --runs folder",
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=PATH[,PATH...]",
        help="an arm given as explicit prediction npz files",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="LEFT:RIGHT",
        help="difference these two arms; defaults to every arm against the first",
    )
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dest", type=Path, default=None, help="write the report")
    return parser.parse_args()


def parse_spec(spec: str, separator: str = "=") -> tuple[str, list[str]]:
    """Splits a `NAME=A,B` specification into its name and its comma-separated parts.

    Args:
        spec (str): The raw command-line value.
        separator (str): What divides the name from the parts.

    Returns:
        tuple[str, list[str]]: The name and its parts.

    Raises:
        ValueError: If either side is empty.
    """
    name, _, joined = spec.partition(separator)
    parts = [part for part in joined.split(",") if part]
    if not name or not parts:
        raise ValueError(f"Expected NAME{separator}A[,B...], got {spec!r}.")
    return name, parts


def stem_bundles(runs: Sequence[Path], stems: Sequence[str]) -> list[Path]:
    """The bundles for those checkpoint stems across every run.

    A run missing one of the stems falls back to `last`, which is what a shorter
    schedule leaves behind; anything else would silently drop a member.

    Args:
        runs (Sequence[Path]): The run folders.
        stems (Sequence[str]): Checkpoint stems, e.g. `step_014000` or `last`.

    Returns:
        list[Path]: One bundle per run per stem.

    Raises:
        FileNotFoundError: If a run holds neither the stem nor `last`.
    """
    paths = []
    for run in runs:
        available = dict(checkpoint_bundles(run))
        for stem in stems:
            if stem in available:
                paths.append(available[stem])
            elif "last" in available:
                paths.append(available["last"])
            else:
                raise FileNotFoundError(f"{run} holds neither {stem} nor last.")
    return paths


def main() -> None:
    """Scores every arm on one shared sequence of subject draws."""
    args = _parse_args()

    definitions: dict[str, list[Path]] = {}
    for spec in args.arm_stems:
        name, stems = parse_spec(spec)
        if not args.runs:
            raise SystemExit("--arm-stems needs --runs.")
        definitions[name] = stem_bundles(args.runs, stems)
    for spec in args.arm:
        name, parts = parse_spec(spec)
        definitions[name] = [Path(part) for part in parts]
    if len(definitions) < 2:
        raise SystemExit("Pass at least two arms.")

    flat = [path for paths in definitions.values() for path in paths]
    stacked, targets, subjects = align_members([load_member(path) for path in flat])

    arms, cursor = {}, 0
    for name, paths in definitions.items():
        arms[name] = stacked[cursor : cursor + len(paths)].mean(axis=0)
        cursor += len(paths)

    point = {name: binary_metrics(scores, targets) for name, scores in arms.items()}
    for name, record in point.items():
        body = "  ".join(f"{metric} {record[metric]:.5f}" for metric in REPORTED)
        print(f"{name:16s} n={len(definitions[name]):>3}  {body}")

    names = list(arms)
    pairs = (
        [
            (left, others[0])
            for left, others in (parse_spec(spec, ":") for spec in args.pair)
        ]
        if args.pair
        else [(name, names[0]) for name in names[1:]]
    )
    for left, right in pairs:
        for name in (left, right):
            if name not in arms:
                raise SystemExit(f"--pair names unknown arm {name!r}.")

    groups = _subject_groups(subjects)
    rng = np.random.default_rng(args.seed)
    draws = {pair: {metric: [] for metric in REPORTED} for pair in pairs}
    for index in range(args.resamples):
        rows = _draw_rows(groups, rng)
        drawn = {
            name: binary_metrics(scores[rows], targets[rows])
            for name, scores in arms.items()
        }
        for pair in pairs:
            for metric in REPORTED:
                difference = drawn[pair[0]][metric] - drawn[pair[1]][metric]
                if not np.isnan(difference):
                    draws[pair][metric].append(difference)
        if (index + 1) % 100 == 0:
            print(f"  draw {index + 1}/{args.resamples}", flush=True)

    report = {
        "resamples": args.resamples,
        "seed": args.seed,
        "n_subjects": len(groups),
        "arms": {
            name: [str(path) for path in paths] for name, paths in definitions.items()
        },
        "point": {
            name: {metric: float(record[metric]) for metric in REPORTED}
            for name, record in point.items()
        },
        "deltas": {},
    }
    print(
        f"\npaired 95% intervals, {args.resamples} resamples over "
        f"{len(groups):,} subjects"
    )
    for left, right in pairs:
        print(f"-- {left} - {right}")
        record = {}
        for metric in REPORTED:
            values = np.array(draws[left, right][metric])
            observed = float(point[left][metric] - point[right][metric])
            low, high = np.quantile(values, 0.025), np.quantile(values, 0.975)
            record[metric] = {
                "value": observed,
                "lo": float(low),
                "hi": float(high),
                "p_left_wins": float((values > 0).mean()),
            }
            print(
                f"   {metric:20s} {observed:+.5f}  [{low:+.5f}, {high:+.5f}]  "
                f"wins {record[metric]['p_left_wins']:.3f}"
            )
        report["deltas"][f"{left}-{right}"] = record

    if args.dest:
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        args.dest.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.dest}")


if __name__ == "__main__":
    main()
