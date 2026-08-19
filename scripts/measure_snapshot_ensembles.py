"""Snapshot ensembles: the average of a model's checkpoint predictions.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/measure_snapshot_ensembles.py \
        --runs motor_output/runs/lora-cfg-* motor_output/runs/lora-sched-seed1 \
        --cross-k 1 2 3 \
        --dest motor_output/comparison/snapshot_ensembles.json

Each run is swept over its last k banked checkpoints, so k=1 is the single final
model and larger k averages further back in training. `--cross-k` then builds the
cross-run ensembles: every run's last k checkpoints in one bag.

Averaging a run's own checkpoints costs nothing -- they were written by a training
run that already happened -- so any gain here is free.
Needs no GPU: it reads the prediction bundles `score_checkpoints.py` already banked.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from thesis.modelling.ensemble.diversity import (
    align_members,
    checkpoint_bundles,
    load_member,
)
from thesis.modelling.motor.training import binary_metrics

ROOT = Path(__file__).resolve().parents[1]
REPORTED = ("auprc", "auroc", "brier", "ece", "precision_at_1pct")


def _parse_args() -> argparse.Namespace:
    """Reads the runs, the k sweep and any explicit member sets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", help="run folders to sweep")
    parser.add_argument(
        "--cross-k",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        help="build a cross-run ensemble from every run's last k checkpoints",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="NAME=PATH[,PATH...]",
        help="an explicit member set, for bundles that sit outside a run folder",
    )
    parser.add_argument("--dest", type=Path, default=None, help="write the report")
    return parser.parse_args()


def parse_group(spec: str) -> tuple[str, list[Path]]:
    """Splits a `NAME=PATH,PATH` group specification.

    Args:
        spec (str): The raw `--group` value.

    Returns:
        tuple[str, list[Path]]: The group's name and its bundle paths.

    Raises:
        ValueError: If the specification carries no `=`, or names no bundle.
    """
    name, _, joined = spec.partition("=")
    paths = [Path(part) for part in joined.split(",") if part]
    if not name or not paths:
        raise ValueError(f"Expected NAME=PATH[,PATH...], got {spec!r}.")
    return name, paths


def score_set(paths: Sequence[Path]) -> dict[str, float]:
    """Metrics for the equal-weight mean of one set of prediction bundles.

    Args:
        paths (Sequence[Path]): One or more banked prediction npz files.

    Returns:
        dict[str, float]: `binary_metrics` of the averaged scores, plus `n_members`,
            the members' `ambiguity` (their mean Brier less the ensemble's, which is
            the Krogh-Vedelsby decomposition), and their mean and best AUPRC.

    Raises:
        ValueError: If no bundle is given.
    """
    if not paths:
        raise ValueError("Cannot score an empty member set.")

    members = [load_member(path) for path in paths]
    if len(members) == 1:
        order = np.lexsort((members[0].times, members[0].subjects))
        scores = members[0].scores[order][None, :]
        targets = members[0].targets[order]
    else:
        scores, targets, _ = align_members(members)

    ensemble = binary_metrics(scores.mean(axis=0), targets)
    briers = ((scores - targets) ** 2).mean(axis=1)
    auprcs = [float(binary_metrics(row, targets)["auprc"]) for row in scores]
    return {
        "n_members": len(members),
        **{name: float(ensemble[name]) for name in REPORTED},
        "ambiguity": float(briers.mean() - ensemble["brier"]),
        "mean_member_auprc": float(np.mean(auprcs)),
        "best_member_auprc": float(max(auprcs)),
    }


def _line(label: str, record: dict[str, float]) -> str:
    """One report row."""
    body = "  ".join(f"{name} {record[name]:.5f}" for name in REPORTED)
    return (
        f"{label:52s} n={record['n_members']:>3}  {body}  "
        f"ambig {record['ambiguity']:.6f}  "
        f"mean-mem {record['mean_member_auprc']:.5f}"
    )


def main() -> None:
    """Sweeps each run, builds the cross-run bags, and reports them."""
    args = _parse_args()
    groups = [parse_group(spec) for spec in args.group]
    if not args.runs and not groups:
        raise SystemExit("Pass --runs or --group.")

    report: dict[str, dict[str, float]] = {}
    banked = {run: checkpoint_bundles(run) for run in args.runs or []}

    for run, bundles in banked.items():
        print(f"-- {run.name}  ({', '.join(stem for stem, _ in bundles)})")
        for k in range(1, len(bundles) + 1):
            window = bundles[-k:]
            record = score_set([path for _, path in window])
            report[f"{run.name}|last{k}"] = record
            stems = ",".join(stem for stem, _ in window)
            print(_line(f"   k={k} [{stems}]", record))

    if banked and args.cross_k:
        print("\n-- cross-run")
        for k in args.cross_k:
            paths = [path for bundles in banked.values() for _, path in bundles[-k:]]
            record = score_set(paths)
            report[f"cross-run|last{k}"] = record
            print(_line(f"   every run's last {k}", record))

    if groups:
        print("\n-- groups")
        for name, paths in groups:
            record = score_set(paths)
            report[name] = record
            print(_line(f"   {name}", record))

    if args.dest:
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        args.dest.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.dest}")


if __name__ == "__main__":
    main()
