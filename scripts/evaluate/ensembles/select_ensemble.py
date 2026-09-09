r"""Greedy forward selection of ensemble members (Caruana et al., 2004).

Run from the repo root:

    .venv-modelling/bin/python scripts/evaluate/ensembles/select_ensemble.py \
        --runs motor_output/runs/lora-cfg-* motor_output/runs/lora-sched-seed1

Members are picked one at a time, with replacement, so a strong member can be picked
repeatedly and effectively weighted.

Selecting and reporting on the same rows is optimistic, so subjects are split in half,
selection runs on one half and the ensemble is scored on the other. Both directions
are reported, plus the select-on-everything bag as the upper bound it is.

Needs no GPU: it reads the prediction bundles `score_checkpoints.py` already banked.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from thesis.modelling.ensemble.diversity import (
    align_members,
    load_member,
    subject_halves,
)
from thesis.modelling.evaluation.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[3]


def best_bundle(run: Path) -> Path:
    """The prediction npz for a run's highest-AUPRC checkpoint.

    Args:
        run (Path): A run folder holding `selection/checkpoint_ranking.json`.

    Returns:
        Path: The banked npz for that checkpoint.

    Raises:
        FileNotFoundError: If the run has not been scored.
    """
    ranking = run / "selection" / "checkpoint_ranking.json"
    if not ranking.is_file():
        raise FileNotFoundError(f"{run} has no checkpoint_ranking.json; score it.")
    rows = json.loads(ranking.read_text())
    best = max(rows, key=lambda row: row["auprc"])
    return run / "selection" / f"{Path(best['checkpoint']).stem}_predictions.npz"


def auprc(scores: np.ndarray, targets: np.ndarray) -> float:
    """AUPRC through the project's one metric definition."""
    return float(binary_metrics(scores, targets)["auprc"])


def greedy_select(
    scores: np.ndarray, targets: np.ndarray, iterations: int
) -> list[int]:
    """Greedy forward selection with replacement.

    Args:
        scores (np.ndarray): Aligned scores, (n_members, n_rows).
        targets (np.ndarray): The shared labels.
        iterations (int): How many picks to make; a member may be picked repeatedly.

    Returns:
        list[int]: The chosen member indices, in the order they were added.
    """
    chosen: list[int] = []
    running = np.zeros_like(targets, dtype=np.float64)
    for _ in range(iterations):
        best_index, best_score = -1, -np.inf
        for index in range(scores.shape[0]):
            candidate = (running + scores[index]) / (len(chosen) + 1)
            score = auprc(candidate, targets)
            if score > best_score:
                best_index, best_score = index, score
        chosen.append(best_index)
        running = running + scores[best_index]
    return chosen


def bag_scores(scores: np.ndarray, chosen: list[int]) -> np.ndarray:
    """The mean prediction of a selected bag, honouring repeats as weights."""
    return scores[chosen].mean(axis=0)


def _parse_args() -> argparse.Namespace:
    """Reads the member set and the selection budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", help="run folders")
    parser.add_argument("--bundles", type=Path, nargs="+", help="explicit npz files")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Selects, reports held-out and in-sample bags, and optionally writes them."""
    args = _parse_args()
    if not args.runs and not args.bundles:
        raise SystemExit("Pass --runs or --bundles.")

    paths = list(args.bundles or []) + [best_bundle(run) for run in args.runs or []]
    names = [path.parents[1].name for path in paths]
    scores, targets, subjects = align_members([load_member(path) for path in paths])

    print(f"{len(names)} candidates, {len(targets):,} labels\n")
    singles = {
        name: auprc(row, targets) for name, row in zip(names, scores, strict=True)
    }
    for name, value in sorted(singles.items(), key=lambda kv: -kv[1]):
        print(f"  {value:.4f}  {name}")

    everything = auprc(scores.mean(axis=0), targets)
    print(f"\naverage of all {len(names)}: {everything:.4f}")

    left, right = subject_halves(subjects, args.seed)
    held_out = []
    for label, fit, test in (("A->B", left, right), ("B->A", right, left)):
        chosen = greedy_select(scores[:, fit], targets[fit], args.iterations)
        value = auprc(bag_scores(scores[:, test], chosen), targets[test])
        baseline = auprc(scores[:, test].mean(axis=0), targets[test])
        counts = {names[i]: chosen.count(i) for i in sorted(set(chosen))}
        held_out.append(value - baseline)
        print(
            f"\n{label}  held-out {value:.4f}  vs average-of-all {baseline:.4f} "
            f"({value - baseline:+.4f})"
        )
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>2}x  {name}")

    chosen = greedy_select(scores, targets, args.iterations)
    in_sample = auprc(bag_scores(scores, chosen), targets)
    print(f"\nselect-on-everything (UPPER BOUND, not a result): {in_sample:.4f}")
    for name, count in sorted(
        {names[i]: chosen.count(i) for i in set(chosen)}.items(), key=lambda kv: -kv[1]
    ):
        print(f"    {count:>2}x  {name}")
    print(f"\nmean held-out gain over averaging everything: {np.mean(held_out):+.4f}")

    if args.dest:
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        args.dest.write_text(
            json.dumps(
                {
                    "candidates": names,
                    "single_auprc": singles,
                    "average_of_all": everything,
                    "held_out_gain": float(np.mean(held_out)),
                    "in_sample_auprc": in_sample,
                    "in_sample_counts": {
                        names[i]: chosen.count(i) for i in set(chosen)
                    },
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.dest}")


if __name__ == "__main__":
    main()
