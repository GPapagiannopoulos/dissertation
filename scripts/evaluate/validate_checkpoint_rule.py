"""Validates the use of a selection rule.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/evaluate/validate_checkpoint_rule.py \
        --runs motor_output/runs/lora-cfg-* motor_output/runs/lora-sched-seed1 \
        --seeds 5 --dest motor_output/comparison/rule_holdout.json

The selection rule is applied on one half of the SUBJECTS and the ensemble it produces
is scored on the other half, both directions, several seeds. An overly optimistic rule
collapses toward zero in the held out set, while a rule that found something real keeps
its margin.

Two rules are compared: validation loss (recomputed per row from the banked scores)
and validation AUPRC. Needs no GPU -- it reads the prediction bundles
`score_checkpoints.py` already banked.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from thesis.modelling.ensemble.diversity import (
    align_members,
    checkpoint_bundles,
    load_member,
    subject_halves,
)
from thesis.modelling.evaluation.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[2]
RULES = ("loss", "auprc")
REPORTED = ("auprc", "auroc", "brier", "ece", "precision_at_1pct")


def _parse_args() -> argparse.Namespace:
    """Reads the runs and how many halvings to average over."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, default=5, help="how many halvings")
    parser.add_argument("--dest", type=Path, default=None, help="write the report")
    return parser.parse_args()


def binary_cross_entropy(scores: np.ndarray, targets: np.ndarray) -> float:
    """The mean binary cross-entropy of banked probabilities.

    Reproduces the quantity `run_training` logs as validation loss, recomputed here
    because the banked bundles hold probabilities rather than a loss.

    Args:
        scores (np.ndarray): Predicted probabilities.
        targets (np.ndarray): The binary labels.

    Returns:
        float: The mean loss.
    """
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    return float(
        -np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped))
    )


def pick(rule: str, scores: np.ndarray, targets: np.ndarray, rows: np.ndarray) -> int:
    """The index of the checkpoint one rule chooses, judged on `rows` alone.

    Args:
        rule (str): Either `loss` or `auprc`.
        scores (np.ndarray): One run's checkpoints, shaped (n_checkpoints, n_labels).
        targets (np.ndarray): The labels those columns carry.
        rows (np.ndarray): The row indices the rule is allowed to look at.

    Returns:
        int: The chosen checkpoint's row in `scores`.

    Raises:
        ValueError: If the rule is not one this script knows.
    """
    if rule == "loss":
        return min(
            range(len(scores)),
            key=lambda index: binary_cross_entropy(scores[index][rows], targets[rows]),
        )
    if rule == "auprc":
        return max(
            range(len(scores)),
            key=lambda index: binary_metrics(scores[index][rows], targets[rows])[
                "auprc"
            ],
        )
    raise ValueError(f"Unknown rule {rule!r}; expected one of {RULES}.")


def main() -> None:
    """Applies each rule on one subject half and scores it on the other."""
    args = _parse_args()

    banked = {run.name: checkpoint_bundles(run) for run in args.runs}
    flat = [path for bundles in banked.values() for _, path in bundles]
    scores, targets, subjects = align_members([load_member(path) for path in flat])

    blocks, cursor = {}, 0
    for name, bundles in banked.items():
        blocks[name] = (
            [stem for stem, _ in bundles],
            scores[cursor : cursor + len(bundles)],
        )
        cursor += len(bundles)
    print(
        f"{len(banked)} runs, {len(flat)} checkpoints, {len(targets):,} labels, "
        f"{len(np.unique(subjects)):,} subjects\n"
    )

    rows = []
    for seed in range(args.seeds):
        left, right = subject_halves(subjects, seed)
        for direction, (fit, test) in enumerate(((left, right), (right, left))):
            fit_rows, test_rows = np.where(fit)[0], np.where(test)[0]
            for rule in RULES:
                chosen = {
                    name: pick(rule, block, targets, fit_rows)
                    for name, (_, block) in blocks.items()
                }
                bag = np.mean(
                    [
                        blocks[name][1][index][test_rows]
                        for name, index in chosen.items()
                    ],
                    axis=0,
                )
                record = binary_metrics(bag, targets[test_rows])
                rows.append(
                    {
                        "seed": seed,
                        "direction": direction,
                        "rule": rule,
                        **{name: float(record[name]) for name in REPORTED},
                        "picked": {
                            name: blocks[name][0][index]
                            for name, index in chosen.items()
                        },
                    }
                )
                body = "  ".join(f"{n} {record[n]:.5f}" for n in REPORTED)
                print(f"seed {seed} dir {direction} {rule:6s} {body}", flush=True)

    print(f"\nheld-out means over {len(rows) // len(RULES)} halvings")
    means = {}
    for rule in RULES:
        subset = [row for row in rows if row["rule"] == rule]
        means[rule] = {n: float(np.mean([row[n] for row in subset])) for n in REPORTED}
        print(
            f"  {rule:6s} " + "  ".join(f"{n} {means[rule][n]:.5f}" for n in REPORTED)
        )

    left_rule, right_rule = RULES[1], RULES[0]
    print(f"\ndeltas, {left_rule} rule - {right_rule} rule, per halving")
    deltas = {}
    ordered = {rule: [row for row in rows if row["rule"] == rule] for rule in RULES}
    for name in REPORTED:
        paired = [
            a[name] - b[name]
            for a, b in zip(ordered[left_rule], ordered[right_rule], strict=True)
        ]
        deltas[name] = {
            "mean": float(np.mean(paired)),
            "min": float(np.min(paired)),
            "max": float(np.max(paired)),
            "positive": int(sum(value > 0 for value in paired)),
            "n": len(paired),
        }
        record = deltas[name]
        print(
            f"  {name:20s} {record['mean']:+.5f}  "
            f"[{record['min']:+.5f}, {record['max']:+.5f}]  "
            f"{record['positive']}/{record['n']} halvings"
        )

    if args.dest:
        args.dest.parent.mkdir(parents=True, exist_ok=True)
        args.dest.write_text(
            json.dumps(
                {"splits": rows, "held_out_means": means, "deltas": deltas}, indent=1
            )
        )
        print(f"\nwrote {args.dest}")


if __name__ == "__main__":
    main()
