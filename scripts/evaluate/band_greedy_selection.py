"""Calculates the benefit of greedy selection with an interval.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/evaluate/band_greedy_selection.py \
        --fold testing --seeds 10 \
        --dest motor_output/comparison/newgrid/greedy_banded_test.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_ensembles import paired_auprc  # noqa: E402
from select_ensemble import auprc, bag_scores, greedy_select  # noqa: E402

from thesis.modelling.ensemble.diversity import (  # noqa: E402
    align_members,
    load_member,
    subject_halves,
)
from thesis.modelling.ensemble.roster import (  # noqa: E402
    LORA_RUNS,
    RUNS,
    SELECTION,
    by_loss_stem,
    checkpoint_bundles,
)


def _parse_args() -> argparse.Namespace:
    """Reads the fold, the number of splits and the resampling budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=("validation", "testing"), default="testing")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--dest", type=Path, required=True)
    return parser.parse_args()


def by_loss_bundles(fold: str) -> list[Path]:
    """Each LoRA run's by-loss checkpoint.

    Args:
        fold (str): `validation` or `testing`.

    Returns:
        list[Path]: One bundle per configuration run.
    """
    return [
        dict(checkpoint_bundles(RUNS / run, SELECTION[fold]))[by_loss_stem(run)]
        for run in LORA_RUNS
    ]


def main() -> None:
    """Selects on one subject half, scores on the other, and bands the difference."""
    args = _parse_args()
    paths = by_loss_bundles(args.fold)
    names = [path.parents[1].name for path in paths]
    scores, targets, subjects = align_members([load_member(path) for path in paths])
    print(f"{args.fold}: {len(names)} candidates, {targets.size:,} labels\n")

    splits, picks = [], {name: 0 for name in names}
    for seed in range(args.seeds):
        left, right = subject_halves(subjects, seed)
        for label, fit, test in (("A->B", left, right), ("B->A", right, left)):
            chosen = greedy_select(scores[:, fit], targets[fit], args.iterations)
            for index in chosen:
                picks[names[index]] += 1
            greedy = bag_scores(scores[:, test], chosen)
            average = scores[:, test].mean(axis=0)
            order = [names[index] for index in chosen]
            trajectory = [
                auprc(bag_scores(scores[:, test], chosen[: k + 1]), targets[test])
                for k in range(len(chosen))
            ]
            gain, low, high = paired_auprc(
                greedy,
                average,
                targets[test],
                subjects[test],
                resamples=args.resamples,
                seed=seed,
            )
            splits.append(
                {
                    "seed": seed,
                    "direction": label,
                    "greedy_auprc": auprc(greedy, targets[test]),
                    "average_auprc": auprc(average, targets[test]),
                    "gain": gain,
                    "lo": low,
                    "hi": high,
                    "excludes_zero": bool(low > 0 or high < 0),
                    "n_distinct_members": len(set(chosen)),
                    "order": order,
                    "held_out_trajectory": trajectory,
                }
            )
            print(
                f"  seed {seed} {label}  greedy {splits[-1]['greedy_auprc']:.5f}  "
                f"average {splits[-1]['average_auprc']:.5f}  "
                f"gain {gain:+.5f} [{low:+.5f}, {high:+.5f}]"
                f"{' *' if splits[-1]['excludes_zero'] else ''}",
                flush=True,
            )
            print(f"      order: {' -> '.join(order)}", flush=True)

    gains = np.array([split["gain"] for split in splits])
    excluding = sum(split["excludes_zero"] for split in splits)
    favouring = sum(split["excludes_zero"] and split["gain"] > 0 for split in splits)
    print(
        f"\n{len(gains)} splits: mean gain {gains.mean():+.5f}, "
        f"sd {gains.std(ddof=1):.5f}, range [{gains.min():+.5f}, {gains.max():+.5f}]"
    )
    print(
        f"{excluding}/{len(gains)} splits have an interval excluding zero, "
        f"{favouring} of them favouring greedy"
    )
    singles = {
        name: auprc(row, targets) for name, row in zip(names, scores, strict=True)
    }
    print("\ntimes each member was picked, over every split:")
    for name, count in sorted(picks.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>4}x  {name}  (alone: {singles[name]:.5f})")
    first = [split["order"][0] for split in splits]
    print("\nfirst pick, by split:")
    for name in sorted(set(first), key=lambda value: -first.count(value)):
        print(f"  {first.count(name):>4}/{len(first)}  {name}")

    report = {
        "fold": args.fold,
        "n_labels": int(targets.size),
        "candidates": names,
        "seeds": args.seeds,
        "iterations": args.iterations,
        "resamples": args.resamples,
        "splits": splits,
        "mean_gain": float(gains.mean()),
        "sd_gain": float(gains.std(ddof=1)),
        "min_gain": float(gains.min()),
        "max_gain": float(gains.max()),
        "splits_excluding_zero": excluding,
        "splits_favouring_greedy": favouring,
        "pick_counts": picks,
        "member_auprc_alone": singles,
    }
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    args.dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.dest}")


if __name__ == "__main__":
    main()
