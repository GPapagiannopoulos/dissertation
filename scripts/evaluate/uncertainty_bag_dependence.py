"""Is the epistemic share a property of the data, or of the member set?

`uncertainty.py::decompose` splits an ensemble's predictive entropy into
`total = H(p_mean)`, `aleatoric = mean H(p_i)` and their difference. The module
docstring warns that the estimate belongs to the bag rather than to the data; this
measures how much, by sweeping bag SIZE on a fixed fold and by mixing model families.

Two intervals are reported and they answer different questions:

* the **subset spread** -- how much the share moves depending on WHICH members of a
  given size you happen to pick. Quantiles across subsets, not a bootstrap.
* the **subject bootstrap** -- how much a named bag's share moves under resampling
  PATIENTS, which is the ordinary sampling uncertainty every other number here carries.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/evaluate/uncertainty_bag_dependence.py \
        --fold testing --dest motor_output/comparison/newgrid/uncertainty_bags_test.json
"""

import argparse
import itertools
import json
from collections.abc import Sequence
from math import comb
from pathlib import Path

import numpy as np

from thesis.modelling.ensemble.diversity import Member, load_member
from thesis.modelling.ensemble.roster import (
    baseline_bundle,
    lora_ensemble,
    monolithic_bundles,
)
from thesis.modelling.ensemble.uncertainty import decompose
from thesis.modelling.evaluation.intervals import _draw_rows, _subject_groups

MONOLITHIC = ("ng-aki-seed0", "ng-aki-seed1", "ng-aki-seed2")
# the pre-collapse window the monolithic snapshot arm uses everywhere else
MONOLITHIC_WINDOW = (6000, 18000)


def _parse_args() -> argparse.Namespace:
    """Reads the fold, the sampling budget and the destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=("validation", "testing"), default="testing")
    parser.add_argument(
        "--subsets",
        type=int,
        default=200,
        help="subsets sampled per bag size; sizes with fewer are enumerated in full",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=1000,
        help="subject draws behind each named bag's interval; 0 skips them",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dest", type=Path, required=True)
    return parser.parse_args()


def align(members: Sequence[Member]) -> tuple[np.ndarray, np.ndarray]:
    """Puts every member on one row order and returns (scores, subjects).

    Args:
        members (Sequence[Member]): Banked bundles, not necessarily on one cohort.

    Returns:
        tuple[np.ndarray, np.ndarray]: Scores shaped (n_members, n_labels) in float64,
            and the subject each column belongs to.

    Raises:
        ValueError: If two members disagree about a label on the same landmark.
    """
    indexed = []
    for member in members:
        key = np.empty(member.scores.size, dtype=[("s", np.int64), ("t", np.int64)])
        key["s"], key["t"] = member.subjects, member.times
        order = np.argsort(key, kind="stable")
        unique, first = np.unique(key[order], return_index=True)
        indexed.append((unique, order[first], member))

    shared = indexed[0][0]
    for unique, _, _ in indexed[1:]:
        shared = np.intersect1d(shared, unique, assume_unique=True)

    rows = []
    for unique, positions, member in indexed:
        take = positions[np.searchsorted(unique, shared)]
        rows.append(Member(*(column[take] for column in member)))
    for index, member in enumerate(rows[1:], start=1):
        if not np.array_equal(member.targets, rows[0].targets):
            raise ValueError(f"Member {index} disagrees with member 0 about a label.")
    # float64 throughout: `decompose` guards concavity at 1e-9, and np.spacing(0.03)
    # in float32 is 1.9e-9, so near-identical members trip the guard on rounding alone
    return np.stack([m.scores for m in rows]).astype(np.float64), rows[0].subjects


def share(scores: np.ndarray) -> dict[str, float]:
    """The epistemic share of total entropy, plus the terms it comes from."""
    split = decompose(scores)
    total = float(split.total.mean())
    epistemic = float(split.epistemic.mean())
    return {
        "n": int(scores.shape[0]),
        "total": total,
        "aleatoric": float(split.aleatoric.mean()),
        "epistemic": epistemic,
        "share_pct": 100.0 * epistemic / total,
        "ambiguity": float(split.variance.mean()),
    }


def size_curve(
    scores: np.ndarray, *, subsets: int, seed: int
) -> list[dict[str, object]]:
    """The share against bag size, with the spread over WHICH members are chosen.

    Subsets are drawn without replacement and hold each member at most once, so a
    bag of size n is always n DISTINCT models. Sizes with no more than `subsets`
    combinations are enumerated exhaustively and carry no sampling error at all.

    Args:
        scores (np.ndarray): The full member matrix to draw from.
        subsets (int): How many subsets to sample per size.
        seed (int): Seeds the subset sampling.

    Returns:
        list[dict[str, object]]: One row per size, mean and spread across subsets.
    """
    rng = np.random.default_rng(seed)
    population = scores.shape[0]
    rows = []
    for size in range(2, population + 1):
        available = comb(population, size)
        combos = list(itertools.combinations(range(population), size))
        exhaustive = available <= subsets
        if not exhaustive:
            picked = rng.choice(available, size=subsets, replace=False)
            combos = [combos[index] for index in picked]
        values = np.array([share(scores[list(combo)])["share_pct"] for combo in combos])
        low, high = (
            np.quantile(values, [0.025, 0.975])
            if values.size > 2
            else (
                values.min(),
                values.max(),
            )
        )
        rows.append(
            {
                "n_members": size,
                "combinations_available": available,
                "n_subsets": len(combos),
                "exhaustive": exhaustive,
                "share_pct_mean": float(values.mean()),
                "share_pct_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "share_pct_min": float(values.min()),
                "share_pct_max": float(values.max()),
                "share_pct_lo": float(low),
                "share_pct_hi": float(high),
            }
        )
        print(
            f"  n={size:2d} {len(combos):4d} subsets  share {values.mean():5.2f}% "
            f"[{low:5.2f}, {high:5.2f}]{'  (exhaustive)' if exhaustive else ''}",
            flush=True,
        )
    return rows


def subject_interval(
    scores: np.ndarray, subjects: np.ndarray, *, resamples: int, seed: int
) -> dict[str, float]:
    """A subject-level bootstrap interval on one named bag's epistemic share.

    This is a different quantity from the subset spread: it holds the members fixed
    and resamples PATIENTS, so it is the ordinary sampling uncertainty.

    Args:
        scores (np.ndarray): One bag's aligned member matrix.
        subjects (np.ndarray): The subject each column belongs to.
        resamples (int): How many subject draws.
        seed (int): Seeds the draws.

    Returns:
        dict[str, float]: The observed share and its 95% bounds.
    """
    observed = share(scores)["share_pct"]
    rng = np.random.default_rng(seed)
    groups = _subject_groups(subjects)
    draws = [
        share(scores[:, _draw_rows(groups, rng)])["share_pct"] for _ in range(resamples)
    ]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"share_pct": observed, "lo": float(low), "hi": float(high)}


def main() -> None:
    """Sweeps bag size, then bands the named bags."""
    args = _parse_args()
    lora = list(lora_ensemble(args.fold))
    # one checkpoint per seed, so the three-seed bag is three DISTINCT training runs
    # rather than one run's ladder wearing three names
    mono = [
        monolithic_bundles(run, MONOLITHIC_WINDOW, args.fold)[-1] for run in MONOLITHIC
    ]
    tree = baseline_bundle(args.fold)

    paths = lora + mono + [tree]
    scores, subjects = align([load_member(path) for path in paths])
    lora_rows = list(range(len(lora)))
    mono_rows = list(range(len(lora), len(lora) + len(mono)))
    tree_row = scores.shape[0] - 1
    print(f"{args.fold}: {scores.shape[0]} bundles, {scores.shape[1]:,} landmarks\n")

    named = {
        "LoRA 23 (by-loss + last)": lora_rows,
        "monolithic 3 seeds": mono_rows,
        "LoRA 23 + monolithic 3": lora_rows + mono_rows,
        "LoRA 23 + XGBoost": lora_rows + [tree_row],
        "LoRA 23 + mono 3 + XGBoost": lora_rows + mono_rows + [tree_row],
        "XGBoost + monolithic seed 0 (2 families, N=2)": [tree_row, mono_rows[0]],
    }
    report: dict[str, object] = {
        "fold": args.fold,
        "n_labels": int(scores.shape[1]),
        "subsets_per_size": args.subsets,
        "resamples": args.resamples,
        "seed": args.seed,
        "sampling": (
            "subsets drawn without replacement; each holds every member at most once"
        ),
        "named": {},
    }
    for label, rows in named.items():
        entry = share(scores[rows])
        if args.resamples:
            entry.update(
                subject_interval(
                    scores[rows], subjects, resamples=args.resamples, seed=args.seed
                )
            )
        report["named"][label] = entry
        band = f" [{entry['lo']:.2f}, {entry['hi']:.2f}]" if args.resamples else ""
        print(f"{label:46s} n={entry['n']:2d} share={entry['share_pct']:5.2f}%{band}")

    print("\nbag-size sweep over the 23 LoRA members:")
    report["size_curve"] = size_curve(
        scores[lora_rows], subsets=args.subsets, seed=args.seed
    )

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    args.dest.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.dest}")


if __name__ == "__main__":
    main()
