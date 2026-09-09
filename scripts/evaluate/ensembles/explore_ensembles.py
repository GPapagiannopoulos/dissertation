r"""Explore ensemble compositions over banked predictions.

Five experiments, all pure npz arithmetic on checkpoints already scored:

1. the hybrid: XGBoost blended with the LoRA ensemble, over a weight sweep;
2. the rank ladder as an ensemble, r=2..32 at alpha=32, by-loss and +last;
3. the rank-diversity prediction: does a run that barely decays snapshot-ensemble
   worse than one that decays a lot?
4. the narrow placements (q/k/v, ff, o) as an ensemble, by-loss and +last;
5. the saturation curve: mean AUPRC over every subset of a given size.

Run from the repo root with the modelling interpreter:

    .venv-modelling/bin/python scripts/evaluate/ensembles/explore_ensembles.py \
        --fold test \
        --dest motor_output/comparison/newgrid/explore_test.json
"""

import argparse
import itertools
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from thesis.modelling.ensemble.diversity import (
    Member,
    flag_overlap,
    load_member,
    off_diagonal,
    pairwise,
    rank_correlation,
)
from thesis.modelling.evaluation.intervals import _draw_rows, _subject_groups
from thesis.modelling.evaluation.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "motor_output" / "runs"
NEWGRID = ROOT / "motor_output" / "comparison" / "newgrid"

# The twelve new-grid LoRA configurations the corpus is built from.
LORA_RUNS: tuple[str, ...] = (
    "ng-all4-r2-a32",
    "ng-all4-r4-a32",
    "ng-all4-r8",
    "pilot-all4-r16-a32",
    "ng-all4-r32-a32",
    "ng-all4-r16",
    "ng-all4-r32",
    "ng-all5-r8",
    "ng-o-r8",
    "ng-qkv-r8",
    "ng-ff-r8",
    "ng-qv-seed1",
)

# The rank ladder: q/k/v/ff at alpha=32, sixteen-fold capacity range.
RANK_LADDER: tuple[tuple[int, str], ...] = (
    (2, "ng-all4-r2-a32"),
    (4, "ng-all4-r4-a32"),
    (8, "ng-all4-r8"),
    (16, "pilot-all4-r16-a32"),
    (32, "ng-all4-r32-a32"),
)

# The narrow placements, matched at r=8.
PLACEMENTS: tuple[str, ...] = ("ng-qkv-r8", "ng-ff-r8", "ng-o-r8")

XGBOOST = {
    "validation": NEWGRID / "xgboost600_predictions.npz",
    "test": NEWGRID / "xgboost600_testing_predictions.npz",
}
SUBDIR = {"validation": "selection", "test": "selection_test"}


def paired_auprc(
    left: np.ndarray,
    right: np.ndarray,
    targets: np.ndarray,
    subjects: np.ndarray,
    *,
    resamples: int,
    seed: int = 0,
) -> tuple[float, float, float]:
    """A subject-level paired interval on AUPRC alone.

    `paired_interval` recomputes every metric on every draw; only AUPRC is wanted
    here, and the full metric set costs roughly four times as much per draw.

    Args:
        left (np.ndarray): The first model's scores.
        right (np.ndarray): The second model's scores, on the same rows.
        targets (np.ndarray): The labels those rows carry.
        subjects (np.ndarray): The subject each row belongs to.
        resamples (int): How many bootstrap draws.
        seed (int): Seeds the draws.

    Returns:
        tuple[float, float, float]: The observed `left - right`, then the 95% bounds.
    """
    observed = float(
        average_precision_score(targets, left) - average_precision_score(targets, right)
    )
    rng = np.random.default_rng(seed)
    groups = _subject_groups(subjects)
    deltas = []
    for _ in range(resamples):
        rows = _draw_rows(groups, rng)
        drawn = targets[rows]
        if len(np.unique(drawn)) < 2:
            continue
        deltas.append(
            average_precision_score(drawn, left[rows])
            - average_precision_score(drawn, right[rows])
        )
    low, high = np.quantile(deltas, [0.025, 0.975])
    return observed, float(low), float(high)


def paired_all(
    left: np.ndarray,
    right: np.ndarray,
    targets: np.ndarray,
    subjects: np.ndarray,
    *,
    resamples: int,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Paired intervals on every metric, sharing one set of draws.

    Args:
        left (np.ndarray): The first model's scores.
        right (np.ndarray): The second model's scores, on the same rows.
        targets (np.ndarray): The labels those rows carry.
        subjects (np.ndarray): The subject each row belongs to.
        resamples (int): How many bootstrap draws.
        seed (int): Seeds the draws.

    Returns:
        dict[str, dict[str, float]]: Per metric, the observed delta and its bounds.
    """
    keys = ("auprc", "auroc", "brier", "ece", "precision_at_1pct")
    observed = {
        key: binary_metrics(left, targets)[key] - binary_metrics(right, targets)[key]
        for key in keys
    }
    rng = np.random.default_rng(seed)
    groups = _subject_groups(subjects)
    draws: dict[str, list[float]] = {key: [] for key in keys}
    for _ in range(resamples):
        rows = _draw_rows(groups, rng)
        drawn = targets[rows]
        if len(np.unique(drawn)) < 2:
            continue
        one = binary_metrics(left[rows], drawn)
        two = binary_metrics(right[rows], drawn)
        for key in keys:
            draws[key].append(one[key] - two[key])
    out = {}
    for key in keys:
        low, high = np.quantile(draws[key], [0.025, 0.975])
        out[key] = {"delta": observed[key], "lo": float(low), "hi": float(high)}
    return out


def _parse_args() -> argparse.Namespace:
    """Reads the fold and the destination off the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=("validation", "test"), default="test")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=500)
    parser.add_argument(
        "--max-subsets",
        type=int,
        default=400,
        help="cap on how many subsets of one size the saturation curve enumerates",
    )
    return parser.parse_args()


def _key(member: Member) -> np.ndarray:
    """The (subject, time) landmark key, as a sortable structured array."""
    key = np.empty(member.scores.size, dtype=[("s", np.int64), ("t", np.int64)])
    key["s"] = member.subjects
    key["t"] = member.times
    return key


def common_rows(members: Sequence[Member]) -> list[Member]:
    """Restricts every member to the landmarks all of them scored.

    `align_members` requires identical cohorts; the XGBoost bundle drops the handful
    of contested `(subject, prediction_time)` keys that overlapping admissions create,
    so the tree and the transformer arms differ by a few rows. This takes the
    intersection instead, and keeps the first row of any duplicated key.

    Args:
        members (Sequence[Member]): Two or more banked bundles.

    Returns:
        list[Member]: The same members, on one shared row order.

    Raises:
        ValueError: If two members disagree about a label on the same landmark.
    """
    indexed = []
    for member in members:
        key = _key(member)
        order = np.argsort(key, kind="stable")
        unique, first = np.unique(key[order], return_index=True)
        indexed.append((unique, order[first], member))

    shared = indexed[0][0]
    for unique, _, _ in indexed[1:]:
        shared = np.intersect1d(shared, unique, assume_unique=True)

    aligned = []
    for unique, positions, member in indexed:
        take = positions[np.searchsorted(unique, shared)]
        aligned.append(Member(*(column[take] for column in member)))

    for index, member in enumerate(aligned[1:], start=1):
        if not np.array_equal(member.targets, aligned[0].targets):
            raise ValueError(f"Member {index} disagrees with member 0 about a label.")
    return aligned


def stack(members: Sequence[Member]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligns members and returns (scores, targets, subjects)."""
    aligned = common_rows(members)
    return (
        np.stack([member.scores for member in aligned]),
        aligned[0].targets,
        aligned[0].subjects,
    )


def ranking(run: str, fold: str) -> list[dict[str, object]]:
    """One run's scored checkpoints on the given fold."""
    path = RUNS / run / SUBDIR[fold] / "checkpoint_ranking.json"
    if not path.is_file():
        raise FileNotFoundError(f"{run} has no {SUBDIR[fold]}/checkpoint_ranking.json.")
    return json.loads(path.read_text())


def bundle(run: str, stem: str, fold: str) -> Path:
    """The npz `score_checkpoints.py` banked for one checkpoint."""
    path = RUNS / run / SUBDIR[fold] / f"{stem}_predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing.")
    return path


def by_loss_stem(run: str, fold: str) -> str:
    """The stem of the run's minimum-validation-loss checkpoint."""
    rows = ranking(run, fold)
    return Path(min(rows, key=lambda row: row["loss"])["checkpoint"]).stem


def member_paths(runs: Sequence[str], fold: str, *, plus_last: bool) -> list[Path]:
    """Each run's by-loss checkpoint, optionally with its `last.pt` as well.

    A run whose by-loss pick already IS `last` contributes one member, never two.

    Args:
        runs (Sequence[str]): Run folder names.
        fold (str): `validation` or `test`.
        plus_last (bool): Whether to add each run's final checkpoint.

    Returns:
        list[Path]: The prediction bundles to average.
    """
    paths = []
    for run in runs:
        stems = [by_loss_stem(run, fold)]
        if plus_last and "last" not in stems:
            stems.append("last")
        paths.extend(bundle(run, stem, fold) for stem in stems)
    return paths


def ensemble_report(
    paths: Sequence[Path], label: str, *, resamples: int
) -> dict[str, object]:
    """Scores the mean of a member set, with its diversity and its gain.

    Args:
        paths (Sequence[Path]): The bundles to average.
        label (str): How the set is named in the report.
        resamples (int): Bootstrap draws for the interval on the gain.

    Returns:
        dict[str, object]: Metrics, member statistics, ambiguity and overlap.
    """
    scores, targets, subjects = stack([load_member(path) for path in paths])
    mean = scores.mean(axis=0)
    members = [binary_metrics(row, targets) for row in scores]
    best = max(range(len(members)), key=lambda i: members[i]["auprc"])
    ensemble = binary_metrics(mean, targets)

    gain, low, high = paired_auprc(
        mean, scores[best], targets, subjects, resamples=resamples
    )
    overlap = off_diagonal(pairwise(scores, flag_overlap))
    correlation = off_diagonal(pairwise(scores, rank_correlation))
    member_brier = float(np.mean([m["brier"] for m in members]))
    return {
        "label": label,
        "n_members": len(paths),
        "members": [str(path.relative_to(ROOT)) for path in paths],
        "ensemble": ensemble,
        "mean_member_auprc": float(np.mean([m["auprc"] for m in members])),
        "best_member_auprc": members[best]["auprc"],
        "best_member": str(paths[best].relative_to(ROOT)),
        "gain_over_best": gain,
        "gain_over_best_lo": low,
        "gain_over_best_hi": high,
        "ambiguity": member_brier - ensemble["brier"],
        "mean_member_brier": member_brier,
        "mean_flag_overlap": float(overlap.mean()),
        "mean_rank_correlation": float(correlation.mean()),
    }


def hybrid(fold: str, *, resamples: int) -> dict[str, object]:
    """Blends the tree with the LoRA ensemble over a weight sweep.

    Args:
        fold (str): Which fold's bundles to read.
        resamples (int): Bootstrap draws for the paired intervals at w=0.5.

    Returns:
        dict[str, object]: The sweep, and 50/50 paired against both parents.
    """
    lora_paths = member_paths(LORA_RUNS, fold, plus_last=True)
    members = [load_member(path) for path in lora_paths]
    tree = load_member(XGBOOST[fold])
    scores, targets, subjects = stack(members + [tree])
    lora, xgboost = scores[:-1].mean(axis=0), scores[-1]

    sweep = []
    for weight in np.round(np.arange(0.0, 1.01, 0.05), 2):
        blend = weight * lora + (1.0 - weight) * xgboost
        sweep.append({"lora_weight": float(weight), **binary_metrics(blend, targets)})

    even = 0.5 * lora + 0.5 * xgboost
    paired = {
        f"hybrid50-{name}": paired_all(
            even, other, targets, subjects, resamples=resamples
        )
        for name, other in (("xgboost600", xgboost), ("lora_ensemble", lora))
    }
    return {
        "fold": fold,
        "n_labels": int(targets.size),
        "n_lora_members": len(lora_paths),
        "sweep": sweep,
        "hybrid50": binary_metrics(even, targets),
        "paired": paired,
    }


def rank_ensemble(fold: str, *, resamples: int) -> list[dict[str, object]]:
    """The r=2..32 ladder averaged, by-loss and with each run's `last`."""
    runs = [run for _, run in RANK_LADDER]
    return [
        ensemble_report(
            member_paths(runs, fold, plus_last=plus_last),
            f"rank ladder r=2..32, {'by-loss + last' if plus_last else 'by-loss'}",
            resamples=resamples,
        )
        for plus_last in (False, True)
    ]


def placement_ensemble(fold: str, *, resamples: int) -> list[dict[str, object]]:
    """The three narrow placements averaged, by-loss and with each run's `last`."""
    return [
        ensemble_report(
            member_paths(PLACEMENTS, fold, plus_last=plus_last),
            f"placements qkv/ff/o, {'by-loss + last' if plus_last else 'by-loss'}",
            resamples=resamples,
        )
        for plus_last in (False, True)
    ]


def rank_diversity(fold: str, *, resamples: int) -> list[dict[str, object]]:
    """Snapshot-ensembles each rank's own ladder.

    Stage 8h found late checkpoints earn their place by overfitting in different
    directions, and the rank ladder decays monotonically in rank. The prediction is
    that r=2, which barely decays, gains least from averaging its own snapshots.

    Args:
        fold (str): Which fold's ladders to read.
        resamples (int): Bootstrap draws for the interval on the gain.

    Returns:
        list[dict[str, object]]: One row per rank, with the whole-ladder gain and the
            per-rank decay it is meant to track.
    """
    rows = []
    for rank, run in RANK_LADDER:
        scored = ranking(run, fold)
        if len(scored) < 3:
            continue
        stems = [Path(row["checkpoint"]).stem for row in scored]
        paths = [bundle(run, stem, fold) for stem in stems]
        label = f"{run} snapshots (k={len(paths)})"
        report = ensemble_report(paths, label, resamples=resamples)
        peak = max(row["auprc"] for row in scored)
        end = next(
            row["auprc"] for row in scored if Path(row["checkpoint"]).stem == "last"
        )
        rows.append({"rank": rank, "run": run, "decay": end - peak, **report})
    return rows


def saturation(fold: str, *, max_subsets: int, seed: int = 0) -> dict[str, object]:
    """Mean ensemble AUPRC against member count, over subsets of the LoRA configs.

    Subsets are drawn over RUNS, not over checkpoints, so the curve varies the
    training budget rather than the selection rule. Every subset of a size is
    enumerated when there are few enough; otherwise `max_subsets` are sampled.

    Args:
        fold (str): Which fold's bundles to read.
        max_subsets (int): Cap on subsets enumerated per size.
        seed (int): Seeds the sampling when a size is capped.

    Returns:
        dict[str, object]: One curve for the by-loss members and one for by-loss+last.
    """
    rng = np.random.default_rng(seed)
    curves = {}
    for name, plus_last in (("by_loss", False), ("by_loss_plus_last", True)):
        per_run = {
            run: [
                load_member(path)
                for path in member_paths([run], fold, plus_last=plus_last)
            ]
            for run in LORA_RUNS
        }
        flat = [member for members in per_run.values() for member in members]
        scores, targets, _ = stack(flat)
        offsets, cursor = {}, 0
        for run, members in per_run.items():
            offsets[run] = list(range(cursor, cursor + len(members)))
            cursor += len(members)

        points = []
        for size in range(1, len(LORA_RUNS) + 1):
            combos = list(itertools.combinations(range(len(LORA_RUNS)), size))
            if len(combos) > max_subsets:
                picked = rng.choice(len(combos), size=max_subsets, replace=False)
                combos = [combos[index] for index in picked]
            values = []
            for combo in combos:
                rows = [
                    index
                    for position in combo
                    for index in offsets[LORA_RUNS[position]]
                ]
                values.append(
                    float(average_precision_score(targets, scores[rows].mean(axis=0)))
                )
            points.append(
                {
                    "n_runs": size,
                    "n_subsets": len(combos),
                    "mean_auprc": float(np.mean(values)),
                    "min_auprc": float(np.min(values)),
                    "max_auprc": float(np.max(values)),
                }
            )
        full = points[-1]["mean_auprc"]
        single = points[0]["mean_auprc"]
        for point in points:
            point["share_of_full_gain"] = (
                (point["mean_auprc"] - single) / (full - single)
                if full > single
                else float("nan")
            )
        curves[name] = points
    return curves


def main() -> None:
    """Runs every experiment and writes one report."""
    args = _parse_args()
    report: dict[str, object] = {"fold": args.fold, "resamples": args.resamples}

    print("hybrid ...", flush=True)
    report["hybrid"] = hybrid(args.fold, resamples=args.resamples)

    print("rank ladder ensemble ...", flush=True)
    report["rank_ensemble"] = rank_ensemble(args.fold, resamples=args.resamples)

    print("placement ensemble ...", flush=True)
    report["placement_ensemble"] = placement_ensemble(
        args.fold, resamples=args.resamples
    )

    print("rank diversity ...", flush=True)
    report["rank_diversity"] = rank_diversity(args.fold, resamples=args.resamples)

    print("saturation ...", flush=True)
    report["saturation"] = saturation(args.fold, max_subsets=args.max_subsets)

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    args.dest.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.dest}")


if __name__ == "__main__":
    main()
