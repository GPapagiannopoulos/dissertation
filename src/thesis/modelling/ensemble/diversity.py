"""Module for measuring module disagreement between members."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.stats import spearmanr

# the paired subject-level bootstrap already exists for the XGBoost comparison; a
# second copy here is exactly the metric drift the project keeps one of everything to
# avoid
from thesis.modelling.evaluation.intervals import paired_interval
from thesis.modelling.evaluation.metrics import binary_metrics


class Member(NamedTuple):
    """One member's banked predictions, as `score_motor` writes them.

    Attributes:
        scores (np.ndarray): Predicted probabilities.
        targets (np.ndarray): The binary labels.
        subjects (np.ndarray): The subject each label belongs to.
        times (np.ndarray): Each label's prediction time, in microseconds since epoch.
    """

    scores: np.ndarray
    targets: np.ndarray
    subjects: np.ndarray
    times: np.ndarray


def load_member(path: Path) -> Member:
    """Reads one banked prediction bundle.

    Args:
        path (Path): An npz written by `score_motor`, four unnamed arrays in order.

    Returns:
        Member: The predictions and their provenance.

    Raises:
        ValueError: If the bundle does not hold exactly four arrays.
    """
    with np.load(path) as bundle:
        arrays = [bundle[name] for name in bundle.files]
    if len(arrays) != 4:
        raise ValueError(
            f"{path} holds {len(arrays)} arrays; a prediction bundle is "
            f"(scores, targets, subjects, times)."
        )
    return Member(*arrays)


def checkpoint_bundles(run: Path) -> list[tuple[str, Path]]:
    """Every scored checkpoint of one run, ordered by training step.

    Args:
        run (Path): A run folder holding `selection/*_predictions.npz`.

    Returns:
        list[tuple[str, Path]]: `(checkpoint stem, bundle path)`. `last` sorts after
            every `step_NNNN`; any other stem sorts before them, since only those two
            forms carry a step this can order on.

    Raises:
        FileNotFoundError: If the run has no banked predictions.
    """
    suffix = "_predictions.npz"
    found = [
        (path.name[: -len(suffix)], path)
        for path in (run / "selection").glob(f"*{suffix}")
    ]
    if not found:
        raise FileNotFoundError(
            f"{run} has no banked predictions; score it first with "
            f"scripts/evaluate/score_checkpoints.py."
        )
    return sorted(found, key=lambda pair: _step_order(pair[0]))


def _step_order(stem: str) -> tuple[int, int]:
    """Sort key over checkpoint stems, putting `last` after every `step_NNNN`."""
    if stem == "last":
        return (1, 0)
    if stem.startswith("step_"):
        return (0, int(stem.split("_")[1]))
    return (-1, 0)


def subject_halves(subjects: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Splits rows into two halves by subject.

    A subject contributes ~11 correlated landmarks, so a row-level split would leak the
    same patient into both halves and make the held-out number overly optimistic.

    Args:
        subjects (np.ndarray): The subject each row belongs to.
        seed (int): Seeds the permutation.

    Returns:
        tuple[np.ndarray, np.ndarray]: Boolean row masks for the two halves.
    """
    unique = np.unique(subjects)
    shuffled = np.random.default_rng(seed).permutation(unique)
    left = np.isin(subjects, shuffled[: len(shuffled) // 2])
    return left, ~left


def align_members(
    members: Sequence[Member],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Puts every member's scores on one common row order.

    Members are sorted by `(subject, time)` to prevent misalignment of scores.
    Comparison of scores that occurred in different landmarks inflates diversity.

    Args:
        members (Sequence[Member]): Two or more members' predictions.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Scores shaped
            (n_members, n_labels), the shared targets, and the shared subjects.

    Raises:
        ValueError: If fewer than two members are given, if they disagree on the set
            of landmarks, or if they disagree on a label.
    """
    if len(members) < 2:
        raise ValueError(f"Diversity needs at least two members, got {len(members)}.")

    ordered = []
    for member in members:
        order = np.lexsort((member.times, member.subjects))
        ordered.append(Member(*(column[order] for column in member)))

    first = ordered[0]
    for index, member in enumerate(ordered[1:], start=1):
        if member.scores.size != first.scores.size:
            raise ValueError(
                f"Member {index} holds {member.scores.size} labels against member "
                f"0's {first.scores.size}; they were scored on different cohorts."
            )
        if not np.array_equal(member.subjects, first.subjects) or not np.array_equal(
            member.times, first.times
        ):
            raise ValueError(
                f"Member {index} names different landmarks than member 0, so no "
                f"row-for-row comparison is possible."
            )
        if not np.array_equal(member.targets, first.targets):
            raise ValueError(
                f"Member {index} disagrees with member 0 about a label on the same "
                f"landmark; one of the two bundles is stale."
            )

    return (
        np.stack([member.scores for member in ordered]),
        first.targets,
        first.subjects,
    )


def top_flagged(scores: np.ndarray, k: float) -> np.ndarray:
    """The row indices a member would flag under a `k` alert budget.

    Args:
        scores (np.ndarray): Predicted probabilities.
        k (float): The fraction to flag, in (0, 1].

    Returns:
        np.ndarray: The flagged indices, sorted.

    Raises:
        ValueError: If `k` is not in (0, 1].
    """
    if not 0.0 < k <= 1.0:
        raise ValueError(f"An alert budget is a fraction in (0, 1], got {k}.")
    flagged = max(1, int(round(k * scores.size)))
    return np.sort(np.argsort(-scores, kind="stable")[:flagged])


def flag_overlap(left: np.ndarray, right: np.ndarray, *, k: float = 0.01) -> float:
    """Jaccard overlap of two members' flagged sets.

    Args:
        left (np.ndarray): One member's scores.
        right (np.ndarray): Another's, on the same rows.
        k (float): The alert budget.

    Returns:
        float: Intersection over union, 1.0 for identical flagging and 0.0 for
            disjoint. Both sets are the same size, so this cannot be inflated by one
            member flagging more.
    """
    a, b = set(top_flagged(left, k).tolist()), set(top_flagged(right, k).tolist())
    return len(a & b) / len(a | b)


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation of two members' rankings.

    Rank rather than value, because AUPRC and the alert budget both read the ordering
    and not the calibrated probability.

    Args:
        left (np.ndarray): One member's scores.
        right (np.ndarray): Another's, on the same rows.

    Returns:
        float: The correlation.
    """
    return float(spearmanr(left, right).statistic)


def alert_region(scores: np.ndarray, k: float) -> np.ndarray:
    """The rows the ensemble mean flags under a `k` alert budget.

    Args:
        scores (np.ndarray): Aligned scores, (n_members, n_labels).
        k (float): The alert budget.

    Returns:
        np.ndarray: The flagged row indices, ascending.
    """
    return top_flagged(scores.mean(axis=0), k)


def pairwise(
    scores: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float]
) -> np.ndarray:
    """Applies a two-member metric across every pair.

    Args:
        scores (np.ndarray): Aligned scores, (n_members, n_labels).
        metric (Callable): Takes two score vectors, returns a float.

    Returns:
        np.ndarray: A symmetric (n_members, n_members) matrix, diagonal included.
    """
    n = scores.shape[0]
    out = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            out[i, j] = out[j, i] = metric(scores[i], scores[j])
    return out


def off_diagonal(matrix: np.ndarray) -> np.ndarray:
    """The pairwise values, each pair once, excluding self-comparisons.

    Args:
        matrix (np.ndarray): A symmetric pairwise matrix.

    Returns:
        np.ndarray: The upper triangle above the diagonal, flattened.
    """
    return matrix[np.triu_indices_from(matrix, k=1)]


def ensemble_gain(
    scores: np.ndarray,
    targets: np.ndarray,
    subjects: np.ndarray | None = None,
    *,
    resamples: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """What averaging the members actually buys.

    The ensemble is the arithmetic mean of the members' probabilities. It is scored
    through 'binary_metrics'.

    Given `subjects`, the gain over the best member also carries a paired
    subject-level interval to account for covariance.

    Args:
        scores (np.ndarray): Aligned score matrix, (n_members, n_labels).
        targets (np.ndarray): The shared labels.
        subjects (np.ndarray | None): The subject each row belongs to. Omitted, the
            interval is skipped.
        resamples (int): Bootstrap draws for that interval.
        seed (int): Seeds the draws.

    Returns:
        dict[str, float]: `ensemble_auprc`, `mean_member_auprc`,
            `best_member_auprc`,`auprc_gain`,`gain_over_best`,`gain_over_best_lo`/`_hi`,
            `ensemble_brier`/`ensemble_ece`, `mean_member_brier`/`mean_member_ece`.
    """
    scored = [binary_metrics(row, targets) for row in scores]
    members = [metrics["auprc"] for metrics in scored]
    best = int(np.argmax(members))

    # the arithmetic mean of probabilities as per the deep-ensembles convention
    ensemble_scores = scores.mean(axis=0)
    ensemble = binary_metrics(ensemble_scores, targets)

    report = {
        "ensemble_auprc": ensemble["auprc"],
        "mean_member_auprc": float(np.mean(members)),
        "best_member_auprc": float(members[best]),
        "auprc_gain": ensemble["auprc"] - float(np.mean(members)),
        "gain_over_best": ensemble["auprc"] - float(members[best]),
        "ensemble_brier": ensemble["brier"],
        "ensemble_ece": ensemble["ece"],
        "mean_member_brier": float(np.mean([m["brier"] for m in scored])),
        "mean_member_ece": float(np.mean([m["ece"] for m in scored])),
    }
    if subjects is not None:
        _, low, high = paired_interval(
            ensemble_scores,
            scores[best],
            targets,
            subjects,
            resamples=resamples,
            seed=seed,
        )
        report["gain_over_best_lo"], report["gain_over_best_hi"] = low, high
    return report


def diversity_report(
    members: Sequence[Member], *, k: float = 0.01, resamples: int = 200
) -> dict[str, object]:
    """Every diversity number for one set of members.

    Args:
        members (Sequence[Member]): Two or more members' predictions.
        k (float): The alert budget `flag_overlap` uses.
        resamples (int): Bootstrap draws for the gain interval.

    Returns:
        dict[str, object]: `n_members`, `n_labels`, the mean and minimum pairwise
            `flag_overlap` and `rank_correlation`, the same rank correlation
            restricted to the ensemble's alert region as
            `mean_alert_rank_correlation` and `min_alert_rank_correlation`, and
            everything `ensemble_gain` returns.
    """
    scores, targets, subjects = align_members(members)
    region = scores[:, alert_region(scores, k)]
    overlaps = off_diagonal(pairwise(scores, lambda a, b: flag_overlap(a, b, k=k)))
    correlations = off_diagonal(pairwise(scores, rank_correlation))
    alert_correlations = off_diagonal(pairwise(region, rank_correlation))

    return {
        "n_members": float(scores.shape[0]),
        "n_labels": float(scores.shape[1]),
        "alert_budget": k,
        "mean_flag_overlap": float(overlaps.mean()),
        "min_flag_overlap": float(overlaps.min()),
        "mean_rank_correlation": float(correlations.mean()),
        "min_rank_correlation": float(correlations.min()),
        "mean_alert_rank_correlation": float(alert_correlations.mean()),
        "min_alert_rank_correlation": float(alert_correlations.min()),
        **ensemble_gain(scores, targets, subjects, resamples=resamples),
    }


def format_report(report: dict[str, object], title: str) -> str:
    """Renders a report as a short block of lines.

    Args:
        report (dict[str, object]): What `diversity_report` returned.
        title (str): A name for the member set.

    Returns:
        str: The block, no trailing newline.
    """
    lines = [
        f"{title}  ({report['n_members']:.0f} members, "
        f"{report['n_labels']:,.0f} labels)",
        f"  flag overlap @{report['alert_budget']:.0%}   "
        f"mean {report['mean_flag_overlap']:.4f}   "
        f"min {report['min_flag_overlap']:.4f}",
        f"  rank correlation      mean {report['mean_rank_correlation']:.4f}   "
        f"min {report['min_rank_correlation']:.4f}",
        f"  rank corr @{report['alert_budget']:.0%}       "
        f"mean {report['mean_alert_rank_correlation']:.4f}   "
        f"min {report['min_alert_rank_correlation']:.4f}",
        f"  auprc  members {report['mean_member_auprc']:.4f} "
        f"(best {report['best_member_auprc']:.4f})   "
        f"ensemble {report['ensemble_auprc']:.4f}",
        f"  gain   over mean {report['auprc_gain']:+.4f}   "
        f"over best {report['gain_over_best']:+.4f}",
        f"  brier  members {report['mean_member_brier']:.5f}   "
        f"ensemble {report['ensemble_brier']:.5f}",
        f"  ece    members {report['mean_member_ece']:.4f}    "
        f"ensemble {report['ensemble_ece']:.4f}",
    ]
    if "gain_over_best_lo" in report:
        lines.append(
            f"         paired 95% CI on the gain over best "
            f"[{report['gain_over_best_lo']:+.4f}, "
            f"{report['gain_over_best_hi']:+.4f}]"
        )
    return "\n".join(lines)
