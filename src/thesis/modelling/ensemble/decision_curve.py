"""Module implementing decision-curve analysis (DCA).

AUPRC and AUROC read the ordering of predictions which don't reflect clinical practice.
A clinician acts when the predicted risk crosses a threshold. Choosing to act above
`t` means being indifferent between acting and not acting at risk `t`, so one true
positive is worth `t / (1 - t)` false alarms. Sweeping `t` therefore sweeps every
exchange rate a clinician might hold.

    net benefit = TP/N - (FP/N) * t / (1 - t)

The units are true positives per landmark adjusted for the cost of false alarms.
It is bounded by the extremes of alerting on everyone, and alerting on nobody. A
beneficial model needs to at least beat the benefit of both these naive approaches.
"""

from collections.abc import Sequence
from itertools import combinations
from typing import NamedTuple

import numpy as np


class SubsetCurves(NamedTuple):
    """Net benefit over every ensemble of one size.

    The spread here is over which members were picked, not over which patients were
    studied. Those are independent uncertainties and a subject-level bootstrap is still
    needed for the second.

    Attributes:
        mean (np.ndarray): The mean curve over all subsets, shaped (n_thresholds,).
        low (np.ndarray): The worst subset's benefit at each threshold.
        high (np.ndarray): The best subset's benefit at each threshold.
        n_subsets (int): How many subsets were enumerated.
        subsets (tuple[tuple[int, ...], ...]): The member indices behind each curve.
        curves (np.ndarray): Every subset's curve, shaped (n_subsets, n_thresholds).
    """

    mean: np.ndarray
    low: np.ndarray
    high: np.ndarray
    n_subsets: int
    subsets: tuple[tuple[int, ...], ...]
    curves: np.ndarray


def _validate(scores: np.ndarray, targets: np.ndarray, thresholds: np.ndarray) -> None:
    """Guards shared by every entry point.

    Raises:
        ValueError: If the arrays disagree in length or are empty, if the scores are
            not probabilities, or if a threshold sits outside [0, 1).
    """
    if scores.shape != targets.shape:
        raise ValueError(
            f"Scores shaped {scores.shape} do not match targets shaped {targets.shape}."
        )
    if scores.size == 0:
        raise ValueError("Cannot build a decision curve from no predictions.")
    if scores.min() < 0.0 or scores.max() > 1.0:
        raise ValueError(
            f"Thresholds are probabilities, so scores must be too; got the range "
            f"[{scores.min()}, {scores.max()}]."
        )
    if thresholds.size and (thresholds.min() < 0.0 or thresholds.max() >= 1.0):
        raise ValueError(
            f"Thresholds live in [0, 1); a threshold of 1 divides by zero. Got "
            f"[{thresholds.min()}, {thresholds.max()}]."
        )


def net_benefit(
    scores: np.ndarray, targets: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """The net benefit of alerting at each threshold.

    A landmark is alerted when its score is greater than or equal to the threshold.

    Args:
        scores (np.ndarray): Predicted probabilities, shaped (n,).
        targets (np.ndarray): Binary labels, shaped (n,).
        thresholds (np.ndarray): Threshold probabilities in [0, 1).

    Returns:
        np.ndarray: Net benefit per threshold, in true positives per landmark.

    Raises:
        ValueError: If the inputs fail `_validate`.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    _validate(scores, targets, thresholds)

    order = np.argsort(-scores, kind="stable")
    ranked = scores[order]
    positives = np.concatenate(([0.0], np.cumsum(targets[order])))

    # how many scores are at or above each threshold; `-ranked` ascends, so a
    # right-insertion counts exactly the entries whose score clears the threshold
    alerted = np.searchsorted(-ranked, -thresholds, side="right")
    true_positive = positives[alerted]
    false_positive = alerted - true_positive

    odds = thresholds / (1.0 - thresholds)
    return (true_positive - false_positive * odds) / scores.size


def treat_all_net_benefit(targets: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """The net benefit of alerting on every landmark.

    The reference a model has to beat to be worth consulting at all. It crosses zero
    exactly at the prevalence, above which blanket alerting does net harm.

    Args:
        targets (np.ndarray): Binary labels, shaped (n,).
        thresholds (np.ndarray): Threshold probabilities in [0, 1).

    Returns:
        np.ndarray: Net benefit per threshold.

    Raises:
        ValueError: If the inputs fail `_validate`.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    _validate(np.zeros_like(targets, dtype=float), targets, thresholds)

    prevalence = float(targets.mean())
    return prevalence - (1.0 - prevalence) * thresholds / (1.0 - thresholds)


def subset_curves(
    members: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray,
    *,
    size: int,
    max_subsets: int | None = None,
    seed: int = 0,
) -> SubsetCurves:
    """Net benefit averaged over every ensemble of `size` members.

    Reports the average and spread of every ensemble with `size` members
    to avoid selection bias.

    Args:
        members (np.ndarray): Member scores, shaped (n_members, n_labels).
        targets (np.ndarray): Binary labels, shaped (n_labels,).
        thresholds (np.ndarray): Threshold probabilities in [0, 1).
        size (int): How many members each ensemble holds.
        max_subsets (int | None): Sample this many subsets at random rather than
            enumerating, for sizes where the count explodes. None enumerates all.
        seed (int): Seeds the sampling when `max_subsets` is given.

    Returns:
        SubsetCurves: The mean, worst and best curves, and the subset count.

    Raises:
        ValueError: If `size` is not between one and the member count, or if the
            member matrix is not two-dimensional.
    """
    if members.ndim != 2:
        raise ValueError(
            f"Expected a (n_members, n_labels) matrix, got shape {members.shape}."
        )
    if not 1 <= size <= members.shape[0]:
        raise ValueError(
            f"Cannot draw ensembles of {size} from {members.shape[0]} members."
        )

    every = list(combinations(range(members.shape[0]), size))
    if max_subsets is not None and len(every) > max_subsets:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(every), size=max_subsets, replace=False)
        every = [every[index] for index in sorted(chosen)]

    curves = np.stack(
        [
            net_benefit(members[list(subset)].mean(axis=0), targets, thresholds)
            for subset in every
        ]
    )
    return SubsetCurves(
        mean=curves.mean(axis=0),
        low=curves.min(axis=0),
        high=curves.max(axis=0),
        n_subsets=len(every),
        subsets=tuple(every),
        curves=curves,
    )


def alert_rate(scores: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """The fraction of landmarks alerted at each threshold.

    Reported beside the curve because a threshold that wins on net benefit while
    alerting on a quarter of the ward is not deployable, and the curve alone hides it.

    Args:
        scores (np.ndarray): Predicted probabilities, shaped (n,).
        thresholds (Sequence[float]): Threshold probabilities.

    Returns:
        np.ndarray: The alerted fraction per threshold.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    ranked = np.sort(scores)[::-1]
    return np.searchsorted(-ranked, -thresholds, side="right") / scores.size


def fraction_beating(curves: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """How often a subset's curve sits strictly above a reference curve.

    Answers the member-choice question directly, with no resampling: of every ensemble
    you could have assembled from the runs you trained, what share beats the
    comparator? A bootstrap cannot answer this, because the uncertainty is over which
    members were picked rather than over which patients were seen.

    Args:
        curves (np.ndarray): Subset curves, shaped (n_subsets, n_thresholds).
        reference (np.ndarray): The curve to beat, shaped (n_thresholds,).

    Returns:
        np.ndarray: The beating fraction per threshold, in [0, 1].

    Raises:
        ValueError: If the reference does not match the curves' threshold count.
    """
    if curves.ndim != 2:
        raise ValueError(
            f"Expected curves shaped (n_subsets, n_thresholds), got {curves.shape}."
        )
    if reference.shape != curves.shape[1:]:
        raise ValueError(
            f"Reference shaped {reference.shape} does not match the curves' "
            f"{curves.shape[1]} thresholds."
        )
    return (curves > reference).mean(axis=0)


def worst_subset(result: SubsetCurves, index: int) -> tuple[int, ...]:
    """The member indices of the subset with the lowest benefit at one threshold.

    Bootstrapping every subset is prohibitive, so the honest shortcut is to give a
    normal paired interval to the WORST one: if even that clears the comparator, so
    does every other subset a practitioner might have ended up with.

    Args:
        result (SubsetCurves): The enumerated subsets.
        index (int): Which threshold to rank on.

    Returns:
        tuple[int, ...]: The member indices of the weakest subset.

    Raises:
        IndexError: If the threshold index is out of range.
    """
    return result.subsets[int(np.argmin(result.curves[:, index]))]
