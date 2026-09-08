"""This module handles the uncertainty BALD split."""

from typing import NamedTuple

import numpy as np

EPSILON = 1e-7

TOLERANCE = 1e-9


class Decomposition(NamedTuple):
    """One column per landmark, all shaped (n_labels,).

    Attributes:
        mean (np.ndarray): The ensemble's prediction, the members' mean.
        total (np.ndarray): `H(mean)`, in nats.
        aleatoric (np.ndarray): `mean H(member)`, in nats.
        epistemic (np.ndarray): `total - aleatoric`, in nats.
        variance (np.ndarray): The members' variance, on the squared-error scale.
    """

    mean: np.ndarray
    total: np.ndarray
    aleatoric: np.ndarray
    epistemic: np.ndarray
    variance: np.ndarray


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    """The entropy of a Bernoulli variable, in nats.

    Args:
        probabilities (np.ndarray): Probabilities of the positive class, any shape.

    Returns:
        np.ndarray: Entropy, elementwise. Approaches zero at 0 and 1, maximal
            `ln 2` at a half.
    """
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))


def decompose(scores: np.ndarray) -> Decomposition:
    """Splits an aligned member matrix into total, aleatoric and epistemic columns.

    Args:
        scores (np.ndarray): Members' probabilities, shaped (n_members, n_labels), as
            `align_members` returns them.

    Returns:
        Decomposition: Five per-landmark columns.

    Raises:
        ValueError: If the matrix is not two-dimensional, holds fewer than two
            members, or carries a value outside [0, 1]. It also raises if the
            epistemic term comes back meaningfully negative, which concavity forbids
            and which therefore means the members are not aligned row for row.
    """
    if scores.ndim != 2:
        raise ValueError(
            f"Expected a (n_members, n_labels) matrix, got shape {scores.shape}."
        )
    if scores.shape[0] < 2:
        raise ValueError(
            f"Uncertainty needs at least two members to disagree, got "
            f"{scores.shape[0]}."
        )
    if scores.min() < 0.0 or scores.max() > 1.0:
        raise ValueError(
            f"Scores must be probabilities; got the range "
            f"[{scores.min()}, {scores.max()}]."
        )

    mean = scores.mean(axis=0)
    total = binary_entropy(mean)
    aleatoric = binary_entropy(scores).mean(axis=0)
    epistemic = total - aleatoric

    if epistemic.min() < -TOLERANCE:
        raise ValueError(
            f"Epistemic uncertainty reached {epistemic.min()}, which concavity "
            f"forbids; the members are probably not aligned row for row."
        )

    return Decomposition(
        mean=mean,
        total=total,
        aleatoric=aleatoric,
        epistemic=np.clip(epistemic, 0.0, None),
        variance=scores.var(axis=0),
    )


def explained_by_bins(values: np.ndarray, by: np.ndarray, *, bins: int = 100) -> float:
    """How much of `values` a step function of `by` accounts for.

    Args:
        values (np.ndarray): The column being explained.
        by (np.ndarray): The column doing the explaining.
        bins (int): How many quantile bins to cut `by` into.

    Returns:
        float: The fraction of variance explained, in [0, 1]. Zero when `values` is
            constant, since then there is nothing to explain.

    Raises:
        ValueError: If the two columns differ in length, or `bins` is not positive.
    """
    if values.shape != by.shape:
        raise ValueError(
            f"Columns must be the same length, got {values.shape} and {by.shape}."
        )
    if bins < 1:
        raise ValueError(f"Binning needs at least one bin, got {bins}.")

    total = float(values.var())
    if total == 0.0:
        return 0.0

    order = np.argsort(by, kind="stable")
    predicted = np.empty_like(values, dtype=float)
    for group in np.array_split(order, min(bins, values.size)):
        if group.size:
            predicted[group] = values[group].mean()

    return float(1.0 - ((values - predicted) ** 2).mean() / total)
