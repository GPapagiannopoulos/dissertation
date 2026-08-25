"""The metric definitions every arm reports through.

They live outside the training loop because the XGBoost baseline, the linear probe
and every ensemble analysis score the same way a fine-tune does; a second definition
anywhere would let two arms drift apart on what "AUPRC" means.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def expected_calibration_error(
    scores: np.ndarray, targets: np.ndarray, *, bins: int = 10
) -> float:
    """The gap between predicted probability and observed frequency, size-weighted.

    Args:
        scores (np.ndarray): The predicted probabilities, shaped (n,).
        targets (np.ndarray): The binary labels, shaped (n,).
        bins (int): How many quantile bins to cut. The bins are only approximately
            equal-mass when scores tie across a cut point, which is why each is
            weighted by its own count rather than by 1/bins.

    Returns:
        float: The weighted mean absolute gap, in probability units. Zero is perfect.

    Raises:
        ValueError: If `bins` is not positive.
    """
    if bins < 1:
        raise ValueError(f"Calibration needs at least one bin, got {bins}.")

    order = np.argsort(scores, kind="stable")
    total = 0.0
    for group in np.array_split(order, min(bins, scores.size)):
        if group.size == 0:
            continue
        gap = abs(float(scores[group].mean()) - float(targets[group].mean()))
        total += gap * group.size
    return total / scores.size


def precision_recall_at_k(
    scores: np.ndarray, targets: np.ndarray, *, k: float
) -> tuple[float, float]:
    """Precision and recall when the top `k` fraction of predictions is flagged.

    Args:
        scores (np.ndarray): The predicted probabilities, shaped (n,).
        targets (np.ndarray): The binary labels, shaped (n,).
        k (float): The fraction to flag, in (0, 1]. The count floors at one, so a
            small evaluation subsample still yields a number.

    Returns:
        tuple[float, float]: Precision and recall among the flagged. Recall is NaN
            when the sample holds no positive at all.

    Raises:
        ValueError: If `k` is not in (0, 1].
    """
    if not 0.0 < k <= 1.0:
        raise ValueError(f"An alert budget is a fraction in (0, 1], got {k}.")

    flagged = max(1, int(round(k * scores.size)))
    top = np.argsort(-scores, kind="stable")[:flagged]

    hits = float(targets[top].sum())
    positives = float(targets.sum())
    return hits / flagged, (hits / positives if positives else float("nan"))


def binary_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Scores one set of predictions against its labels.

    AUPRC leads because the task is 3.56% positive, where AUROC is dominated by the
    negatives and a useless model still reads around 0.5 rather than around the base
    rate.

    Args:
        scores (np.ndarray): The predicted probabilities, shaped (n,).
        targets (np.ndarray): The binary labels, shaped (n,).

    Returns:
        dict[str, float]: `auprc`, `auroc`, `brier`, `ece`, `precision_at_1pct`,
            `recall_at_1pct`, `precision_at_5pct`, `base_rate`, `n` and `n_positive`.
            Both areas come back as NaN when the labels are all one class, which is a
            real possibility on a small evaluation subsample and not an error.

    Raises:
        ValueError: If the two arrays disagree in length, or if either is empty.
    """
    if scores.shape != targets.shape:
        raise ValueError(
            f"Scores shaped {scores.shape} do not match targets shaped {targets.shape}."
        )
    if scores.size == 0:
        raise ValueError("Cannot score an empty set of predictions.")

    single_class = len(np.unique(targets)) < 2
    precision_1, recall_1 = precision_recall_at_k(scores, targets, k=0.01)
    precision_5, _ = precision_recall_at_k(scores, targets, k=0.05)

    return {
        "auprc": float("nan")
        if single_class
        else float(average_precision_score(targets, scores)),
        "auroc": float("nan")
        if single_class
        else float(roc_auc_score(targets, scores)),
        "brier": float(np.mean((scores - targets) ** 2)),
        "ece": expected_calibration_error(scores, targets),
        "precision_at_1pct": precision_1,
        "recall_at_1pct": recall_1,
        "precision_at_5pct": precision_5,
        "base_rate": float(targets.mean()),
        "n": float(targets.size),
        "n_positive": float(targets.sum()),
    }
