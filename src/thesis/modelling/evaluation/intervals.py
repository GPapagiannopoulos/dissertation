"""Subject-level bootstrap intervals, paired and unpaired.

Confidence intervals resample subjects instead of landmarks. The 12-hourly grid puts
around ten highly correlated predictions inside one admission, so a row-level
bootstrap would report an interval several times too narrow.

The headline number is always the paired difference. Per-model intervals overlap even
when one model wins on nearly every resample, because each carries the variance of
the cohort; scoring both models on the same draw cancels it out.
"""

import numpy as np

from thesis.modelling.evaluation.metrics import binary_metrics


def _subject_groups(subjects: np.ndarray) -> list[np.ndarray]:
    """Row indices grouped by subject, so a draw is a concatenation, not a scan.

    Args:
        subjects (np.ndarray): The subject each row belongs to.

    Returns:
        list[np.ndarray]: One array of row indices per distinct subject.
    """
    unique, inverse = np.unique(subjects, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    # indexes where each new subject index begins
    boundaries = np.searchsorted(inverse[order], np.arange(unique.size + 1))
    return [order[boundaries[i] : boundaries[i + 1]] for i in range(unique.size)]


def _draw_rows(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One bootstrap resample: subjects with replacement, all their rows with them."""
    picked = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[i] for i in picked])


def bootstrap_interval(
    scores: np.ndarray,
    targets: np.ndarray,
    subjects: np.ndarray,
    *,
    metric: str = "auprc",
    resamples: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """A subject-level bootstrap interval for one metric.

    Subjects are resampled with replacement. All of the subject landmarks are included.
    This accounts for the inter-admission correlation between landmarks.

    Args:
        scores (np.ndarray): Predicted probabilities.
        targets (np.ndarray): Binary labels.
        subjects (np.ndarray): The subject each row belongs to.
        metric (str): Which key of `binary_metrics` to bound.
        resamples (int): How many bootstrap draws.
        seed (int): Seeds the draws.
        alpha (float): Two-sided width, so 0.05 gives a 95% interval.

    Returns:
        tuple[float, float]: The lower and upper bounds.

    Raises:
        ValueError: if the shapes of the input ndarrays do not match
    """
    shapes = {scores.shape, targets.shape, subjects.shape}
    if len(shapes) != 1:
        raise ValueError(
            "The shapes of 'scores', 'targets', and 'subjects' do not match."
        )

    rng = np.random.default_rng(seed)
    groups = _subject_groups(subjects)

    draws: list[float] = []
    for _ in range(resamples):
        rows = _draw_rows(groups, rng)
        value = binary_metrics(scores[rows], targets[rows])[metric]
        if not np.isnan(value):
            draws.append(value)

    return (
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )


def paired_interval(
    left: np.ndarray,
    right: np.ndarray,
    targets: np.ndarray,
    subjects: np.ndarray,
    *,
    metric: str = "auprc",
    resamples: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """A subject-level interval on the difference between two models.

    One model intervals overlap freely even when one model beats the other on nearly
    every resample, because they carry the variance of the cohort itself. Scoring both
    models on the same draw cancels out the covariance.

    Args:
        left (np.ndarray): The first model's scores.
        right (np.ndarray): The second model's scores, on the same rows.
        targets (np.ndarray): The labels those rows carry.
        subjects (np.ndarray): The subject each row belongs to.
        metric (str): Which key of `binary_metrics` to difference.
        resamples (int): How many bootstrap draws.
        seed (int): Seeds the draws.
        alpha (float): Two-sided width, so 0.05 gives a 95% interval.

    Returns:
        tuple[float, float, float]: The observed difference `left - right` on the
            full cohort, then the interval's lower and upper bounds.

    Raises:
        ValueError: If the arrays disagree in length, which means they are not the
            paired rows this function's arithmetic assumes.
    """
    shapes = {left.shape, right.shape, targets.shape, subjects.shape}
    if len(shapes) != 1:
        raise ValueError(
            f"A paired bootstrap needs one row per landmark in every array, got "
            f"shapes {sorted(str(shape) for shape in shapes)}. Align them with "
            f"`align_predictions` first."
        )

    observed = (
        binary_metrics(left, targets)[metric] - binary_metrics(right, targets)[metric]
    )

    rng = np.random.default_rng(seed)
    groups = _subject_groups(subjects)

    draws: list[float] = []
    for _ in range(resamples):
        rows = _draw_rows(groups, rng)
        difference = (
            binary_metrics(left[rows], targets[rows])[metric]
            - binary_metrics(right[rows], targets[rows])[metric]
        )
        if not np.isnan(difference):
            draws.append(difference)

    return (
        float(observed),
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )
