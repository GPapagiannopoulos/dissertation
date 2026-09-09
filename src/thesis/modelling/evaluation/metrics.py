"""The metric definitions every arm reports through."""

from typing import NamedTuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import SplineTransformer

# the logit's asymptotes are clipped at the same point `training.py` clips its
# loss, so a landmark's log-odds here and its cross-entropy there agree
EPSILON = 1e-7

MAX_ITER = 200


def expected_calibration_error(
    scores: np.ndarray, targets: np.ndarray, *, bins: int = 10
) -> float:
    """The gap between predicted probability and observed frequency, size-weighted.

    Args:
        scores (np.ndarray): The predicted probabilities, shaped (n,).
        targets (np.ndarray): The binary labels, shaped (n,).
        bins (int): How many quantile bins to cut.

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


class CalibrationCurve(NamedTuple):
    """One reliability diagram: what was predicted against what was observed.

    Attributes:
        mean_score (np.ndarray): Mean predicted probability in each bin.
        observed_rate (np.ndarray): Observed positive rate in each bin.
        count (np.ndarray): How many rows fell in each bin.
        edges (np.ndarray): The `n_bins + 1` boundaries used, outermost infinite so
            every score falls somewhere.
    """

    mean_score: np.ndarray
    observed_rate: np.ndarray
    count: np.ndarray
    edges: np.ndarray


def calibration_edges(scores: np.ndarray, *, bins: int = 10) -> np.ndarray:
    """Boundaries cutting `scores` into equal-count bins.

    Quantile binning to account for low prevalence.

    Args:
        scores (np.ndarray): The predicted probabilities, shaped (n,).
        bins (int): How many bins to aim for; ties may yield fewer.

    Returns:
        np.ndarray: Ascending boundaries, `-inf` first and `+inf` last.

    Raises:
        ValueError: If `bins` is not positive.
    """
    if bins < 1:
        raise ValueError(f"Calibration needs at least one bin, got {bins}.")

    order = np.argsort(scores, kind="stable")
    groups = np.array_split(order, min(bins, scores.size))
    interior = {float(scores[group[-1]]) for group in groups[:-1] if group.size}
    return np.array([-np.inf, *sorted(interior), np.inf])


def calibration_curve(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 10,
    edges: np.ndarray | None = None,
) -> CalibrationCurve:
    """Predicted probability against observed frequency by bin.

    Args:
        scores (np.ndarray): The predicted probabilities, shaped (n,).
        targets (np.ndarray): The binary labels, shaped (n,).
        bins (int): How many equal-count bins to cut. Ignored when `edges` is given.
        edges (np.ndarray | None): Boundaries from `calibration_edges`, to reuse a
            partition.

    Returns:
        CalibrationCurve: One entry per bin, ascending in score. Empty bins carry NaN.

    Raises:
        ValueError: If `scores` and `targets` are different lengths, or `bins` is not
            positive.
    """
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if scores.shape != targets.shape:
        raise ValueError(
            f"{scores.shape} scores against {targets.shape} targets; they do not "
            f"describe the same rows."
        )

    if edges is None:
        edges = calibration_edges(scores, bins=bins)
    edges = np.asarray(edges, dtype=float)

    interior = edges[1:-1]
    index = np.digitize(scores, interior, right=True)
    width = interior.size + 1

    count = np.bincount(index, minlength=width).astype(float)
    total_score = np.bincount(index, weights=scores, minlength=width)
    total_target = np.bincount(index, weights=targets, minlength=width)

    # empty bins divide by zero; the guard keeps the warning out of a driver's output
    filled = count > 0
    safe = np.where(filled, count, 1.0)
    return CalibrationCurve(
        mean_score=np.where(filled, total_score / safe, np.nan),
        observed_rate=np.where(filled, total_target / safe, np.nan),
        count=count,
        edges=edges,
    )


class FlexibleCalibrationCurve(NamedTuple):
    """A smooth calibration curve.

    Attributes:
        grid_score (np.ndarray): Predicted probabilities the curve is evaluated on.
        fitted_rate (np.ndarray): The smoothed observed rate at each of those.
        intercept (float): Calibration intercept.
        slope (float): Calibration slope.
    """

    grid_score: np.ndarray
    fitted_rate: np.ndarray
    intercept: float
    slope: float


def _logit(probabilities: np.ndarray) -> np.ndarray:
    """The log-odds of a probability, clipped away from the asymptotes."""
    clipped = np.clip(np.asarray(probabilities, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def calibration_intercept_slope(
    scores: np.ndarray, targets: np.ndarray
) -> tuple[float, float]:
    """The calibration curve intercept and slope in logit-space.

    Args:
        scores (np.ndarray): Predicted probabilities, shaped (n,).
        targets (np.ndarray): Binary labels, shaped (n,).

    Returns:
        tuple[float, float]: (intercept, slope).

    Raises:
        ValueError: If the arrays disagree in length, or the labels are one class.
    """
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if scores.shape != targets.shape:
        raise ValueError(
            f"{scores.shape} scores against {targets.shape} targets; they do not "
            f"describe the same rows."
        )
    if len(np.unique(targets)) < 2:
        raise ValueError("Calibration needs both classes present.")

    logit = _logit(scores)
    slope = float(
        LogisticRegression(penalty=None, max_iter=MAX_ITER)
        .fit(logit.reshape(-1, 1), targets)
        .coef_[0][0]
    )
    return _fixed_slope_intercept(logit, targets), slope


def _fixed_slope_intercept(logit: np.ndarray, targets: np.ndarray) -> float:
    """The intercept of `logit(p) + a`, fitted by Newton steps on `a` alone.

    Args:
        logit (np.ndarray): The model's log-odds.
        targets (np.ndarray): Binary labels.

    Returns:
        float: The shift `a` maximising the Bernoulli likelihood of `logit + a`.
    """
    shift = 0.0
    for _ in range(MAX_ITER):
        fitted = 1.0 / (1.0 + np.exp(-(logit + shift)))
        gradient = float(np.sum(targets - fitted))
        hessian = float(np.sum(fitted * (1.0 - fitted)))
        if hessian <= 0.0:
            break
        step = gradient / hessian
        shift += step
        if abs(step) < 1e-10:
            break
    return shift


def flexible_calibration_curve(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    knots: int = 5,
    grid: int = 100,
    grid_scores: np.ndarray | None = None,
) -> FlexibleCalibrationCurve:
    """Generates a FlexibleCalibrationCurve describing a spline curve.

    This function fits a restricted cubic spline in `logit(p)` against
    the observed outcome. We find on logit-space to account for low
    prevalence.

    Args:
        scores (np.ndarray): Predicted probabilities, shaped (n,).
        targets (np.ndarray): Binary labels, shaped (n,).
        knots (int): Spline knots, placed at quantiles of the logit.
        grid (int): How many points to evaluate the fitted curve at. Ignored when
            `grid_scores` is given.
        grid_scores (np.ndarray | None): Evaluate at exactly these probabilities
            instead.

    Returns:
        FlexibleCalibrationCurve: The grid, the fitted rate on it, the intercept
        and slope.

    Raises:
        ValueError: If the arrays disagree in length, the labels are one class, or
            `knots` is below two.
    """
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if scores.shape != targets.shape:
        raise ValueError(
            f"{scores.shape} scores against {targets.shape} targets; they do not "
            f"describe the same rows."
        )
    if len(np.unique(targets)) < 2:
        raise ValueError("Calibration needs both classes present.")
    if knots < 2:
        raise ValueError(f"A spline needs at least two knots, got {knots}.")

    logit = _logit(scores).reshape(-1, 1)
    spline = SplineTransformer(
        n_knots=knots, degree=3, knots="quantile", extrapolation="linear"
    ).fit(logit)
    model = LogisticRegression(penalty=None, max_iter=MAX_ITER).fit(
        spline.transform(logit), targets
    )

    if grid_scores is None:
        grid_scores = np.quantile(scores, np.linspace(0.01, 0.999, grid))
    grid_scores = np.asarray(grid_scores, dtype=float)
    fitted = model.predict_proba(spline.transform(_logit(grid_scores).reshape(-1, 1)))
    intercept, slope = calibration_intercept_slope(scores, targets)
    return FlexibleCalibrationCurve(
        grid_score=grid_scores,
        fitted_rate=fitted[:, 1],
        intercept=intercept,
        slope=slope,
    )


def precision_recall_at_k(
    scores: np.ndarray, targets: np.ndarray, *, k: float
) -> tuple[float, float]:
    """Precision and recall on the top `k` fraction of predictions by magnitude.

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
    # pinned version of numpy doesn't implement 'descending'
    top = np.argsort(-scores, kind="stable")[:flagged]

    hits = float(targets[top].sum())
    positives = float(targets.sum())
    return hits / flagged, (hits / positives if positives else float("nan"))


def binary_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Scores one set of predictions against its labels.

    Args:
        scores (np.ndarray): The predicted probabilities, shaped (n,).
        targets (np.ndarray): The binary labels, shaped (n,).

    Returns:
        dict[str, float]: `auprc`, `auroc`, `brier`, `ece`, `precision_at_1pct`,
            `recall_at_1pct`, `precision_at_5pct`, `base_rate`, `n` and `n_positive`.

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
