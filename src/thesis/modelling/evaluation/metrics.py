"""The metric definitions every arm reports through.

They live outside the training loop because the XGBoost baseline, the linear probe
and every ensemble analysis score the same way a fine-tune does; a second definition
anywhere would let two arms drift apart on what "AUPRC" means.
"""

from typing import NamedTuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import SplineTransformer

# the logit's asymptotes are clipped at the same point `training.py` clips its
# loss, so a landmark's log-odds here and its cross-entropy there agree
EPSILON = 1e-7
# both fits are convex in a handful of parameters; this is a ceiling, not a budget
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

    Equal-COUNT rather than equal-width, matching `expected_calibration_error`. At this
    task's 3.4% prevalence the scores pile up against zero, so ten equal-width bins put
    over 99% of rows in the first one and the diagram degenerates to a single point.

    Edges are deduplicated, so a score value spanning a boundary merges the two bins
    rather than leaving one empty. That is the only way this partition can differ from
    the one `expected_calibration_error` cuts, which splits ties across bins by rank.

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
    # the top of each group but the last, which is what `np.digitize(right=True)`
    # closes each bin on
    interior = {float(scores[group[-1]]) for group in groups[:-1] if group.size}
    return np.array([-np.inf, *sorted(interior), np.inf])


def calibration_curve(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 10,
    edges: np.ndarray | None = None,
) -> CalibrationCurve:
    """Predicted probability against observed frequency, bin by bin.

    This is the curve `expected_calibration_error` collapses to one number: that
    function sums `count * |mean_score - observed_rate|` over exactly these bins and
    divides by `n`. Reporting the curve therefore cannot disagree with the reported
    ECE, and shows WHERE a model is miscalibrated rather than only how much.

    Pass `edges` to hold the partition fixed across bootstrap draws. Recutting
    quantiles inside every draw moves the bins as well as the rates, and the resulting
    band measures bin drift rather than calibration uncertainty.

    Args:
        scores (np.ndarray): The predicted probabilities, shaped (n,).
        targets (np.ndarray): The binary labels, shaped (n,).
        bins (int): How many equal-count bins to cut. Ignored when `edges` is given.
        edges (np.ndarray | None): Boundaries from `calibration_edges`, to reuse a
            partition cut on other rows.

    Returns:
        CalibrationCurve: One entry per bin, ascending in score. A bin no row fell in
            carries NaN for both rates and a count of zero, which happens only under
            a supplied `edges` and is left in rather than dropped so every draw of a
            bootstrap keeps the same bins in the same order.

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
    """A smooth calibration curve, as Van Calster's MODERATE level asks for.

    Attributes:
        grid_score (np.ndarray): Predicted probabilities the curve is evaluated at,
            ascending, spanning the fold's own score range.
        fitted_rate (np.ndarray): The smoothed observed rate at each of those.
        intercept (float): Calibration intercept, the WEAK level: the logistic
            intercept when `logit(p)` is offset in with slope fixed at one. Zero is
            perfect; positive means the model under-predicts overall.
        slope (float): Calibration slope, also the weak level, from regressing the
            outcome on `logit(p)`. One is perfect; below one means predictions are
            too extreme, which is the overfitting signature.
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
    """Van Calster's WEAK calibration: the intercept and slope on the logit scale.

    The slope comes from regressing the outcome on `logit(p)` and the intercept from
    a second fit with that slope fixed at one, which is the standard pair -- taking
    both from one fit gives an intercept conditional on a slope that is not one, and
    the two numbers then do not mean what they are usually read as meaning.

    Args:
        scores (np.ndarray): Predicted probabilities, shaped (n,).
        targets (np.ndarray): Binary labels, shaped (n,).

    Returns:
        tuple[float, float]: The intercept (0 is perfect) and slope (1 is perfect).

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
    # penalty=None matters: sklearn's L2 default shrinks the slope toward zero, which
    # silently flatters an over-confident model on exactly the metric meant to catch it
    slope = float(
        LogisticRegression(penalty=None, max_iter=MAX_ITER)
        .fit(logit.reshape(-1, 1), targets)
        .coef_[0][0]
    )
    return _fixed_slope_intercept(logit, targets), slope


def _fixed_slope_intercept(logit: np.ndarray, targets: np.ndarray) -> float:
    """The intercept of `logit(p) + a`, fitted by Newton steps on `a` alone.

    This is the calibration intercept: the slope is pinned at one and only the shift
    is free. sklearn takes no offset term, and the one-parameter problem is convex
    with a closed-form derivative, so solving it directly is both cheaper and exact
    where manufacturing a design matrix for it would be neither.

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
    """A spline-smoothed calibration curve, Van Calster's MODERATE level.

    Moderate calibration asks that `E(Y | p_hat = p) = p` at every p, which the binned
    diagram approximates by grouping. This fits the relationship instead: a restricted
    cubic spline in `logit(p)` against the outcome, which is the form `rms::val.prob`
    uses. Three choices are load-bearing:

    * **the logit scale**, because at this task's 3.4% prevalence the scores pile
      against zero and a spline in probability space spends its flexibility on an
      empty right half;
    * **`penalty=None`**, because sklearn's L2 default shrinks the curve toward the
      diagonal and makes every model look better calibrated than it is;
    * **quantile knots with linear extrapolation**, so the ends are pinned by data
      rather than diverging past the last observation.

    The curve is APPARENT -- fitted on the rows it assesses, as `val.prob` also is.
    That is a statement about the fold, not a held-out claim.

    Args:
        scores (np.ndarray): Predicted probabilities, shaped (n,).
        targets (np.ndarray): Binary labels, shaped (n,).
        knots (int): Spline knots, placed at quantiles of the logit.
        grid (int): How many points to evaluate the fitted curve at. Ignored when
            `grid_scores` is given.
        grid_scores (np.ndarray | None): Evaluate at exactly these probabilities
            instead. Pass the full-fold grid across bootstrap draws, or each draw
            reports its curve at its own abscissae and the band is meaningless.

    Returns:
        FlexibleCalibrationCurve: The grid, the fitted rate on it, and the weak-level
            intercept and slope.

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
        # the lower bound is the 1st percentile, not the 0.1st: below the lowest knot
        # the spline is extrapolating linearly IN THE LOGIT, and at scores of order
        # 1e-6 that extrapolation is both unsupported and visibly non-monotone. At
        # 3.4% prevalence the excluded tail is scores under ~1e-4, which no operating
        # point reaches
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
    # pinned version of numpy doesn't implement 'descending'
    top = np.argsort(-scores, kind="stable")[:flagged]

    hits = float(targets[top].sum())
    positives = float(targets.sum())
    return hits / flagged, (hits / positives if positives else float("nan"))


def binary_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Scores one set of predictions against its labels.

    AUPRC leads because the task is 3.350% positive, where AUROC is dominated by the
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
