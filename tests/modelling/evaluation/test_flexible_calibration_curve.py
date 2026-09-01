"""Tests for the flexible calibration curve and the weak-level intercept and slope.

These are the only metrics in the project that FIT something, so the tests are built
around recovering a distortion of known size rather than around literal expected
frames: a model whose logits are stretched by `k` must come back with slope `1/k`,
and one shifted by `a` with intercept `-a`.
"""

import numpy as np
import pytest

from thesis.modelling.evaluation.metrics import (
    calibration_intercept_slope,
    flexible_calibration_curve,
)

TOLERANCE = 0.03


@pytest.fixture
def calibrated() -> tuple[np.ndarray, np.ndarray]:
    """Probabilities and labels drawn so the model is perfectly calibrated.

    The beta is shaped to this task's roughly 3.4% prevalence with a long right tail,
    because a spline fitted on a uniform score has no skew to cope with and would
    pass a test the real fold could fail.
    """
    rng = np.random.default_rng(0)
    scores = rng.beta(0.6, 17.0, 200_000)
    return scores, (rng.random(scores.size) < scores).astype(float)


def distort(scores: np.ndarray, *, stretch: float = 1.0, shift: float = 0.0):
    """The same scores with their logits scaled and shifted."""
    logit = np.log(scores / (1.0 - scores))
    return 1.0 / (1.0 + np.exp(-(stretch * logit + shift)))


def test_perfect_model_reads_intercept_zero_slope_one(calibrated):
    """A calibrated model must come back at the reference point."""
    intercept, slope = calibration_intercept_slope(*calibrated)
    assert intercept == pytest.approx(0.0, abs=TOLERANCE)
    assert slope == pytest.approx(1.0, abs=TOLERANCE)


@pytest.mark.parametrize("stretch", [0.5, 1.5, 2.0])
def test_slope_recovers_the_inverse_of_the_stretch(calibrated, stretch):
    """Scaling the logits by k must read back as slope 1/k."""
    scores, targets = calibrated
    _, slope = calibration_intercept_slope(distort(scores, stretch=stretch), targets)
    assert slope == pytest.approx(1.0 / stretch, rel=0.05)


@pytest.mark.parametrize("shift", [-0.7, 0.4])
def test_intercept_recovers_minus_the_shift(calibrated, shift):
    """Shifting the logits by a must read back as intercept -a.

    The sign is the part worth pinning: a model shifted DOWN under-predicts, and the
    calibration intercept that corrects it is POSITIVE.
    """
    scores, targets = calibrated
    intercept, _ = calibration_intercept_slope(distort(scores, shift=shift), targets)
    assert intercept == pytest.approx(-shift, abs=TOLERANCE)


def test_intercept_is_not_taken_from_the_slope_fit(calibrated):
    """The intercept holds the slope at one, so a pure stretch barely moves it.

    Reading both numbers off one two-parameter fit is the plausible wrong
    implementation: there the intercept absorbs the stretch and moves a long way.
    """
    scores, targets = calibrated
    intercept, slope = calibration_intercept_slope(
        distort(scores, stretch=2.0), targets
    )
    assert slope == pytest.approx(0.5, rel=0.05)
    assert abs(intercept) > 0.5


def test_curve_tracks_the_diagonal_for_a_calibrated_model(calibrated):
    """Moderate calibration holds, so fitted rate must match predicted risk."""
    curve = flexible_calibration_curve(*calibrated, grid=20)
    assert np.allclose(curve.fitted_rate, curve.grid_score, rtol=0.15, atol=0.002)


def test_curve_detects_a_model_that_under_predicts(calibrated):
    """Shifting the logits down must put the whole curve ABOVE the diagonal.

    A shift is used rather than a stretch because the direction of a stretch is not
    obvious below p = 0.5: every logit here is negative, so scaling one by 2 makes
    the prediction SMALLER, not larger. That trap cost this test one revision, and
    the slope test above is the unambiguous check on stretching.
    """
    scores, targets = calibrated
    curve = flexible_calibration_curve(distort(scores, shift=-0.7), targets, grid=20)
    assert np.all(curve.fitted_rate > curve.grid_score)


def test_grid_scores_override_is_honoured(calibrated):
    """A supplied grid is used verbatim, so bootstrap draws share abscissae."""
    grid = np.array([0.001, 0.01, 0.05, 0.2])
    curve = flexible_calibration_curve(*calibrated, grid_scores=grid)
    assert np.array_equal(curve.grid_score, grid)
    assert curve.fitted_rate.shape == grid.shape


def test_fitted_rate_is_monotone_in_the_score(calibrated):
    """The fit is a spline in the logit, and the fold's ordering is preserved.

    Not guaranteed by the model class, so it is worth asserting: a curve that folds
    back on itself would mean the smoother has started fitting noise.
    """
    curve = flexible_calibration_curve(*calibrated, grid=50)
    assert np.all(np.diff(curve.fitted_rate) > 0)


@pytest.mark.parametrize(
    ("scores", "targets", "message"),
    [
        (np.array([0.1, 0.2]), np.array([0.0]), "same rows"),
        (np.array([0.1, 0.2]), np.array([1.0, 1.0]), "both classes"),
    ],
)
def test_guards_raise(scores, targets, message):
    """Mismatched rows and a single-class fold are refused, not fitted."""
    with pytest.raises(ValueError, match=message):
        flexible_calibration_curve(scores, targets)
    with pytest.raises(ValueError, match=message):
        calibration_intercept_slope(scores, targets)


def test_too_few_knots_raises(calibrated):
    """A spline needs knots; one is not a curve."""
    with pytest.raises(ValueError, match="at least two knots"):
        flexible_calibration_curve(*calibrated, knots=1)


def test_probabilities_at_the_asymptotes_do_not_produce_nan(calibrated):
    """Scores of exactly 0 and 1 are clipped, not turned into infinities."""
    scores, targets = calibrated
    scores = scores.copy()
    scores[:100], scores[100:200] = 0.0, 1.0
    curve = flexible_calibration_curve(scores, targets, grid=10)
    assert np.all(np.isfinite(curve.fitted_rate))
    assert np.isfinite(curve.intercept) and np.isfinite(curve.slope)
