"""Testing suite for `bootstrap_interval`.

The one-model interval, reported beside every headline score. It is not what any
delta rests on -- two of these overlap freely even when one model wins on nearly every
resample, which is why `paired_interval` exists -- but it is what a reader sees next
to a number, so its width has to be honest.
"""

import numpy as np
import pytest

from thesis.modelling.evaluation.intervals import (
    _draw_rows,
    _subject_groups,
    bootstrap_interval,
)
from thesis.modelling.evaluation.metrics import binary_metrics


def cohort(
    n_subjects: int = 40, rows: int = 10, seed: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A cohort whose landmarks are correlated WITHIN a subject, as the real one is.

    Each subject carries its own latent risk, and all of its landmarks are drawn at
    that risk. Rows generated independently would leave nothing for the subject-level
    grouping to account for, and the width tests below would be vacuous.
    """
    rng = np.random.default_rng(seed)
    risk = rng.random(n_subjects)
    subjects = np.repeat(np.arange(n_subjects), rows)
    targets = (rng.random(n_subjects * rows) < risk[subjects]).astype(float)
    scores = np.clip(
        risk[subjects] * 0.8 + rng.random(n_subjects * rows) * 0.2, 0.0, 1.0
    )
    return scores, targets, subjects


def test_the_bounds_are_ordered_and_bracket_the_point_estimate() -> None:
    """An interval that does not contain its own metric is measuring something else."""
    from thesis.modelling.evaluation.metrics import binary_metrics

    scores, targets, subjects = cohort()

    low, high = bootstrap_interval(scores, targets, subjects, resamples=200, seed=0)

    assert low <= binary_metrics(scores, targets)["auprc"] <= high


def test_the_same_seed_reproduces_the_bounds_exactly() -> None:
    """A quoted interval has to be recomputable, not merely re-estimable."""
    scores, targets, subjects = cohort()

    first = bootstrap_interval(scores, targets, subjects, resamples=50, seed=4)
    second = bootstrap_interval(scores, targets, subjects, resamples=50, seed=4)

    assert first == second


def test_a_wider_alpha_gives_a_narrower_interval() -> None:
    """`alpha` is the two-sided width; 0.05 keeps 95%, 0.5 keeps the middle half."""
    scores, targets, subjects = cohort()

    wide_low, wide_high = bootstrap_interval(
        scores, targets, subjects, resamples=200, seed=0, alpha=0.05
    )
    tight_low, tight_high = bootstrap_interval(
        scores, targets, subjects, resamples=200, seed=0, alpha=0.5
    )

    assert wide_high - wide_low > tight_high - tight_low


def test_the_bounds_are_the_two_sided_quantiles_of_the_draws() -> None:
    """`alpha` is split across both tails: 0.2 cuts at 10% and 90%, not 20% and 80%.

    Recomputes the draws at the same seed and asserts the exact quantile levels,
    because no coarser property distinguishes them -- a one-sided reading orders
    every interval the same way and only reports the wrong coverage.
    """
    scores, targets, subjects = cohort()
    alpha, resamples = 0.2, 40

    low, high = bootstrap_interval(
        scores, targets, subjects, resamples=resamples, seed=0, alpha=alpha
    )

    rng = np.random.default_rng(0)
    groups = _subject_groups(subjects)
    draws = [
        binary_metrics(scores[rows], targets[rows])["auprc"]
        for rows in (_draw_rows(groups, rng) for _ in range(resamples))
    ]

    assert low == pytest.approx(float(np.quantile(draws, alpha / 2)))
    assert high == pytest.approx(float(np.quantile(draws, 1 - alpha / 2)))


def test_correlated_landmarks_widen_the_interval() -> None:
    """The reason subjects are the sampling unit, asserted rather than assumed.

    The same 400 rows are read twice: once as 40 subjects of 10 correlated landmarks,
    once as 400 independent subjects. Treating them as independent -- which is what a
    row-level bootstrap does -- reports a materially tighter interval on identical
    data. Measured at 1.6x to 2.3x across seeds, so 1.5x is a floor, not the effect.
    """
    scores, targets, subjects = cohort()

    grouped = bootstrap_interval(scores, targets, subjects, resamples=200, seed=0)
    as_rows = bootstrap_interval(
        scores, targets, np.arange(subjects.size), resamples=200, seed=0
    )

    assert (grouped[1] - grouped[0]) > 1.5 * (as_rows[1] - as_rows[0])


def test_a_draw_without_a_positive_is_skipped_rather_than_poisoning_the_quantiles() -> (
    None
):
    """AUPRC is NaN on a single-class draw, and `np.quantile` propagates NaN.

    One positive subject in twenty means many draws hold no positive at all. The
    bounds must still come back as real numbers.
    """
    subjects = np.repeat(np.arange(20), 5)
    targets = np.zeros(100)
    targets[:5] = 1.0
    scores = np.linspace(0.0, 1.0, 100)

    low, high = bootstrap_interval(scores, targets, subjects, resamples=100, seed=0)

    assert np.isfinite(low) and np.isfinite(high)


def test_another_metric_can_be_bounded() -> None:
    """Calibration is reported with an interval too, not only discrimination."""
    scores, targets, subjects = cohort()

    low, high = bootstrap_interval(
        scores, targets, subjects, metric="brier", resamples=50, seed=0
    )

    assert 0.0 <= low <= high <= 1.0


def test_a_single_subject_has_no_between_subject_variance() -> None:
    """Every draw is the same rows, so the interval collapses to a point."""
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    targets = np.array([1.0, 0.0, 1.0, 0.0])

    low, high = bootstrap_interval(scores, targets, np.zeros(4), resamples=20, seed=0)

    assert low == pytest.approx(high)
