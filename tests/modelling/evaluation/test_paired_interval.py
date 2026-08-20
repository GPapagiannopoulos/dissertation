"""Testing suite for `paired_interval`.

Every delta the thesis claims rests on this function: the ensemble's margin over the
monolithic fine-tune, over its own best member, and the checkpoint-rule comparisons.
The pairing is the whole mechanism -- both models are scored on the SAME draw, so a
draw that happens to hold easy patients is easy for both and the cohort's own variance
cancels. Misaligned rows would silently turn it back into an unpaired comparison with
a spuriously tight interval, which is why the shape guard is load-bearing.
"""

import numpy as np
import pytest

from thesis.modelling.evaluation.intervals import (
    _draw_rows,
    _subject_groups,
    paired_interval,
)
from thesis.modelling.evaluation.metrics import binary_metrics


def cohort(
    n_subjects: int = 40, rows: int = 10, seed: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A cohort where `left` ranks better than `right`, with several rows a subject."""
    rng = np.random.default_rng(seed)
    size = n_subjects * rows
    subjects = np.repeat(np.arange(n_subjects), rows)
    targets = (rng.random(size) < 0.3).astype(float)
    left = np.clip(targets * 0.6 + rng.random(size) * 0.4, 0.0, 1.0)
    right = np.clip(targets * 0.2 + rng.random(size) * 0.8, 0.0, 1.0)
    return left, right, targets, subjects


def test_the_observed_difference_is_the_two_full_cohort_scores() -> None:
    """The point estimate is not a bootstrap mean; the draws only bound it."""
    from thesis.modelling.evaluation.metrics import binary_metrics

    left, right, targets, subjects = cohort()

    observed, _, _ = paired_interval(
        left, right, targets, subjects, resamples=20, seed=0
    )

    assert observed == pytest.approx(
        binary_metrics(left, targets)["auprc"] - binary_metrics(right, targets)["auprc"]
    )


def test_a_model_compared_with_itself_has_no_interval() -> None:
    """Every draw scores the same array twice, so every difference is exactly zero.

    An unpaired bootstrap would also centre on zero but would report a width, which is
    the failure this asserts against.
    """
    left, _, targets, subjects = cohort()

    assert paired_interval(left, left, targets, subjects, resamples=20, seed=0) == (
        0.0,
        0.0,
        0.0,
    )


def test_swapping_the_arms_negates_the_whole_result() -> None:
    """Same seed, same draws, so the two orderings are exact mirrors.

    This is the sharpest available check that both arms really are scored on one draw:
    if either side resampled independently, the bounds would only mirror in
    distribution, not to the last digit.
    """
    left, right, targets, subjects = cohort()

    observed, low, high = paired_interval(
        left, right, targets, subjects, resamples=50, seed=0
    )
    flipped, flipped_low, flipped_high = paired_interval(
        right, left, targets, subjects, resamples=50, seed=0
    )

    assert flipped == pytest.approx(-observed)
    assert flipped_low == pytest.approx(-high)
    assert flipped_high == pytest.approx(-low)


def test_a_clearly_better_arm_gives_an_interval_clear_of_zero() -> None:
    """The claim the project actually makes, in miniature."""
    left, right, targets, subjects = cohort()

    observed, low, high = paired_interval(
        left, right, targets, subjects, resamples=200, seed=0
    )

    assert observed > 0.0
    assert 0.0 < low < high


def test_the_interval_brackets_the_observed_difference() -> None:
    """A point estimate outside its own bounds means the draws are not of this data."""
    left, right, targets, subjects = cohort()

    observed, low, high = paired_interval(
        left, right, targets, subjects, resamples=200, seed=0
    )

    assert low <= observed <= high


def test_a_wider_alpha_gives_a_narrower_interval() -> None:
    """`alpha` is the two-sided width, so 0.5 cuts the quartiles, not the tails."""
    left, right, targets, subjects = cohort()

    _, wide_low, wide_high = paired_interval(
        left, right, targets, subjects, resamples=200, seed=0, alpha=0.05
    )
    _, tight_low, tight_high = paired_interval(
        left, right, targets, subjects, resamples=200, seed=0, alpha=0.5
    )

    assert wide_high - wide_low > tight_high - tight_low


def test_the_bounds_are_the_two_sided_quantiles_of_the_draws() -> None:
    """`alpha` is split across both tails: 0.05 cuts at 2.5% and 97.5%, not 5% and 95%.

    This one deliberately recomputes the draws with the same seed and asserts the
    exact quantile levels, because the levels are the thing under test and no
    coarser property distinguishes them -- a one-sided reading orders every interval
    the same way, it just reports the wrong coverage.
    """
    left, right, targets, subjects = cohort()
    alpha, resamples = 0.2, 40

    _, low, high = paired_interval(
        left, right, targets, subjects, resamples=resamples, seed=0, alpha=alpha
    )

    rng = np.random.default_rng(0)
    groups = _subject_groups(subjects)
    draws = []
    for _ in range(resamples):
        rows = _draw_rows(groups, rng)
        draws.append(
            binary_metrics(left[rows], targets[rows])["auprc"]
            - binary_metrics(right[rows], targets[rows])["auprc"]
        )

    assert low == pytest.approx(float(np.quantile(draws, alpha / 2)))
    assert high == pytest.approx(float(np.quantile(draws, 1 - alpha / 2)))


def test_the_same_seed_reproduces_the_bounds_exactly() -> None:
    """A quoted interval has to be recomputable, not merely re-estimable."""
    left, right, targets, subjects = cohort()

    first = paired_interval(left, right, targets, subjects, resamples=50, seed=7)
    second = paired_interval(left, right, targets, subjects, resamples=50, seed=7)

    assert first == second


def test_another_metric_can_be_bounded() -> None:
    """Selection and reporting are different quantities so the metric is a parameter."""
    left, right, targets, subjects = cohort()

    observed, low, high = paired_interval(
        left, right, targets, subjects, metric="brier", resamples=50, seed=0
    )

    assert observed < 0.0
    assert low <= observed <= high


@pytest.mark.parametrize("truncate", ["left", "right", "targets", "subjects"])
def test_arrays_of_different_lengths_are_refused(truncate: str) -> None:
    """One short array means the rows are not paired, whichever array it is."""
    left, right, targets, subjects = cohort()
    columns = {"left": left, "right": right, "targets": targets, "subjects": subjects}
    columns[truncate] = columns[truncate][:-1]

    with pytest.raises(ValueError, match="one row per landmark"):
        paired_interval(**columns, resamples=2)
