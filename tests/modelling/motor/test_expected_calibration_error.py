"""Testing suite for `expected_calibration_error`.

Calibration is the one reported metric with no sklearn implementation behind it, so
nothing else cross-checks it. It is also the kind of function that fails quietly:
wrong bin weighting or a dropped absolute value still returns a small plausible
number in probability units, which is exactly what a reader would accept.
"""

import numpy as np
import pytest

from thesis.modelling.motor.training import expected_calibration_error


def arrays(scores: list[float], targets: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """One cohort as the two arrays the metric takes."""
    return np.array(scores, dtype=float), np.array(targets, dtype=float)


def test_a_perfectly_calibrated_cohort_reads_zero() -> None:
    """Every bin's mean score equals its observed frequency, so there is no gap."""
    scores, targets = arrays([0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(scores, targets, bins=2) == 0.0


def test_the_gap_is_absolute_and_bins_do_not_cancel() -> None:
    """One over- and one under-confident bin must add, never sum to nothing.

    The lower bin predicts 0.0 against an observed 0.5, the upper 1.0 against an
    observed 0.5. Signed, they cancel exactly and the metric would read zero on a
    cohort that is badly calibrated in both directions.
    """
    scores, targets = arrays([0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0])

    assert expected_calibration_error(scores, targets, bins=2) == 0.5


def test_bins_are_weighted_by_their_own_count() -> None:
    """Five rows into two bins gives sizes 3 and 2, which are not interchangeable.

    Only the two-row bin carries a gap, of 1.0. Size-weighted that is 2/5 = 0.4; an
    unweighted mean over the two bins would read 0.5.
    """
    scores, targets = arrays([0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0])

    assert expected_calibration_error(scores, targets, bins=2) == pytest.approx(0.4)


def test_more_bins_than_rows_degrades_to_one_row_each() -> None:
    """`bins` is capped at the row count, so the metric stays defined on a subsample.

    With one row per bin the result is the mean absolute residual: the sorted gaps are
    0.2, 0.5 and 0.1.
    """
    scores, targets = arrays([0.2, 0.9, 0.5], [0.0, 1.0, 1.0])

    assert expected_calibration_error(scores, targets, bins=10) == pytest.approx(
        0.8 / 3
    )


def test_one_bin_is_the_gap_between_the_two_means() -> None:
    """The coarsest possible reading: mean predicted against mean observed."""
    scores, targets = arrays([0.6, 0.6, 0.6, 0.6], [1.0, 1.0, 0.0, 0.0])

    assert expected_calibration_error(scores, targets, bins=1) == pytest.approx(0.1)


def test_tied_scores_are_split_in_their_original_order() -> None:
    """A constant predictor still bins, because the sort is stable, not by value.

    All four scores tie at 0.6, so the two bins are rows 0-1 and rows 2-3 as given.
    Their observed rates are 1.0 and 0.0, for gaps of 0.4 and 0.6. A cohort this
    miscalibrated must not hide behind its ties.
    """
    scores, targets = arrays([0.6, 0.6, 0.6, 0.6], [1.0, 1.0, 0.0, 0.0])

    assert expected_calibration_error(scores, targets, bins=2) == pytest.approx(0.5)


def test_bins_are_cut_from_the_lowest_score_upward() -> None:
    """Sorting direction changes the grouping whenever the bins come out uneven.

    Five rows into two bins gives sizes 3 and 2, so ascending puts the three
    zero-scored rows together and descending puts one of them with the two
    one-scored rows. Only the ascending reading pairs each score with the rows that
    actually share it.
    """
    scores, targets = arrays([0.0, 0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0])

    assert expected_calibration_error(scores, targets, bins=2) == pytest.approx(0.6)


@pytest.mark.parametrize("bins", [0, -1])
def test_a_non_positive_bin_count_is_refused(bins: int) -> None:
    """`np.array_split` would raise something unreadable, or divide by zero."""
    scores, targets = arrays([0.5, 0.5], [1.0, 0.0])

    with pytest.raises(ValueError, match="at least one bin"):
        expected_calibration_error(scores, targets, bins=bins)


def test_the_result_is_in_probability_units() -> None:
    """A predictor uniformly wrong by 0.25 reads 0.25, whatever the binning."""
    scores, targets = arrays([0.25, 0.25, 0.25, 0.25], [0.0, 0.0, 0.0, 0.0])

    for bins in (1, 2, 4, 10):
        assert expected_calibration_error(scores, targets, bins=bins) == pytest.approx(
            0.25
        )
