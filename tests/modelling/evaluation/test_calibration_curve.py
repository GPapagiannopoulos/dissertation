"""Testing suite for `calibration_curve` and `calibration_edges`.

The curve's whole claim is that it decomposes the reported ECE rather than offering a
second opinion on it, so most of what matters here is an identity: summing the curve's
size-weighted gaps must reproduce `expected_calibration_error` on the same bins. A
curve that binned slightly differently would still plot as a plausible reliability
diagram while no longer explaining any number the project reports.

The other half is the fixed-edge path, which exists so a bootstrap draw lands in the
bins cut on the full cohort. Recutting quantiles per draw moves the bins as well as
the rates, and nothing about the output would look wrong.
"""

import numpy as np
import pytest

from thesis.modelling.evaluation.metrics import (
    calibration_curve,
    calibration_edges,
    expected_calibration_error,
)


def arrays(scores: list[float], targets: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """One cohort as the two arrays the metric takes."""
    return np.array(scores, dtype=float), np.array(targets, dtype=float)


def ece_of(curve) -> float:
    """The size-weighted mean absolute gap the curve decomposes."""
    gaps = np.abs(curve.mean_score - curve.observed_rate)
    return float(np.nansum(curve.count * gaps) / curve.count.sum())


def test_each_bin_reports_its_own_mean_and_observed_rate() -> None:
    """Four rows into two bins: the lower predicts 0.0, the upper 1.0."""
    scores, targets = arrays([0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0])

    curve = calibration_curve(scores, targets, bins=2)

    assert curve.mean_score.tolist() == [0.0, 1.0]
    assert curve.observed_rate.tolist() == [0.5, 1.0]
    assert curve.count.tolist() == [2.0, 2.0]


def test_bins_are_ordered_by_score_not_by_arrival() -> None:
    """A shuffled cohort must give the same curve as a sorted one.

    The bins are quantiles of the score, so row order carries no information. An
    implementation splitting the raw array rather than its `argsort` passes every
    single-bin case and fails only here.
    """
    scores, targets = arrays([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0])

    curve = calibration_curve(scores, targets, bins=2)

    assert curve.mean_score.tolist() == [0.0, 1.0]
    assert curve.observed_rate.tolist() == [0.5, 1.0]


@pytest.mark.parametrize("bins", [2, 5, 10, 20])
def test_the_curve_sums_to_the_reported_ece(bins: int) -> None:
    """The identity the figure rests on, over a cohort with no tied scores."""
    rng = np.random.default_rng(0)
    scores = rng.random(500) ** 3
    targets = (rng.random(500) < scores).astype(float)

    curve = calibration_curve(scores, targets, bins=bins)

    assert ece_of(curve) == pytest.approx(
        expected_calibration_error(scores, targets, bins=bins)
    )


def test_edges_bracket_every_score() -> None:
    """The outermost edges are infinite, so no row can fall outside the bins."""
    scores, _ = arrays([0.1, 0.4, 0.9], [0.0, 0.0, 0.0])

    edges = calibration_edges(scores, bins=3)

    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    assert np.all(np.diff(edges) > 0)


def test_tied_scores_merge_bins_rather_than_emptying_one() -> None:
    """A value spanning a boundary cannot be split, so the two bins become one.

    This is the one place the curve's partition differs from
    `expected_calibration_error`, which splits ties across bins by rank. Merging is
    the honest reading: the model cannot distinguish rows it scored identically, so
    reporting them as two bins with different observed rates would invent a
    resolution the scores do not have.
    """
    scores, targets = arrays([0.1, 0.1, 0.1, 0.9], [0.0, 1.0, 0.0, 1.0])

    curve = calibration_curve(scores, targets, bins=4)

    assert curve.count.tolist() == [3.0, 1.0]
    assert curve.observed_rate.tolist() == [pytest.approx(1 / 3), 1.0]


def test_supplied_edges_are_used_instead_of_recutting() -> None:
    """A resample keeps the cohort's bins, so its counts are free to be uneven.

    Recut on these three rows the bins would hold one row each; held to the cohort's
    edges they hold two and one, which is the point.
    """
    cohort, _ = arrays([0.0, 0.1, 0.2, 0.9], [0.0, 0.0, 0.0, 0.0])
    edges = calibration_edges(cohort, bins=2)
    scores, targets = arrays([0.0, 0.1, 0.9], [1.0, 0.0, 1.0])

    curve = calibration_curve(scores, targets, edges=edges)

    assert curve.count.tolist() == [2.0, 1.0]
    assert curve.observed_rate.tolist() == [0.5, 1.0]


def test_a_bin_no_row_falls_in_survives_as_nan() -> None:
    """Empty bins are kept so every draw of a bootstrap carries the same bins.

    Dropping them would shorten one draw's curve and silently misalign the band
    against the bins it is supposed to bound.
    """
    cohort, _ = arrays([0.0, 0.1, 0.2, 0.9], [0.0, 0.0, 0.0, 0.0])
    edges = calibration_edges(cohort, bins=2)
    scores, targets = arrays([0.0, 0.1], [1.0, 0.0])

    curve = calibration_curve(scores, targets, edges=edges)

    assert curve.count.tolist() == [2.0, 0.0]
    assert curve.observed_rate[0] == 0.5
    assert np.isnan(curve.observed_rate[1])
    assert np.isnan(curve.mean_score[1])


def test_mismatched_lengths_raise() -> None:
    """Two arrays of different lengths cannot describe the same rows."""
    with pytest.raises(ValueError, match="same rows"):
        calibration_curve(np.array([0.1, 0.2]), np.array([1.0]))


@pytest.mark.parametrize("bins", [0, -1])
def test_a_non_positive_bin_count_raises(bins: int) -> None:
    """`np.array_split` accepts zero sections by raising something unreadable."""
    with pytest.raises(ValueError, match="at least one bin"):
        calibration_edges(np.array([0.1, 0.2]), bins=bins)
