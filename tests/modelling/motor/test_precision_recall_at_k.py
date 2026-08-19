"""Testing suite for `precision_recall_at_k`.

This is the metric the clinical framing rests on -- given an alert budget, how many of
the flagged landmarks are real. It cuts at a hard row index, so the flagged count, the
rounding that produces it and the tie-break at the boundary are all load-bearing.
"""

import numpy as np
import pytest

from thesis.modelling.motor.training import precision_recall_at_k


def arrays(scores: list[float], targets: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """One cohort as the two arrays the metric takes."""
    return np.array(scores, dtype=float), np.array(targets, dtype=float)


def test_the_top_fraction_is_flagged_by_score() -> None:
    """Ten rows at k=0.5 flags five: three of them positive, of four in the cohort."""
    scores, targets = arrays(
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )

    precision, recall = precision_recall_at_k(scores, targets, k=0.5)

    assert precision == pytest.approx(0.6)
    assert recall == pytest.approx(0.75)


def test_the_flagged_count_floors_at_one() -> None:
    """A 1% budget over four rows rounds to zero, which would divide by zero.

    The floor is what lets the training loop report p@1% on a small evaluation
    subsample instead of returning NaN.
    """
    scores, targets = arrays([0.9, 0.8, 0.3, 0.1], [1.0, 0.0, 1.0, 0.0])

    precision, recall = precision_recall_at_k(scores, targets, k=0.01)

    assert precision == 1.0
    assert recall == pytest.approx(0.5)


def test_the_count_rounds_to_even_at_a_half() -> None:
    """Five rows at k=0.5 is 2.5, and Python rounds that to 2, not 3.

    Both flagged rows are positive and the cohort holds exactly two positives, so
    precision and recall are both 1.0. Rounding up instead would flag a negative
    third row and read 2/3.
    """
    scores, targets = arrays([0.9, 0.8, 0.7, 0.6, 0.5], [1.0, 1.0, 0.0, 0.0, 0.0])

    assert precision_recall_at_k(scores, targets, k=0.5) == (1.0, 1.0)


def test_a_full_budget_flags_everything() -> None:
    """At k=1 precision is the base rate and recall is one, by construction."""
    scores, targets = arrays([0.9, 0.8, 0.3, 0.1], [1.0, 0.0, 1.0, 0.0])

    assert precision_recall_at_k(scores, targets, k=1.0) == (0.5, 1.0)


def test_recall_is_nan_without_a_positive() -> None:
    """No positive to recall; precision is still a real zero, not undefined.

    A NaN recall is a real condition on a small subsample, which is why the caller
    filters NaN out of bootstrap draws rather than treating it as an error.
    """
    scores, targets = arrays([0.9, 0.1], [0.0, 0.0])

    precision, recall = precision_recall_at_k(scores, targets, k=0.5)

    assert precision == 0.0
    assert np.isnan(recall)


def test_ties_are_broken_by_row_order() -> None:
    """A constant predictor flags the first row, not the best-labelled one.

    All four scores tie, so the stable sort keeps the given order and flags row 0, a
    negative. A tie-break that quietly preferred positives would report a useless
    model as perfect.
    """
    scores, targets = arrays([0.5, 0.5, 0.5, 0.5], [0.0, 1.0, 1.0, 1.0])

    assert precision_recall_at_k(scores, targets, k=0.25) == (0.0, 0.0)


@pytest.mark.parametrize("k", [0.0, -0.1, 1.5])
def test_a_budget_outside_the_unit_interval_is_refused(k: float) -> None:
    """Zero flags nothing and more than one flags rows that do not exist."""
    scores, targets = arrays([0.9, 0.1], [1.0, 0.0])

    with pytest.raises(ValueError, match="fraction in"):
        precision_recall_at_k(scores, targets, k=k)
