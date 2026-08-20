"""Testing suite for `binary_metrics`.

The cohort below is small enough to check by hand: sorted by score it is
positive, negative, positive, negative.
"""

import numpy as np
import pytest

from thesis.modelling.evaluation.metrics import binary_metrics

SCORES = np.array([0.9, 0.8, 0.3, 0.1])
TARGETS = np.array([1.0, 0.0, 1.0, 0.0])

EXPECTED = {
    # one of the two positives ranks first, the other third
    "auprc": 1.0 / 2 + 1.0 / 3,
    # three of the four positive-negative pairs are ordered correctly
    "auroc": 0.75,
    # (0.01 + 0.64 + 0.49 + 0.01) / 4
    "brier": 0.2875,
    # four rows into ten bins is one row each: (0.1 + 0.8 + 0.7 + 0.1) / 4
    "ece": 0.425,
    # a 1% budget over four rows floors to one flag, which hits
    "precision_at_1pct": 1.0,
    "recall_at_1pct": 0.5,
    "precision_at_5pct": 1.0,
    "base_rate": 0.5,
    "n": 4.0,
    "n_positive": 2.0,
}


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_each_metric_reads_its_hand_computed_value(key: str) -> None:
    """The whole record, one assertion per key, so a failure names the metric."""
    assert binary_metrics(SCORES, TARGETS)[key] == pytest.approx(EXPECTED[key])


def test_the_record_holds_exactly_the_documented_keys() -> None:
    """Callers index this dict by name; a renamed key is a silent KeyError upstream."""
    assert sorted(binary_metrics(SCORES, TARGETS)) == sorted(EXPECTED)


def test_ranking_metrics_are_nan_on_a_single_class() -> None:
    """A draw holding no positive is a real condition, not an error.

    Bootstrap draws over a rare outcome hit this, and the callers filter NaN out. The
    calibration and count columns stay defined, because they need no second class.
    """
    record = binary_metrics(np.array([0.2, 0.8]), np.array([0.0, 0.0]))

    assert np.isnan(record["auprc"])
    assert np.isnan(record["auroc"])
    assert record["brier"] == pytest.approx(0.34)
    assert record["base_rate"] == 0.0
    assert record["n_positive"] == 0.0


def test_ranking_metrics_ignore_a_monotone_rescaling() -> None:
    """AUPRC and AUROC read the ordering; Brier and ECE read the values.

    This is why the project can select on loss and report AUPRC as different
    quantities, and why an ensemble's ranking gain cannot be a calibration effect.
    """
    record = binary_metrics(SCORES, TARGETS)
    rescaled = binary_metrics(SCORES / 2.0, TARGETS)

    assert rescaled["auprc"] == pytest.approx(record["auprc"])
    assert rescaled["auroc"] == pytest.approx(record["auroc"])
    assert rescaled["brier"] != pytest.approx(record["brier"])


def test_mismatched_arrays_are_refused() -> None:
    """Scores and labels from different cohorts would score one against the other."""
    with pytest.raises(ValueError, match="do not match"):
        binary_metrics(SCORES, TARGETS[:2])


def test_an_empty_cohort_is_refused() -> None:
    """Every column would be a division by zero, several of them silently NaN."""
    with pytest.raises(ValueError, match="empty"):
        binary_metrics(np.array([]), np.array([]))


def test_a_perfect_ranker_reads_one_on_both_areas() -> None:
    """The upper end of the scale, so the metric is anchored at both ends."""
    record = binary_metrics(
        np.array([0.9, 0.8, 0.2, 0.1]), np.array([1.0, 1.0, 0.0, 0.0])
    )

    assert record["auprc"] == pytest.approx(1.0)
    assert record["auroc"] == pytest.approx(1.0)


def test_an_inverted_ranker_reads_the_floor_on_auroc() -> None:
    """Ranking every positive last is 0.0, not 0.5 -- worse than chance is reachable."""
    record = binary_metrics(
        np.array([0.1, 0.2, 0.8, 0.9]), np.array([1.0, 1.0, 0.0, 0.0])
    )

    assert record["auroc"] == pytest.approx(0.0)
