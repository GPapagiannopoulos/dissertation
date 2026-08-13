"""Testing suite for the ensemble diversity metrics.

Grouped per module rather than per helper, as `test_normalise.py` is: these are one
cohesive set of measurements over the same aligned matrix.
"""

from pathlib import Path

import numpy as np
import pytest

from thesis.modelling.ensemble.diversity import (
    Member,
    align_members,
    ensemble_gain,
    flag_overlap,
    load_member,
    off_diagonal,
    pairwise,
    rank_correlation,
    top_flagged,
)


def make_member(scores: list[float], targets: list[int], shift: int = 0) -> Member:
    """A member over `len(scores)` landmarks, one per subject."""
    n = len(scores)
    return Member(
        scores=np.array(scores, dtype=float),
        targets=np.array(targets, dtype=float),
        subjects=np.arange(n) + shift,
        times=np.arange(n) * 100,
    )


def test_a_bundle_round_trips(tmp_path: Path) -> None:
    """Four unnamed arrays in order, as `score_motor` saves them."""
    path = tmp_path / "bundle.npz"
    np.savez(path, np.array([0.1]), np.array([1.0]), np.array([7]), np.array([99]))

    member = load_member(path)

    assert member.scores.tolist() == [0.1]
    assert member.subjects.tolist() == [7]


def test_a_malformed_bundle_is_refused(tmp_path: Path) -> None:
    """Three arrays would silently bind times to subjects."""
    path = tmp_path / "bad.npz"
    np.savez(path, np.array([0.1]), np.array([1.0]), np.array([7]))

    with pytest.raises(ValueError, match="prediction bundle"):
        load_member(path)


def test_members_are_sorted_onto_a_common_order() -> None:
    """Row order is never trusted; a misalignment would read as diversity."""
    first = make_member([0.9, 0.1, 0.5], [1, 0, 0])
    shuffled = Member(*(column[[2, 0, 1]] for column in first))

    scores, targets, subjects = align_members([first, shuffled])

    assert scores[0].tolist() == scores[1].tolist()
    assert subjects.tolist() == [0, 1, 2]
    assert targets.tolist() == [1.0, 0.0, 0.0]


def test_one_member_is_refused() -> None:
    """Diversity is a pairwise quantity."""
    with pytest.raises(ValueError, match="at least two members"):
        align_members([make_member([0.5], [1])])


def test_different_cohorts_are_refused() -> None:
    """Scoring on different folds makes every comparison meaningless."""
    with pytest.raises(ValueError, match="different cohorts"):
        align_members([make_member([0.5, 0.4], [1, 0]), make_member([0.5], [1])])


def test_disagreeing_labels_are_refused() -> None:
    """The same landmark cannot carry two truths; one bundle is stale."""
    with pytest.raises(ValueError, match="disagrees"):
        align_members(
            [make_member([0.5, 0.4], [1, 0]), make_member([0.5, 0.4], [0, 0])]
        )


def test_different_landmarks_are_refused() -> None:
    """Same row count, different patients -- the pairing would be silent nonsense."""
    with pytest.raises(ValueError, match="different landmarks"):
        align_members(
            [make_member([0.5, 0.4], [1, 0]), make_member([0.5, 0.4], [1, 0], shift=50)]
        )


@pytest.mark.parametrize(
    ("k", "expected"), [(0.5, [0, 1]), (0.25, [0]), (1.0, [0, 1, 2, 3])]
)
def test_the_alert_budget_picks_the_top_scores(k: float, expected: list[int]) -> None:
    """A budget of k flags the k fraction with the highest scores."""
    scores = np.array([0.9, 0.8, 0.2, 0.1])

    assert top_flagged(scores, k).tolist() == expected


def test_an_empty_budget_is_refused() -> None:
    """Zero would flag nobody and make the overlap 0/0."""
    with pytest.raises(ValueError, match="fraction in"):
        top_flagged(np.array([0.5]), 0.0)


def test_at_least_one_row_is_always_flagged() -> None:
    """A 1% budget over a small set still has to name somebody."""
    assert top_flagged(np.array([0.9, 0.1]), 0.01).tolist() == [0]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([0.9, 0.8, 0.1, 0.2], [0.9, 0.8, 0.1, 0.2], 1.0),
        ([0.9, 0.8, 0.1, 0.2], [0.1, 0.2, 0.9, 0.8], 0.0),
        ([0.9, 0.8, 0.1, 0.2], [0.9, 0.1, 0.8, 0.2], 1 / 3),
    ],
)
def test_flag_overlap_is_the_jaccard_of_the_flagged(
    left: list[float], right: list[float], expected: float
) -> None:
    """Identical flagging is 1.0, disjoint is 0.0, and both sets are the same size."""
    overlap = flag_overlap(np.array(left), np.array(right), k=0.5)

    assert overlap == pytest.approx(expected)


def test_rank_correlation_ignores_calibration() -> None:
    """A monotone rescaling is the same ranking, and AUPRC reads the ranking."""
    scores = np.array([0.1, 0.4, 0.35, 0.8])

    assert rank_correlation(scores, scores / 10) == pytest.approx(1.0)
    assert rank_correlation(scores, -scores) == pytest.approx(-1.0)


def test_pairwise_is_symmetric_with_self_comparisons_on_the_diagonal() -> None:
    """`off_diagonal` then takes each pair once."""
    scores = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.4]])

    matrix = pairwise(scores, rank_correlation)

    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)
    assert off_diagonal(matrix).size == 3


def test_averaging_complementary_members_beats_both() -> None:
    """The whole premise: members that err on different rows cancel."""
    targets = np.array([1.0, 1.0, 0.0, 0.0])
    scores = np.array([[0.9, 0.2, 0.8, 0.1], [0.2, 0.9, 0.1, 0.8]])

    gain = ensemble_gain(scores, targets)

    assert gain["ensemble_auprc"] > gain["best_member_auprc"]
    assert gain["gain_over_best"] > 0


def test_the_interval_is_only_computed_when_subjects_are_given() -> None:
    """It is a subject-level bootstrap; without subjects there is no resampling unit."""
    targets = np.array([1.0, 0.0, 1.0, 0.0])
    scores = np.array([[0.9, 0.1, 0.8, 0.2], [0.8, 0.2, 0.9, 0.1]])

    assert "gain_over_best_lo" not in ensemble_gain(scores, targets)
    assert "gain_over_best_lo" in ensemble_gain(
        scores, targets, np.arange(4), resamples=5
    )
