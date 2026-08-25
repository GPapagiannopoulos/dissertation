"""Testing suite for `_subject_groups` and `_draw_rows`.

Grouped in one file, as `test_diversity.py` is: they are two halves of one mechanism,
and neither is meaningful alone. This is where the intervals get their width -- the
12-hourly grid puts ~9 correlated landmarks inside one admission, so resampling rows
would report an interval several times too narrow while looking entirely normal.
"""

import numpy as np

from thesis.modelling.evaluation.intervals import _draw_rows, _subject_groups


def test_rows_are_grouped_by_subject_in_sorted_subject_order() -> None:
    """Subjects arrive interleaved and out of order; the groups collect them anyway."""
    groups = _subject_groups(np.array([7, 3, 7, 3, 9]))

    assert [group.tolist() for group in groups] == [[1, 3], [0, 2], [4]]


def test_every_row_belongs_to_exactly_one_group() -> None:
    """A dropped or duplicated row would bias every draw built on these groups."""
    subjects = np.array([2, 2, 5, 1, 5, 5, 1])

    collected = np.sort(np.concatenate(_subject_groups(subjects)))

    assert collected.tolist() == list(range(subjects.size))


def test_one_row_per_subject_gives_one_group_each() -> None:
    """The degenerate case the row-level bootstrap would be correct for."""
    groups = _subject_groups(np.array([4, 1, 9]))

    assert [group.tolist() for group in groups] == [[1], [0], [2]]


def test_a_draw_takes_whole_subjects() -> None:
    """A drawn subject brings all its landmarks, which is the point of the grouping.

    Subject 0 owns three rows, subjects 1 and 2 one each. However many times subject 0
    is drawn, its rows arrive in complete threes -- so the count of its rows in any
    draw is a multiple of three.
    """
    groups = _subject_groups(np.array([0, 0, 0, 1, 2]))
    rng = np.random.default_rng(0)

    for _ in range(50):
        rows = _draw_rows(groups, rng)
        assert np.isin(rows, [0, 1, 2]).sum() % 3 == 0


def test_a_draw_keeps_the_subject_count_and_resamples_with_replacement() -> None:
    """Three subjects are drawn from three, with repeats, so rows may appear twice."""
    groups = _subject_groups(np.array([0, 1, 2]))
    rng = np.random.default_rng(0)

    draws = [_draw_rows(groups, rng) for _ in range(50)]

    assert all(len(rows) == 3 for rows in draws)
    assert any(len(np.unique(rows)) < 3 for rows in draws)


def test_draws_are_reproducible_from_the_generator() -> None:
    """Two runs of a bootstrap at one seed must agree exactly, not just in law."""
    groups = _subject_groups(np.array([0, 0, 1, 2, 2]))

    first = [_draw_rows(groups, np.random.default_rng(3)) for _ in range(5)]
    second = [_draw_rows(groups, np.random.default_rng(3)) for _ in range(5)]

    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_a_single_subject_draws_itself_every_time() -> None:
    """With no between-subject variation there is nothing for an interval to find."""
    groups = _subject_groups(np.array([8, 8, 8]))
    rng = np.random.default_rng(0)

    for _ in range(10):
        assert _draw_rows(groups, rng).tolist() == [0, 1, 2]
