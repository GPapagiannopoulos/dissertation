"""Tests for the guard against admissions whose window runs backwards."""

from collections.abc import Callable
from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.modelling.cohort.labeller import filter_false_admissions


@pytest.mark.parametrize(
    ("dischtime", "kept"),
    [
        pytest.param(datetime(2020, 1, 6, 12), True, id="positive_los"),
        pytest.param(datetime(2020, 1, 1, 8, 0, 0, 1), True, id="one_microsecond_los"),
        pytest.param(datetime(2020, 1, 1, 8), False, id="zero_los"),
        pytest.param(datetime(2020, 1, 1, 7), False, id="negative_los"),
        pytest.param(None, False, id="null_dischtime"),
    ],
)
def test_filter_false_admissions_requires_a_forward_running_window(
    make_admission_windows: Callable, dischtime: datetime | None, kept: bool
) -> None:
    """A window must have strictly positive length to hold a landmark.

    MIMIC-IV holds 175 admissions discharged before they were admitted and 5
    discharged in the same minute. Neither can carry a grid, and a backwards
    window would make every downstream duration negative rather than empty.
    Null propagates through the comparison, so an absent dischtime drops too.
    """
    windows = make_admission_windows(
        subject_id=[1],
        visit_id=[10],
        visit_code=["Visit/IP"],
        admittime=[datetime(2020, 1, 1, 8)],
        dischtime=[dischtime],
    )

    assert (filter_false_admissions(windows).collect().height == 1) is kept


def test_filter_false_admissions_keeps_a_clean_frame_whole(
    make_admission_windows: Callable, window_schema: dict[str, pl.DataType]
) -> None:
    """On the 545,848 well-formed admissions the step is the identity."""
    windows = make_admission_windows()

    assert_frame_equal(
        filter_false_admissions(windows).collect(),
        windows.collect(),
    )
    assert filter_false_admissions(windows).collect_schema() == window_schema


def test_filter_false_admissions_is_indifferent_to_visit_type(
    make_admission_windows: Callable,
) -> None:
    """The two filters commute, so the run_* wrapper may apply them either way."""
    windows = make_admission_windows(
        admittime=[datetime(2020, 1, 2, 8)] + [datetime(2020, 1, 1, 8)] * 3,
        dischtime=[datetime(2020, 1, 1, 8)] + [datetime(2020, 1, 6, 12)] * 3,
    )

    assert filter_false_admissions(windows).collect()["visit_id"].to_list() == [
        20,
        30,
        40,
    ]


def test_filter_false_admissions_stays_lazy(make_admission_windows: Callable) -> None:
    """The filters compose into one plan that the run_* wrapper collects once."""
    assert isinstance(filter_false_admissions(make_admission_windows()), pl.LazyFrame)


def test_filter_false_admissions_survives_an_empty_input(
    make_admission_windows: Callable, window_schema: dict[str, pl.DataType]
) -> None:
    """An all-backwards input yields an empty frame that keeps its dtypes."""
    windows = make_admission_windows(
        admittime=[datetime(2020, 5, 1, 8)] * 4,
        dischtime=[datetime(2020, 1, 1, 8)] * 4,
    )

    assert_frame_equal(
        filter_false_admissions(windows).collect(),
        pl.DataFrame(schema=window_schema),
    )
