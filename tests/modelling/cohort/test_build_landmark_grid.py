"""Tests for the per-admission landmark grid, the heart of stage 4.

Every landmark is one prediction_time: the moment forecast from and the cutoff
features may be drawn up to.
"""

from collections.abc import Callable
from datetime import datetime

import polars as pl
import pytest

from thesis.modelling.cohort.labeller import build_landmark_grid


def _landmarks(grid: pl.LazyFrame) -> list[datetime]:
    """Reduces a grid to the landmarks it holds, which is what cases assert on."""
    return grid.collect()["prediction_time"].to_list()


def test_build_landmark_grid_starts_one_delta_after_admission(
    make_windows_with_onset: Callable,
) -> None:
    """The first landmark is one delta in, not at admission and not at the gate.

    A landmark AT admittime is structurally negative: the measured minimum onset
    offset across the 34,326 positives is 48.02h, so a 48h horizon from admittime
    reaches 48.00h and can never contain one.

    One delta in it can. A landmark at admittime + 12h forecasting 48h covers
    onsets in (48.02h, 60h], so it is a genuine prediction. Starting at the 48h
    gate instead would delete these.
    """
    grid = build_landmark_grid(make_windows_with_onset())

    assert _landmarks(grid)[0] == datetime(2020, 1, 1, 12)


def test_build_landmark_grid_spaces_landmarks_by_delta(
    make_windows_with_onset: Callable,
) -> None:
    """A negative admission runs the full grid to, but excluding, discharge."""
    grid = build_landmark_grid(make_windows_with_onset())

    assert _landmarks(grid) == [
        datetime(2020, 1, 1, 12),
        datetime(2020, 1, 2, 0),
        datetime(2020, 1, 2, 12),
        datetime(2020, 1, 3, 0),
        datetime(2020, 1, 3, 12),
        datetime(2020, 1, 4, 0),
        datetime(2020, 1, 4, 12),
        datetime(2020, 1, 5, 0),
        datetime(2020, 1, 5, 12),
    ]


@pytest.mark.parametrize(
    ("dischtime", "diagtime", "expected"),
    [
        pytest.param(
            datetime(2020, 1, 6),
            None,
            datetime(2020, 1, 5, 12),
            id="negative_admission_censors_at_discharge",
        ),
        pytest.param(
            datetime(2020, 1, 6),
            datetime(2020, 1, 4, 6),
            datetime(2020, 1, 4, 0),
            id="diagnosis_before_discharge_censors_at_diagnosis",
        ),
        pytest.param(
            datetime(2020, 1, 4),
            datetime(2020, 1, 5),
            datetime(2020, 1, 3, 12),
            id="discharge_before_diagnosis_censors_at_discharge",
        ),
    ],
)
def test_build_landmark_grid_censors_at_the_earlier_of_discharge_and_diagnosis(
    make_windows_with_onset: Callable,
    dischtime: datetime,
    diagtime: datetime | None,
    expected: datetime,
) -> None:
    """The censor is min(dischtime, diagtime), with a null onset meaning none.

    pl.min_horizontal ignores nulls rather than propagating them, which is what
    makes one expression serve both arms: a negative admission carries a null
    diagtime and falls back to its discharge. 34 MIMIC-IV positives are recorded
    as diagnosed after discharge; those censor at discharge.
    """
    grid = build_landmark_grid(
        make_windows_with_onset(dischtime=[dischtime], diagtime=[diagtime])
    )

    assert _landmarks(grid)[-1] == expected


@pytest.mark.parametrize(
    ("dischtime", "diagtime"),
    [
        pytest.param(datetime(2020, 1, 4), None, id="discharge_on_a_grid_point"),
        pytest.param(
            datetime(2020, 1, 6), datetime(2020, 1, 4), id="diagnosis_on_a_grid_point"
        ),
    ],
)
def test_build_landmark_grid_excludes_the_censor_itself(
    make_windows_with_onset: Callable, dischtime: datetime, diagtime: datetime | None
) -> None:
    """A landmark landing exactly on the censor is dropped, not kept.

    We do not want a prediction at the moment of diagnosis or of discharge, so
    the range is open at both ends. Both cases censor at 2020-01-04 00:00, which
    is a grid point, and neither emits it.
    """
    grid = build_landmark_grid(
        make_windows_with_onset(dischtime=[dischtime], diagtime=[diagtime])
    )

    assert _landmarks(grid) == [
        datetime(2020, 1, 1, 12),
        datetime(2020, 1, 2, 0),
        datetime(2020, 1, 2, 12),
        datetime(2020, 1, 3, 0),
        datetime(2020, 1, 3, 12),
    ]


@pytest.mark.parametrize(
    ("dischtime", "diagtime"),
    [
        pytest.param(datetime(2020, 1, 1, 6), None, id="stay_shorter_than_one_delta"),
        pytest.param(datetime(2020, 1, 1, 12), None, id="stay_of_exactly_one_delta"),
        pytest.param(
            datetime(2020, 1, 6),
            datetime(2020, 1, 1, 6),
            id="diagnosis_before_the_first_landmark",
        ),
    ],
)
def test_build_landmark_grid_emits_no_row_for_an_empty_grid(
    make_windows_with_onset: Callable, dischtime: datetime, diagtime: datetime | None
) -> None:
    """An admission with no room for a landmark contributes nothing at all.

    Polars explode turns an empty list into one null row rather than none, so the
    empty ranges have to be dropped explicitly, and a null prediction_time reaching
    femr is a label with no timestamp.

    Because the interval is open at both ends, a stay of exactly one delta is empty
    too: its only grid point is the excluded endpoint. The third case cannot arise
    in real data -- the HA-AKI gate forbids an onset inside the first 48h -- but the
    function must not depend on that.
    """
    grid = build_landmark_grid(
        make_windows_with_onset(dischtime=[dischtime], diagtime=[diagtime])
    ).collect()

    assert grid.height == 0
    assert grid.schema["prediction_time"] == pl.Datetime("us")


def test_build_landmark_grid_honours_a_custom_delta(
    make_windows_with_onset: Callable,
) -> None:
    """Landmark spacing is a parameter: the 12h default is a choice, not a law."""
    grid = build_landmark_grid(
        make_windows_with_onset(dischtime=[datetime(2020, 1, 2)]), "6h"
    )

    assert _landmarks(grid) == [
        datetime(2020, 1, 1, 6),
        datetime(2020, 1, 1, 12),
        datetime(2020, 1, 1, 18),
    ]


def test_build_landmark_grid_repeats_the_admission_on_every_landmark(
    make_windows_with_onset: Callable,
) -> None:
    """Each landmark keeps its admission's identity, which the join downstream needs.

    Admission identity is implicit in the timestamp, but carrying visit_id makes
    the per-admission metrics and the lead-time curve possible without
    reconstructing it.
    """
    windows = make_windows_with_onset(
        subject_id=[1, 1],
        visit_id=[10, 20],
        visit_code=["Visit/IP", "Visit/ERIP"],
        admittime=[datetime(2020, 1, 1), datetime(2020, 2, 1)],
        dischtime=[datetime(2020, 1, 4), datetime(2020, 2, 4)],
        diagtime=[None, None],
    )

    grid = build_landmark_grid(windows).collect()

    assert grid["visit_id"].to_list() == [10] * 5 + [20] * 5
    assert grid["subject_id"].to_list() == [1] * 10
    assert grid["visit_code"].to_list() == ["Visit/IP"] * 5 + ["Visit/ERIP"] * 5


def test_build_landmark_grid_stays_lazy(make_windows_with_onset: Callable) -> None:
    """The whole of stage 4 is one plan the run_* wrapper collects once."""
    assert isinstance(build_landmark_grid(make_windows_with_onset()), pl.LazyFrame)
