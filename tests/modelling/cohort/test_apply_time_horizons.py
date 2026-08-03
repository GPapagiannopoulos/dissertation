"""Tests for turning each landmark into the boolean the model is trained on.

The question asked at every landmark is: does AKI arrive within the horizon, or
by discharge, whichever comes first? Discharge truncates rather than censors
because hospital-acquired AKI is defined inside the admission, so an admission
that ends without it is an observed negative and not an unknown.
"""

from collections.abc import Callable
from datetime import datetime

import polars as pl
import pytest

from thesis.modelling.cohort.labeller import apply_time_horizons


@pytest.mark.parametrize(
    ("diagtime", "expected"),
    [
        pytest.param(datetime(2020, 1, 4), True, id="diagnosis_inside_the_horizon"),
        pytest.param(datetime(2020, 1, 5), True, id="diagnosis_exactly_at_the_end"),
        pytest.param(datetime(2020, 1, 6), False, id="diagnosis_beyond_the_horizon"),
        pytest.param(None, False, id="no_diagnosis_at_all"),
    ],
)
def test_apply_time_horizons_labels_against_the_horizon(
    make_landmark_grid: Callable, diagtime: datetime | None, expected: bool
) -> None:
    """The label is whether the onset lands in (landmark, landmark + horizon].

    The grid stops strictly before the onset, so a landmark never sits on its
    own diagnosis; the end of the horizon is inclusive, which is why an onset
    exactly 48h out is a positive.
    """
    labelled = apply_time_horizons(
        make_landmark_grid(diagtime=[diagtime]), "48h"
    ).collect()

    assert labelled["label"].to_list() == [expected]


def test_apply_time_horizons_never_emits_a_null_label(
    make_landmark_grid: Callable,
) -> None:
    """A null onset must read as False, not propagate into a null label.

    Kleene logic makes ``False & null`` False, so the is_not_null guard has to
    come first in the conjunction. A null reaching femr is not a usable label.
    """
    labelled = apply_time_horizons(make_landmark_grid(), "48h").collect()

    assert labelled["label"].null_count() == 0
    assert labelled.schema["label"] == pl.Boolean


@pytest.mark.parametrize(
    ("dischtime", "expected_end"),
    [
        pytest.param(
            datetime(2020, 2, 1), datetime(2020, 1, 5), id="discharge_after_the_horizon"
        ),
        pytest.param(
            datetime(2020, 1, 4), datetime(2020, 1, 4), id="discharge_truncates"
        ),
        pytest.param(
            datetime(2020, 1, 5),
            datetime(2020, 1, 5),
            id="discharge_exactly_at_the_horizon",
        ),
    ],
)
def test_apply_time_horizons_truncates_the_horizon_at_discharge(
    make_landmark_grid: Callable, dischtime: datetime, expected_end: datetime
) -> None:
    """The horizon ends at min(landmark + horizon, dischtime).

    955,891 landmarks, 32.2% of the grid, end early this way, with a median
    effective horizon of 21h against the nominal 48h.
    """
    labelled = apply_time_horizons(
        make_landmark_grid(dischtime=[dischtime]), "48h"
    ).collect()

    assert labelled["horizon_time"].to_list() == [expected_end]
    assert labelled.schema["horizon_time"] == pl.Datetime("us")


def test_apply_time_horizons_truncation_can_flip_a_label(
    make_landmark_grid: Callable,
) -> None:
    """An onset recorded after discharge falls outside the truncated horizon.

    34 MIMIC-IV admissions carry an onset later than their own discharge; those
    are the only admissions whose labels a truncated horizon changes.
    """
    grid = make_landmark_grid(
        dischtime=[datetime(2020, 1, 4)], diagtime=[datetime(2020, 1, 4, 12)]
    )

    assert apply_time_horizons(grid, "48h").collect()["label"].to_list() == [False]


@pytest.mark.parametrize(
    ("horizon", "expected_end"),
    [
        pytest.param("48h", datetime(2020, 1, 5), id="hours"),
        pytest.param("2d", datetime(2020, 1, 5), id="days"),
        pytest.param("1w", datetime(2020, 1, 10), id="weeks"),
    ],
)
def test_apply_time_horizons_parses_every_supported_unit(
    make_landmark_grid: Callable, horizon: str, expected_end: datetime
) -> None:
    """The amount is an integer to pl.duration, never the string it was sliced from.

    pl.duration reads a string argument as a column name, so passing "48"
    straight through raises ColumnNotFoundError rather than building 48 hours.
    """
    labelled = apply_time_horizons(make_landmark_grid(), horizon).collect()

    assert labelled["horizon_time"].to_list() == [expected_end]


@pytest.mark.parametrize(
    "horizon",
    [
        pytest.param("48m", id="unsupported_unit"),
        pytest.param("48", id="no_unit_at_all"),
    ],
)
def test_apply_time_horizons_rejects_an_unusable_horizon(
    make_landmark_grid: Callable, horizon: str
) -> None:
    """A bad argument value is a ValueError, not an AttributeError."""
    with pytest.raises(ValueError):
        apply_time_horizons(make_landmark_grid(), horizon)


def test_apply_time_horizons_keeps_the_grid_columns(
    make_landmark_grid: Callable, grid_schema: dict[str, pl.DataType]
) -> None:
    """Every landmark keeps its admission so the side table can be cut from it."""
    labelled = apply_time_horizons(make_landmark_grid(), "48h")

    assert set(grid_schema).issubset(labelled.collect_schema().names())


def test_apply_time_horizons_stays_lazy(make_landmark_grid: Callable) -> None:
    """The whole of stage 4 is one plan the run_* wrapper collects once."""
    assert isinstance(apply_time_horizons(make_landmark_grid(), "48h"), pl.LazyFrame)
