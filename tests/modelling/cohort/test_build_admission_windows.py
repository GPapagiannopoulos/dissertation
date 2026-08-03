"""Tests for stage 4's window extractor, which every landmark is measured from."""

from collections.abc import Callable
from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.modelling.cohort.labeller import build_admission_windows

WINDOW_SCHEMA = {
    "subject_id": pl.Int64,
    "visit_id": pl.Int64,
    "visit_code": pl.String,
    "admittime": pl.Datetime("us"),
    "dischtime": pl.Datetime("us"),
}


def test_build_admission_windows_keeps_every_visit_type(
    make_normalised_events: Callable,
) -> None:
    """All four visit types survive; narrowing to inpatient is a separate step.

    Visit/ERIP is the trap: it does not start with "Visit/IP", so a prefix
    written against the inpatient code alone silently discards 177,459
    admissions and 43% of the AKI positives.
    """
    windows = build_admission_windows(make_normalised_events()).collect()

    assert windows["visit_code"].to_list() == [
        "Visit/IP",
        "Visit/ERIP",
        "Visit/ER",
        "Visit/OP",
    ]


def test_build_admission_windows_drops_events_that_only_carry_an_end(
    make_normalised_events: Callable,
) -> None:
    """The predicate is the code, not a populated end.

    inputevents and procedureevents fill end on 2.1M rows, none of which is an
    admission.
    """
    windows = build_admission_windows(make_normalised_events()).collect()

    assert "MIMIC_IV_INPUT/1" not in windows["visit_code"].to_list()
    assert windows.height == 4


def test_build_admission_windows_projects_and_renames(
    make_normalised_events: Callable,
) -> None:
    """Time and end become the window; the other sixteen columns are dropped."""
    windows = build_admission_windows(make_normalised_events()).collect()

    assert_frame_equal(
        windows,
        pl.DataFrame(
            {
                "subject_id": [1, 2, 3, 4],
                "visit_id": [10, 20, 30, 40],
                "visit_code": ["Visit/IP", "Visit/ERIP", "Visit/ER", "Visit/OP"],
                "admittime": [
                    datetime(2020, 1, 1, 8),
                    datetime(2020, 2, 1, 7),
                    datetime(2020, 3, 1, 6),
                    datetime(2020, 4, 1, 5),
                ],
                "dischtime": [
                    datetime(2020, 1, 6, 12),
                    datetime(2020, 2, 4, 10),
                    datetime(2020, 3, 1, 20),
                    datetime(2020, 4, 2, 5),
                ],
            },
            schema=WINDOW_SCHEMA,
        ),
    )


def test_build_admission_windows_stays_lazy(make_normalised_events: Callable) -> None:
    """The caller composes this with the rest of stage 4 before collecting once."""
    assert isinstance(build_admission_windows(make_normalised_events()), pl.LazyFrame)


def test_build_admission_windows_keeps_every_admission_of_a_subject(
    make_normalised_events: Callable,
) -> None:
    """One row per admission, not per subject: the grid is laid per admission."""
    events = make_normalised_events(subject_id=[1, 1, 1, 1, 1])

    windows = build_admission_windows(events).collect()

    assert windows["subject_id"].to_list() == [1, 1, 1, 1]
    assert windows["visit_id"].to_list() == [10, 20, 30, 40]


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(["MIMIC_IV_INPUT/1"] * 5, id="no_visit_events"),
        pytest.param(["Visit"] * 5, id="prefix_without_the_separator"),
    ],
)
def test_build_admission_windows_accepts_events_holding_no_admission(
    make_normalised_events: Callable, code: list[str]
) -> None:
    """An empty result keeps its dtypes, so the downstream join still resolves."""
    windows = build_admission_windows(make_normalised_events(code=code)).collect()

    assert_frame_equal(windows, pl.DataFrame(schema=WINDOW_SCHEMA))
