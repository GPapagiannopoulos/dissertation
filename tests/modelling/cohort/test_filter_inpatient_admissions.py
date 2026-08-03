"""Tests for the cohort restriction to encounters with an inpatient stay."""

from collections.abc import Callable
from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.modelling.cohort.labeller import filter_inpatient_admissions


@pytest.mark.parametrize(
    ("visit_code", "kept"),
    [
        pytest.param("Visit/IP", True, id="inpatient"),
        pytest.param("Visit/ERIP", True, id="ed_then_inpatient"),
        pytest.param("Visit/ER", False, id="ed_only"),
        pytest.param("Visit/OP", False, id="outpatient"),
        pytest.param(None, False, id="null_visit_code"),
    ],
)
def test_filter_inpatient_admissions_applies_the_cohort_restriction(
    make_admission_windows: Callable, visit_code: str | None, kept: bool
) -> None:
    """Both inpatient types survive; ED-only and outpatient encounters do not.

    Visit/ERIP carries 177,459 admissions and 14,897 of the AKI positives, so
    dropping it alongside Visit/ER would cost 43% of the label. Visit/ER and
    Visit/OP sit at a 0.2% positive rate against the inpatient 8.5%: a
    hospital-acquired diagnosis on an encounter with no inpatient stay is a
    coding artefact, not a case.
    """
    windows = make_admission_windows(
        subject_id=[1],
        visit_id=[10],
        visit_code=[visit_code],
        admittime=[datetime(2020, 1, 1, 8)],
        dischtime=[datetime(2020, 1, 6, 12)],
    )

    assert (filter_inpatient_admissions(windows).collect().height == 1) is kept


def test_filter_inpatient_admissions_keeps_the_rows_intact(
    make_admission_windows: Callable, window_schema: dict[str, pl.DataType]
) -> None:
    """The step only removes rows: no column is touched and none is reordered."""
    assert_frame_equal(
        filter_inpatient_admissions(make_admission_windows()).collect(),
        pl.DataFrame(
            {
                "subject_id": [1, 2],
                "visit_id": [10, 20],
                "visit_code": ["Visit/IP", "Visit/ERIP"],
                "admittime": [datetime(2020, 1, 1, 8), datetime(2020, 2, 1, 7)],
                "dischtime": [datetime(2020, 1, 6, 12), datetime(2020, 2, 4, 10)],
            },
            schema=window_schema,
        ),
    )


def test_filter_inpatient_admissions_stays_lazy(
    make_admission_windows: Callable,
) -> None:
    """The filters compose into one plan that the run_* wrapper collects once."""
    assert isinstance(
        filter_inpatient_admissions(make_admission_windows()), pl.LazyFrame
    )


def test_filter_inpatient_admissions_survives_an_empty_input(
    make_admission_windows: Callable, window_schema: dict[str, pl.DataType]
) -> None:
    """An all-outpatient input yields an empty frame that keeps its dtypes."""
    windows = make_admission_windows(visit_code=["Visit/OP"] * 4)

    assert_frame_equal(
        filter_inpatient_admissions(windows).collect(),
        pl.DataFrame(schema=window_schema),
    )
