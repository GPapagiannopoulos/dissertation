"""Tests for marking the admissions the patient did not leave alive.

The flag changes no label. It exists because truncating the horizon at
discharge reads "no AKI by discharge" as an observed negative, and that reading
is wrong when the admission ended in death: 8,500 admissions and 99,035
landmarks are negative only because the patient died first.
"""

from collections.abc import Callable
from datetime import datetime

import polars as pl
import pytest

from thesis.modelling.cohort.labeller import flag_in_hospital_death


@pytest.fixture
def make_death_events() -> Callable:
    """Returns a factory for MEDS_DEATH-shaped events.

    Stage 1 sources these from patients.dod, so every one of the 38,301 sits at
    midnight and carries no visit_id: death is a subject-level fact, and which
    admission it fell in has to be worked out from the dates.
    """

    def _make(**columns: list) -> pl.LazyFrame:
        defaults = {
            "subject_id": [1],
            "time": [datetime(2020, 1, 4)],
            "code": ["MEDS_DEATH"],
        }
        return pl.LazyFrame(
            defaults | columns,
            schema_overrides={
                "subject_id": pl.Int64,
                "time": pl.Datetime("us"),
                "code": pl.String,
            },
        )

    return _make


@pytest.mark.parametrize(
    ("death", "expected"),
    [
        pytest.param(datetime(2020, 1, 3), True, id="died_mid_admission"),
        pytest.param(datetime(2020, 1, 1), True, id="died_on_the_admission_day"),
        pytest.param(datetime(2020, 1, 6), True, id="died_on_the_discharge_day"),
        pytest.param(datetime(2019, 12, 31), False, id="died_before_the_admission"),
        pytest.param(datetime(2020, 1, 7), False, id="died_after_the_discharge"),
    ],
)
def test_flag_in_hospital_death_compares_dates_not_timestamps(
    make_admission_windows: Callable,
    make_death_events: Callable,
    death: datetime,
    expected: bool,
) -> None:
    """A death at midnight still counts against the day it discharged on.

    MEDS_DEATH is date-resolution, so a timestamp comparison puts every death
    before that day's discharge and misses it. Measured against
    hospital_expire_flag, the date rule catches all 11,559 in-hospital deaths
    among the 394,712 inpatient admissions; the timestamp rule misses 573.
    """
    windows = make_admission_windows(
        subject_id=[1],
        visit_id=[10],
        visit_code=["Visit/IP"],
        admittime=[datetime(2020, 1, 1, 8)],
        dischtime=[datetime(2020, 1, 6, 12)],
    )

    flagged = flag_in_hospital_death(windows, make_death_events(time=[death])).collect()

    assert flagged["died_in_hospital"].to_list() == [expected]


def test_flag_in_hospital_death_leaves_survivors_false(
    make_admission_windows: Callable, make_death_events: Callable
) -> None:
    """A subject with no MEDS_DEATH at all is False, never null.

    The left join yields a null death date for the 326,326 subjects who never
    die in the record, and a null flag would poison the sensitivity filter.
    """
    flagged = flag_in_hospital_death(
        make_admission_windows(), make_death_events(subject_id=[99])
    ).collect()

    assert flagged["died_in_hospital"].to_list() == [False] * 4
    assert flagged.schema["died_in_hospital"] == pl.Boolean


def test_flag_in_hospital_death_reads_only_the_death_code(
    make_admission_windows: Callable, make_normalised_events: Callable
) -> None:
    """Ordinary events share the subject and must not be mistaken for a death."""
    flagged = flag_in_hospital_death(
        make_admission_windows(), make_normalised_events()
    ).collect()

    assert flagged["died_in_hospital"].to_list() == [False] * 4


def test_flag_in_hospital_death_marks_only_the_admission_that_contains_it(
    make_admission_windows: Callable, make_death_events: Callable
) -> None:
    """Death is subject-level, so only the admission spanning it is flagged."""
    windows = make_admission_windows(
        subject_id=[1, 1],
        visit_id=[10, 20],
        visit_code=["Visit/IP", "Visit/IP"],
        admittime=[datetime(2020, 1, 1), datetime(2020, 3, 1)],
        dischtime=[datetime(2020, 1, 6), datetime(2020, 3, 6)],
    )

    flagged = flag_in_hospital_death(windows, make_death_events()).collect()

    assert flagged.sort("visit_id")["died_in_hospital"].to_list() == [True, False]


def test_flag_in_hospital_death_keeps_every_admission(
    make_admission_windows: Callable,
    make_death_events: Callable,
    window_schema: dict[str, pl.DataType],
) -> None:
    """The step annotates the cohort and must not narrow it or leak death_date."""
    windows = make_admission_windows()

    flagged = flag_in_hospital_death(windows, make_death_events())

    assert flagged.collect().height == windows.collect().height
    assert flagged.collect_schema().names() == [*window_schema, "died_in_hospital"]


def test_flag_in_hospital_death_survives_a_duplicated_death_event(
    make_admission_windows: Callable, make_death_events: Callable
) -> None:
    """Two death events for one subject must not fan the admission out.

    No subject carries two today, which is exactly why the guard is worth
    holding: nothing downstream would notice the admission counted twice.
    """
    deaths = make_death_events(
        subject_id=[1, 1],
        time=[datetime(2020, 1, 3), datetime(2020, 1, 3)],
        code=["MEDS_DEATH", "MEDS_DEATH"],
    )

    flagged = flag_in_hospital_death(make_admission_windows(), deaths).collect()

    assert flagged.height == 4


def test_flag_in_hospital_death_stays_lazy(
    make_admission_windows: Callable, make_death_events: Callable
) -> None:
    """The whole of stage 4 is one plan the run_* wrapper collects once."""
    assert isinstance(
        flag_in_hospital_death(make_admission_windows(), make_death_events()),
        pl.LazyFrame,
    )
