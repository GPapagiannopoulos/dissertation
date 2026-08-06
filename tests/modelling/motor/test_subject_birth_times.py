"""Testing suite for lifting each subject's time origin out of the event stream.

Every age in the pipeline is measured from this one row, and it is the input most
easily assembled wrongly at a call site: taken after tokenisation it comes back empty,
because `MEDS_BIRTH` carries no OMOP concept and is dropped. The failure is quiet
downstream -- an empty birth table makes the inner join drop every subject, and the
sequence builder returns nothing at all rather than raising.
"""

from collections.abc import Callable
from datetime import datetime

import polars as pl
import pytest

from thesis.modelling.motor.sequences import subject_birth_times

BIRTH = datetime(2100, 1, 1)


@pytest.fixture
def make_events() -> Callable:
    """Returns a factory for raw MEDS events, from (subject, code, time) triples."""

    def _make(*rows: tuple[int, str, datetime]) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "subject_id": [subject for subject, _, _ in rows],
                "code": [code for _, code, _ in rows],
                "time": [time for _, _, time in rows],
            },
            schema_overrides={"subject_id": pl.Int64, "time": pl.Datetime("us")},
        ).lazy()

    return _make


def test_takes_the_birth_row(make_events: Callable) -> None:
    """The origin is the MEDS_BIRTH event's own timestamp."""
    events = make_events(
        (1, "MEDS_BIRTH", BIRTH), (1, "SNOMED/1", datetime(2150, 6, 1))
    )

    result = subject_birth_times(events).collect()

    assert result.to_dicts() == [{"subject_id": 1, "birth": BIRTH}]


def test_ignores_every_other_code(make_events: Callable) -> None:
    """The first event is not the birth: demographics share its timestamp.

    Taking the earliest event instead would be right for most subjects and wrong for
    any whose record opens before their birth row is written.
    """
    events = make_events(
        (1, "Gender/M", datetime(2099, 1, 1)), (1, "MEDS_BIRTH", BIRTH)
    )

    result = subject_birth_times(events).collect()

    assert result["birth"].to_list() == [BIRTH]


def test_one_row_per_subject(make_events: Callable) -> None:
    """This is joined onto every event, so a second row would double the timeline."""
    events = make_events(
        (1, "MEDS_BIRTH", BIRTH),
        (2, "MEDS_BIRTH", datetime(2101, 1, 1)),
        (3, "MEDS_BIRTH", datetime(2102, 1, 1)),
    )

    result = subject_birth_times(events).collect()

    assert result.height == 3
    assert result["subject_id"].n_unique() == 3


def test_a_duplicated_origin_collapses_to_the_earliest(make_events: Callable) -> None:
    """Malformed input must not multiply a subject's events through the join.

    Two birth rows would otherwise give every one of that subject's events twice, at
    two different ages, and the sequence would be twice as long as the record.
    """
    events = make_events(
        (1, "MEDS_BIRTH", datetime(2101, 1, 1)), (1, "MEDS_BIRTH", BIRTH)
    )

    result = subject_birth_times(events).collect()

    assert result.to_dicts() == [{"subject_id": 1, "birth": BIRTH}]


def test_a_subject_without_a_birth_is_absent(make_events: Callable) -> None:
    """Absent here means dropped by the builder's inner join, which is the policy.

    Returning a null origin instead would give that subject null ages, and the encoder
    would receive a NaN rotary table rather than a missing subject.
    """
    events = make_events((1, "MEDS_BIRTH", BIRTH), (2, "SNOMED/1", BIRTH))

    result = subject_birth_times(events).collect()

    assert result["subject_id"].to_list() == [1]


def test_no_births_yields_an_empty_frame(make_events: Callable) -> None:
    """The shape a caller gets if this is run after tokenisation rather than before."""
    result = subject_birth_times(make_events((1, "SNOMED/1", BIRTH))).collect()

    assert result.height == 0
