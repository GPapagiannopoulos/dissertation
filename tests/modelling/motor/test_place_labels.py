"""Testing suite for pinning landmarks onto sequence positions.

This is the leakage boundary of the whole experiment. A landmark is a moment, and the
features it is scored from are whatever position it lands on; attach it one event too
late and the model reads a measurement the clinician had not yet seen. Nothing
downstream can detect that -- the training curve improves, which is the wrong signal
entirely.

The second failure is quieter: an event in a chunk overlap exists at two positions,
and taking the wrong one gives a real position in a real sequence with almost none of
the subject's history in front of it.
"""

from collections.abc import Callable
from datetime import datetime, timedelta

import polars as pl
import pytest

from thesis.modelling.motor.sequences import (
    LABEL_SCHEMA,
    build_sequences,
    place_labels,
)

BIRTH = datetime(2100, 1, 1)
DAY = timedelta(days=1)


@pytest.fixture
def make_sequences() -> Callable:
    """Returns a factory for sequences built from one event per day per subject."""

    def _make(
        *,
        n: int = 4,
        subjects: tuple[int, ...] = (1,),
        length: int = 4,
        stride: int = 2,
    ) -> pl.LazyFrame:
        events = pl.DataFrame(
            {
                "subject_id": [s for s in subjects for _ in range(n)],
                "time": [BIRTH + i * DAY for _ in subjects for i in range(n)],
                "index": [10 + i for _ in subjects for i in range(n)],
            },
            schema_overrides={"subject_id": pl.Int64, "index": pl.UInt32},
        ).lazy()
        births = pl.DataFrame(
            {"subject_id": list(subjects), "birth": [BIRTH] * len(subjects)},
            schema_overrides={"subject_id": pl.Int64},
        ).lazy()
        return build_sequences(
            events, births, length=length, stride=stride, age_mean=0.0, age_std=1.0
        )

    return _make


@pytest.fixture
def make_labels() -> Callable:
    """Returns a factory for landmarks, timed in days from the shared birth."""

    def _make(*landmarks: tuple[int, float, bool]) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "subject_id": [subject for subject, _, _ in landmarks],
                "prediction_time": [BIRTH + days * DAY for _, days, _ in landmarks],
                "boolean_value": [value for _, _, value in landmarks],
            },
            schema_overrides={
                "subject_id": pl.Int64,
                "prediction_time": pl.Datetime("us"),
                "boolean_value": pl.Boolean,
            },
        ).lazy()

    return _make


def test_a_landmark_lands_on_the_last_event_before_it(
    make_sequences: Callable, make_labels: Callable
) -> None:
    """Between two events the earlier one wins. This IS the leakage boundary.

    A forward join would read the event after the prediction time, which is the
    measurement the model is meant to be predicting from having not seen.
    """
    result = place_labels(
        make_sequences(n=4, length=8), make_labels((1, 2.5, True))
    ).collect()

    assert result["position"].to_list() == [2]


def test_a_landmark_exactly_on_an_event_takes_that_event(
    make_sequences: Callable, make_labels: Callable
) -> None:
    """'At or before', not 'strictly before'.

    An admission landmark shares its timestamp with the admission event; excluding it
    would throw away the one event that says the encounter has begun.
    """
    result = place_labels(
        make_sequences(n=4, length=8), make_labels((1, 2.0, True))
    ).collect()

    assert result["position"].to_list() == [2]


def test_a_landmark_after_every_event_takes_the_last_one(
    make_sequences: Callable, make_labels: Callable
) -> None:
    """The record simply stops; the last position is still the right history."""
    result = place_labels(
        make_sequences(n=4, length=8), make_labels((1, 99.0, False))
    ).collect()

    assert result["position"].to_list() == [3]


def test_a_landmark_before_every_event_is_dropped(
    make_sequences: Callable, make_labels: Callable
) -> None:
    """There is no position to read and nothing to predict from.

    Kept with a null position it would reach the collate as an index into nothing.
    """
    result = place_labels(
        make_sequences(n=4, length=8), make_labels((1, -1.0, True))
    ).collect()

    assert result.height == 0


def test_an_overlapping_event_takes_the_chunk_with_more_history(
    make_sequences: Callable, make_labels: Callable
) -> None:
    """Event 2 sits at position 2 of chunk 0 and position 0 of chunk 1.

    Position 0 of chunk 1 has nothing in front of it. Taking the larger position is
    what keeps a landmark's attention window full, and both choices produce a valid
    sequence and position, so only this assertion separates them.
    """
    sequences = make_sequences(n=6, length=4, stride=2)

    result = place_labels(sequences, make_labels((1, 2.0, True))).collect()

    assert result["sequence_id"].to_list() == [0]
    assert result["position"].to_list() == [2]


def test_a_landmark_late_in_the_timeline_reaches_the_last_chunk(
    make_sequences: Callable, make_labels: Callable
) -> None:
    """The rule must still pick a chunk that actually contains the event."""
    sequences = make_sequences(n=7, length=4, stride=2)

    result = place_labels(sequences, make_labels((1, 6.0, True))).collect()

    assert result["sequence_id"].to_list() == [2]
    assert result["position"].to_list() == [2]


def test_landmarks_pinning_to_one_position_are_both_kept(
    make_sequences: Callable, make_labels: Callable
) -> None:
    """A 12-hour grid outruns a sparse record, and the two labels may disagree.

    They share a history but not a horizon, so collapsing them would discard a real
    observation rather than a duplicate.
    """
    labels = make_labels((1, 2.1, True), (1, 2.6, False))

    result = place_labels(make_sequences(n=4, length=8), labels).collect()

    assert result["position"].to_list() == [2, 2]
    assert result["boolean_value"].to_list() == [True, False]


def test_a_landmark_never_reaches_another_subject(
    make_sequences: Callable, make_labels: Callable
) -> None:
    """The asof join is grouped by subject; without that it would find any timeline.

    Subject 2's own events are at the same timestamps, so an ungrouped join returns a
    position that exists and is wrong.
    """
    sequences = make_sequences(n=4, subjects=(1, 2), length=8)

    result = place_labels(sequences, make_labels((2, 2.5, True))).collect()

    assert result["subject_id"].to_list() == [2]
    assert result["sequence_id"].to_list() == [1]


def test_carries_the_label_and_its_time(
    make_sequences: Callable, make_labels: Callable
) -> None:
    """The outcome is the point, and the time is what makes a placement auditable."""
    result = place_labels(
        make_sequences(n=4, length=8), make_labels((1, 2.5, True))
    ).collect()

    assert result["boolean_value"].to_list() == [True]
    assert result["prediction_time"].to_list() == [BIRTH + 2.5 * DAY]


def test_output_is_sorted_and_typed(
    make_sequences: Callable, make_labels: Callable
) -> None:
    """The collate gathers by position, and femr requires them ascending."""
    labels = make_labels((1, 5.0, True), (1, 1.0, False), (1, 3.0, True))

    result = place_labels(make_sequences(n=7, length=4, stride=2), labels).collect()

    assert dict(result.schema) == LABEL_SCHEMA
    assert result.sort("sequence_id", "position", "prediction_time").equals(result)


def test_no_landmarks_yields_a_typed_empty_frame(
    make_sequences: Callable, make_labels: Callable
) -> None:
    """A shard of sequences may hold no labelled subject once it has been split."""
    result = place_labels(make_sequences(n=4), make_labels()).collect()

    assert result.height == 0
    assert dict(result.schema) == LABEL_SCHEMA
