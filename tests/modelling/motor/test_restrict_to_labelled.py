"""Testing suite for cutting the corpus down to what a prediction could read.

Both cuts are free in the sense that nothing is lost: a subject with no landmark is
never scored, and an event after a subject's last landmark cannot reach any prediction
because attention is causal. Getting either wrong is not free. Too generous and two
thirds of the corpus is encoded to be thrown away; too aggressive and a landmark loses
the history it was supposed to be read from.
"""

from collections.abc import Callable
from datetime import datetime, timedelta

import polars as pl
import pytest

from thesis.modelling.motor.sequences import restrict_to_labelled

START = datetime(2150, 1, 1)
HOUR = timedelta(hours=1)


@pytest.fixture
def make_events() -> Callable:
    """Returns a factory for events, from (subject, hours-from-start) pairs."""

    def _make(*rows: tuple[int, float]) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "subject_id": [subject for subject, _ in rows],
                "time": [START + hours * HOUR for _, hours in rows],
                "code": [f"SNOMED/{index}" for index in range(len(rows))],
            },
            schema_overrides={"subject_id": pl.Int64, "time": pl.Datetime("us")},
        ).lazy()

    return _make


@pytest.fixture
def make_labels() -> Callable:
    """Returns a factory for landmarks, from (subject, hours-from-start) pairs."""

    def _make(*rows: tuple[int, float]) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "subject_id": [subject for subject, _ in rows],
                "prediction_time": [START + hours * HOUR for _, hours in rows],
            },
            schema_overrides={
                "subject_id": pl.Int64,
                "prediction_time": pl.Datetime("us"),
            },
        ).lazy()

    return _make


def test_drops_subjects_carrying_no_landmark(
    make_events: Callable, make_labels: Callable
) -> None:
    """141,174 of our subjects were never admitted and so are never scored."""
    events = make_events((1, 0.0), (2, 0.0))

    result = restrict_to_labelled(events, make_labels((1, 5.0))).collect()

    assert result["subject_id"].unique().to_list() == [1]


def test_drops_events_after_the_last_landmark(
    make_events: Callable, make_labels: Callable
) -> None:
    """No position can influence a prediction made before it; attention is causal.

    Keeping them would encode 28% more of the corpus for nothing.
    """
    events = make_events((1, 1.0), (1, 5.0), (1, 9.0))

    result = restrict_to_labelled(events, make_labels((1, 5.0))).collect()

    assert result.height == 2


def test_keeps_an_event_exactly_on_the_last_landmark(
    make_events: Callable, make_labels: Callable
) -> None:
    """The landmark reads the event at its own timestamp, so it must survive.

    Cutting strictly before would remove the admission event from the admission's own
    first prediction.
    """
    events = make_events((1, 5.0))

    result = restrict_to_labelled(events, make_labels((1, 5.0))).collect()

    assert result.height == 1


def test_the_cut_is_the_subject_own_last_landmark(
    make_events: Callable, make_labels: Callable
) -> None:
    """A global cut would keep one subject's tail and truncate another's history."""
    events = make_events((1, 1.0), (1, 9.0), (2, 1.0), (2, 9.0))
    labels = make_labels((1, 2.0), (2, 20.0))

    result = restrict_to_labelled(events, labels).collect()

    assert result.filter(pl.col("subject_id") == 1).height == 1
    assert result.filter(pl.col("subject_id") == 2).height == 2


def test_the_latest_landmark_sets_the_cut(
    make_events: Callable, make_labels: Callable
) -> None:
    """A subject holds many landmarks and the events serve all of them.

    Cutting at the first would leave every later landmark reading a truncated record.
    """
    events = make_events((1, 1.0), (1, 5.0), (1, 9.0))
    labels = make_labels((1, 2.0), (1, 6.0), (1, 10.0))

    result = restrict_to_labelled(events, labels).collect()

    assert result.height == 3


def test_leaves_no_join_columns_behind(
    make_events: Callable, make_labels: Callable
) -> None:
    """The result is tokenised next, and an extra column would ride into the shards."""
    events = make_events((1, 1.0))

    result = restrict_to_labelled(events, make_labels((1, 5.0))).collect()

    assert result.columns == ["subject_id", "time", "code"]


def test_no_labels_yields_no_events(
    make_events: Callable, make_labels: Callable
) -> None:
    """A shard may hold no labelled subject at all; it must not raise."""
    result = restrict_to_labelled(make_events((1, 1.0)), make_labels()).collect()

    assert result.height == 0
