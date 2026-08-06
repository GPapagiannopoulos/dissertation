"""Testing suite for assembling tokenised events into sequences.

Two things decided here are unrecoverable downstream. A position that shifts hands an
event another event's neighbours, and every attention weight after it is computed over
the wrong history. An age that is wrong by a factor rotates the whole sequence on the
wrong clock, which the encoder cannot detect because any age produces a finite table.

The chunking cases walk the boundary one event at a time rather than sampling it,
because an off-by-one only shows up at the exact length where a chunk is added.
"""

from collections.abc import Callable
from datetime import datetime, timedelta

import polars as pl
import pytest

from thesis.modelling.motor.sequences import SEQUENCE_SCHEMA, build_sequences

BIRTH = datetime(2100, 1, 1)
AGE_MEAN, AGE_STD = 100.0, 50.0


@pytest.fixture
def make_events() -> Callable:
    """Returns a factory for tokenised events, one per day from a subject's birth."""

    def _make(
        subject_id: int = 1,
        *,
        days: list[float] | None = None,
        n: int = 1,
        indices: list[int] | None = None,
    ) -> pl.LazyFrame:
        offsets = days if days is not None else list(range(n))
        return pl.DataFrame(
            {
                "subject_id": [subject_id] * len(offsets),
                "time": [BIRTH + timedelta(days=offset) for offset in offsets],
                "index": indices or list(range(10, 10 + len(offsets))),
            },
            schema_overrides={"subject_id": pl.Int64, "index": pl.UInt32},
        ).lazy()

    return _make


@pytest.fixture
def make_births() -> Callable:
    """Returns a factory for the birth table the ages are measured from."""

    def _make(*subject_ids: int) -> pl.LazyFrame:
        ids = list(subject_ids) or [1]
        return pl.DataFrame(
            {"subject_id": ids, "birth": [BIRTH] * len(ids)},
            schema_overrides={"subject_id": pl.Int64},
        ).lazy()

    return _make


def _build(events: pl.LazyFrame, births: pl.LazyFrame, **overrides) -> pl.DataFrame:
    """Runs the builder at toy widths, overriding only what a case cares about."""
    arguments = {
        "length": 4,
        "stride": 2,
        "age_mean": AGE_MEAN,
        "age_std": AGE_STD,
    } | overrides
    return build_sequences(events, births, **arguments).collect()


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        # 0. One event is still a sequence
        (1, [[0]]),
        # 1. Short of the length, so no cut at all
        (3, [[0, 1, 2]]),
        # 2. Exactly the length: the boundary where a second chunk must NOT appear
        (4, [[0, 1, 2, 3]]),
        # 3. One past it: the second chunk opens at the stride, not at the length
        (5, [[0, 1, 2, 3], [2, 3, 4]]),
        # 4. The second chunk fills
        (6, [[0, 1, 2, 3], [2, 3, 4, 5]]),
        # 5. A third opens
        (7, [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6]]),
        # 6. Two full strides past the length
        (8, [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]]),
    ],
)
def test_chunks_cover_the_timeline_with_the_right_overlap(
    make_events: Callable, make_births: Callable, n: int, expected: list[list[int]]
) -> None:
    """Every event reaches a chunk, and each chunk starts one stride after the last.

    A gap here loses events silently; too much overlap multiplies the corpus.
    """
    result = _build(make_events(n=n), make_births())

    actual = [
        result.filter(pl.col("sequence_id") == sequence)["subject_position"].to_list()
        for sequence in result["sequence_id"].unique().sort()
    ]
    assert actual == expected


@pytest.mark.parametrize("length", [8, 16, 64])
def test_a_short_subject_is_one_sequence_at_any_length(
    make_events: Callable, make_births: Callable, length: int
) -> None:
    """A timeline well inside the length must never be cut.

    Regression: the chunk count subtracted the length from a UInt32 `pl.len()`, so a
    subject shorter than it wrapped to 4.29e9 chunks rather than clipping to one. The
    surplus chunks were then capped by the position, which hid the wrap at short
    lengths and produced spurious half-length sequences at long ones.
    """
    result = _build(make_events(n=4), make_births(), length=length, stride=length // 2)

    assert result["sequence_id"].n_unique() == 1
    assert result["position"].to_list() == [0, 1, 2, 3]


def test_a_chunk_restarts_its_positions_from_zero(
    make_events: Callable, make_births: Callable
) -> None:
    """The encoder indexes a sequence, not a timeline.

    A local position that carried on counting would run past the buffer and index
    another sequence's rows.
    """
    result = _build(make_events(n=7), make_births())

    second = result.filter(pl.col("sequence_id") == 1)
    assert second["position"].to_list() == [0, 1, 2, 3]
    assert second["subject_position"].to_list() == [2, 3, 4, 5]


def test_an_overlapping_event_appears_in_both_chunks(
    make_events: Callable, make_births: Callable
) -> None:
    """The overlap is the point: it is what gives the later chunk its history.

    The same event is a different position in each, which is why labels have to
    choose a chunk rather than inheriting one.
    """
    result = _build(make_events(n=6), make_births())

    shared = result.filter(pl.col("subject_position") == 2)
    assert shared["sequence_id"].to_list() == [0, 1]
    assert shared["position"].to_list() == [2, 0]


def test_a_stride_of_the_full_length_leaves_no_overlap(
    make_events: Callable, make_births: Callable
) -> None:
    """Overlap is a choice, and turning it off must not drop or duplicate an event."""
    result = _build(make_events(n=7), make_births(), stride=4)

    assert result.height == 7
    assert result["subject_position"].to_list() == [0, 1, 2, 3, 4, 5, 6]
    assert result["position"].to_list() == [0, 1, 2, 3, 0, 1, 2]


def test_age_is_days_since_birth(make_events: Callable, make_births: Callable) -> None:
    """Days, not minutes. The rotary schedule is calibrated to the unit.

    In minutes the slowest clock wraps every 43.6 days and the model cannot tell a
    40-year-old from a 41-year-old.
    """
    result = _build(make_events(days=[0.0, 1.0, 365.0]), make_births())

    assert result["age"].to_list() == [0.0, 1.0, 365.0]


def test_age_truncates_to_whole_minutes(
    make_events: Callable, make_births: Callable
) -> None:
    """The count is of whole minutes, divided by 1440; the truncation is real.

    Reproducing the division without the truncation would drift from the checkpoint's
    own arithmetic at every sub-minute event.
    """
    result = _build(
        make_events(days=[30 / 86_400, 90 / 86_400, 1439 / 1440]), make_births()
    )

    assert result["age"].to_list() == pytest.approx([0.0, 1 / 1440, 1439 / 1440])


def test_normed_age_uses_the_dictionary_statistics(
    make_events: Callable, make_births: Callable
) -> None:
    """The checkpoint ships its own age_stats, and the batch's would differ.

    Both produce plausible small numbers, so nothing downstream can tell them apart.
    """
    result = _build(make_events(days=[100.0, 150.0]), make_births())

    assert result["normed_age"].to_list() == pytest.approx([0.0, 1.0])


def test_positions_follow_time_not_arrival(
    make_events: Callable, make_births: Callable
) -> None:
    """A shard is sorted, but nothing in the type system says so."""
    result = _build(make_events(days=[5.0, 1.0, 3.0]), make_births())

    assert result["age"].to_list() == [1.0, 3.0, 5.0]
    assert result["subject_position"].to_list() == [0, 1, 2]


def test_events_sharing_a_timestamp_keep_their_arrival_order(
    make_events: Callable, make_births: Callable
) -> None:
    """A Polars sort is not stable, so a tie must be broken explicitly.

    Reproducing it needs the production shape, which is why this case is bigger than
    the rest: several subjects, INTERLEAVED, sharing a timestamp. One subject's tied
    rows survive a sort even at 200,000, and so do contiguous blocks of subjects --
    but interleaved subjects reorder from 24 rows up. The births join runs before the
    sort and returns hash-probe order, so interleaved is what the sort really sees.
    """
    subjects, per_subject = 3, 8
    total = subjects * per_subject
    events = pl.DataFrame(
        {
            "subject_id": [index % subjects for index in range(total)],
            "time": [BIRTH + timedelta(days=2)] * total,
            "index": list(range(total)),
        },
        schema_overrides={"subject_id": pl.Int64, "index": pl.UInt32},
    ).lazy()

    result = _build(events, make_births(*range(subjects)), length=total, stride=total)

    for subject in range(subjects):
        arrived = [index for index in range(total) if index % subjects == subject]
        rows = result.filter(pl.col("subject_id") == subject)
        assert rows["index"].to_list() == arrived


def test_subjects_do_not_share_a_sequence(
    make_events: Callable, make_births: Callable
) -> None:
    """One subject reading another's history is the failure that invalidates a study."""
    events = pl.concat([make_events(1, n=3), make_events(2, n=2)])

    result = _build(events, make_births(1, 2))

    assert result["sequence_id"].to_list() == [0, 0, 0, 1, 1]
    assert result["subject_position"].to_list() == [0, 1, 2, 0, 1]


def test_sequence_ids_are_dense_and_zero_based(
    make_events: Callable, make_births: Callable
) -> None:
    """The collate uses them to group, and the encoder to place a batch row."""
    events = pl.concat([make_events(1, n=7), make_events(2, n=1)])

    result = _build(events, make_births(1, 2))

    assert result["sequence_id"].unique().sort().to_list() == [0, 1, 2, 3]


def test_a_subject_without_a_birth_is_dropped(
    make_events: Callable, make_births: Callable
) -> None:
    """An age is not optional: a null one reaches the encoder as a NaN rotary table.

    `MEDS_BIRTH` carries no OMOP concept and so never survives tokenisation, which is
    exactly why the origin arrives separately and can go missing.
    """
    events = pl.concat([make_events(1, n=2), make_events(2, n=2)])

    result = _build(events, make_births(1))

    assert result["subject_id"].unique().to_list() == [1]


def test_output_is_sorted_and_typed(
    make_events: Callable, make_births: Callable
) -> None:
    """This is written to parquet, so its order is its contract."""
    result = _build(make_events(n=7), make_births())

    assert dict(result.schema) == SEQUENCE_SCHEMA
    assert result.sort("sequence_id", "position").equals(result)


def test_an_empty_input_yields_a_typed_empty_frame(
    make_events: Callable, make_births: Callable
) -> None:
    """A shard may hold no labelled subject at all once it has been filtered."""
    result = _build(make_events(n=1).filter(pl.col("index") > 1000), make_births())

    assert result.height == 0
    assert dict(result.schema) == SEQUENCE_SCHEMA


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"length": 0}, "at least 1", id="zero_length"),
        pytest.param({"stride": 0}, "between 1", id="zero_stride"),
        pytest.param({"stride": 5}, "between 1", id="stride_past_length"),
        pytest.param({"age_std": 0.0}, "must be positive", id="zero_std"),
    ],
)
def test_rejects_malformed_arguments(
    make_events: Callable, make_births: Callable, overrides: dict, match: str
) -> None:
    """A stride past the length would leave gaps between chunks, losing events."""
    with pytest.raises(ValueError, match=match):
        _build(make_events(n=4), make_births(), **overrides)
