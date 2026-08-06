"""Testing suite for the leaf token assignment.

A leaf token is the event's own row in the embedding table, before the ancestor
expansion adds the rest. Three lookups feed it, tried in order -- the value bin,
the text value, then the bare code -- and every one of them can fail quietly:
a wrong bin, a duplicated row or a token borrowed from another code all produce a
well-formed frame full of the wrong concepts.
"""

from collections.abc import Callable

import polars as pl
import pytest

from thesis.modelling.motor.tokenizer import assign_leaf_tokens


@pytest.mark.parametrize(
    "value, expected",
    [
        # 0. Below every cut point, in the bin open at -DBL_MAX
        (0.2, 1),
        # 1. Inside the middle bin
        (1.5, 2),
        # 2. Above every cut point, in the bin open at +DBL_MAX
        (9.0, 3),
        # 3. Exactly on a cut point: bins are [start, end), so it opens the next one
        (1.0, 2),
        # 4. Exactly on the last cut point
        (2.0, 3),
    ],
)
def test_numeric_value_lands_in_its_bin(
    make_events: Callable,
    make_token_table: Callable,
    make_bins: Callable,
    value: float,
    expected: int,
) -> None:
    """Asserts the bin holding the value wins, including on its lower edge."""
    table = make_token_table(
        numeric_tokens=make_bins("LOINC/lab", [1.0, 2.0], first_index=1)
    )
    events = make_events(code=["LOINC/lab"], numeric_value=[value])

    result = assign_leaf_tokens(events, table).collect()

    assert result["index"].to_list() == [expected]


def test_bins_are_right_when_events_arrive_unsorted(
    make_events: Callable, make_token_table: Callable, make_bins: Callable
) -> None:
    """MEDS shards are sorted by subject and time, never by value.

    join_asof binary-searches the left frame and does not verify it is ordered when
    `by` groups are given -- it only warns. Handed shard order it returns a bin for
    every row, all of them plausible and most of them wrong, so this is the one
    failure in the module that no downstream check would catch.
    """
    values = [4.5, 0.1, 2.5, 1.5, 0.9, 3.5, 1.1, 2.1, 0.5, 4.1]
    cuts = [1.0, 2.0, 3.0, 4.0]
    table = make_token_table(numeric_tokens=make_bins("LOINC/lab", cuts, first_index=1))
    events = make_events(code=["LOINC/lab"], numeric_value=values)

    result = assign_leaf_tokens(events, table).collect()

    expected = [1 + sum(cut <= value for cut in cuts) for value in values]
    assert result["index"].to_list() == expected


def test_preserves_the_order_of_the_events(
    make_events: Callable, make_token_table: Callable, make_bins: Callable
) -> None:
    """Positions are assigned by row order downstream, and MEDS demands chronology.

    Sorting for the asof join is legitimate; leaving the frame sorted is not.
    """
    table = make_token_table(
        numeric_tokens=make_bins("LOINC/lab", [1.0], first_index=1)
    )
    events = make_events(
        code=["LOINC/lab"],
        numeric_value=[5.0, 0.5, 3.0, 0.1],
        text_value=[None, None, None, None],
    ).with_columns(position=pl.int_range(pl.len(), dtype=pl.UInt32))

    result = assign_leaf_tokens(events, table).collect()

    assert result["position"].to_list() == [0, 1, 2, 3]


def test_falls_back_to_the_text_token(
    make_events: Callable, make_token_table: Callable, make_texts: Callable
) -> None:
    """A code with no bins takes the token for the value it recorded."""
    table = make_token_table(
        text_tokens=make_texts(("SNOMED/culture", "N", 5), ("SNOMED/culture", "Y", 6))
    )
    events = make_events(code=["SNOMED/culture"], text_value=["Y"])

    result = assign_leaf_tokens(events, table).collect()

    assert result["index"].to_list() == [6]


def test_text_value_must_match_the_token_exactly(
    make_events: Callable, make_token_table: Callable, make_texts: Callable
) -> None:
    """An unrecognised value is not a token; joining only on code would say it is."""
    table = make_token_table(text_tokens=make_texts(("SNOMED/culture", "N", 5)))
    events = make_events(code=["SNOMED/culture"], text_value=["POSITIVE"])

    result = assign_leaf_tokens(events, table).collect()

    assert result.height == 0


def test_text_tokens_do_not_duplicate_the_event(
    make_events: Callable, make_token_table: Callable, make_texts: Callable
) -> None:
    """A code owning several text tokens must still emit one row.

    Duplication here is silent: every copy is a real event with a real token, and it
    inflates that concept's frequency in the sequence.
    """
    table = make_token_table(
        text_tokens=make_texts(
            ("SNOMED/culture", "N", 5),
            ("SNOMED/culture", "Y", 6),
            ("SNOMED/culture", "I", 7),
        )
    )
    events = make_events(code=["SNOMED/culture"], text_value=["N"])

    result = assign_leaf_tokens(events, table).collect()

    assert result.height == 1


def test_falls_back_to_the_plain_code_when_the_value_is_missing(
    make_events: Callable, make_token_table: Callable, make_bins: Callable
) -> None:
    """Roughly 8% of events on a binned code carry no value. Code still tokenises."""
    table = make_token_table(
        code_tokens={"LOINC/lab": 9},
        numeric_tokens=make_bins("LOINC/lab", [1.0], first_index=1),
    )
    events = make_events(code=["LOINC/lab"], numeric_value=[None])

    result = assign_leaf_tokens(events, table).collect()

    assert result["index"].to_list() == [9]


def test_a_null_value_does_not_reach_the_lowest_bin(
    make_events: Callable, make_token_table: Callable, make_bins: Callable
) -> None:
    """The first bin opens at -DBL_MAX, so a missing value must not sort into it."""
    table = make_token_table(
        numeric_tokens=make_bins("LOINC/lab", [1.0], first_index=1)
    )
    events = make_events(code=["LOINC/lab"], numeric_value=[None])

    result = assign_leaf_tokens(events, table).collect()

    assert result.height == 0


def test_the_bin_outranks_the_text_and_the_code(
    make_events: Callable,
    make_token_table: Callable,
    make_bins: Callable,
    make_texts: Callable,
) -> None:
    """A code may hold all three kinds; the value is the most specific of them."""
    table = make_token_table(
        code_tokens={"LOINC/lab": 9},
        numeric_tokens=make_bins("LOINC/lab", [1.0], first_index=1),
        text_tokens=make_texts(("LOINC/lab", "N", 5)),
    )
    events = make_events(code=["LOINC/lab"], numeric_value=[0.5], text_value=["N"])

    result = assign_leaf_tokens(events, table).collect()

    assert result["index"].to_list() == [1]


def test_the_text_token_outranks_the_code(
    make_events: Callable, make_token_table: Callable, make_texts: Callable
) -> None:
    """Between the two remaining kinds the recorded value is still the specific one."""
    table = make_token_table(
        code_tokens={"SNOMED/culture": 9},
        text_tokens=make_texts(("SNOMED/culture", "N", 5)),
    )
    events = make_events(code=["SNOMED/culture"], text_value=["N"])

    result = assign_leaf_tokens(events, table).collect()

    assert result["index"].to_list() == [5]


def test_drops_events_the_vocabulary_cannot_name(
    make_events: Callable, make_token_table: Callable
) -> None:
    """Two thirds of MIMIC's events reach no MOTOR token and cannot be embedded."""
    events = make_events(
        code=["SNOMED/plain", "MIMIC_IV_Absent"], numeric_value=[None, None]
    )

    result = assign_leaf_tokens(events, make_token_table()).collect()

    assert result["code"].to_list() == ["SNOMED/plain"]


def test_the_index_is_an_unsigned_32_bit_row(
    make_events: Callable, make_token_table: Callable
) -> None:
    """It indexes a 65,536-row table and is emitted straight into the sparse pairs."""
    result = assign_leaf_tokens(make_events(), make_token_table()).collect()

    assert result.schema["index"] == pl.UInt32


def test_leaves_no_lookup_columns_behind(
    make_events: Callable, make_token_table: Callable, make_bins: Callable
) -> None:
    """The joins bring their own columns; the caller writes this frame to parquet."""
    table = make_token_table(
        numeric_tokens=make_bins("LOINC/lab", [1.0], first_index=1)
    )
    events = make_events(code=["LOINC/lab"], numeric_value=[0.5])

    result = assign_leaf_tokens(events, table).collect()

    assert set(result.columns) == {
        "subject_id",
        "code",
        "numeric_value",
        "text_value",
        "index",
    }
