"""Testing suite for the MOTOR token table loader.

Unlike the rest of this package there is no oracle here: batching lives in femr's
compiled extension, so these assert the loader's own contract instead. The one that
matters most is that a token's index is its position in the rollup -- that position
is the row it owns in the embedding table, so an off-by-one silently embeds the
wrong concept rather than raising.
"""

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.modelling.motor.tokenizer import load_token_table

CODE, NUMERIC, TEXT, UNUSED = 0, 1, 2, 3

# the open ends of a code's bins, C's DBL_MAX, as the dictionary stores them
LOWEST, HIGHEST = -1.7976931348623157e308, 1.7976931348623157e308

NUMERIC_SCHEMA = {
    "code": pl.String,
    "val_start": pl.Float64,
    "val_end": pl.Float64,
    "numeric_indices": pl.UInt32,
}
TEXT_SCHEMA = {
    "code": pl.String,
    "text_string": pl.String,
    "text_indices": pl.UInt32,
}


def test_partitions_entries_by_type(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """Asserts each entry type populates only its own lookup."""
    rollup = [
        rollup_entry("SNOMED/1", CODE),
        rollup_entry("LOINC/2", NUMERIC, val_start=0.0, val_end=1.0),
        rollup_entry("SNOMED/3", TEXT, text_string="N"),
    ]
    path = make_dictionary(ontology_rollup=rollup)

    table = load_token_table(path, vocab_size=len(rollup))

    assert table.code_tokens == {"SNOMED/1": 0}
    assert_frame_equal(
        table.numeric_tokens,
        pl.DataFrame(
            {
                "code": ["LOINC/2"],
                "val_start": [0.0],
                "val_end": [1.0],
                "numeric_indices": [1],
            },
            schema=NUMERIC_SCHEMA,
        ),
    )
    assert_frame_equal(
        table.text_tokens,
        pl.DataFrame(
            {"code": ["SNOMED/3"], "text_string": ["N"], "text_indices": [2]},
            schema=TEXT_SCHEMA,
        ),
    )


@pytest.mark.parametrize(
    "unused_type",
    [
        # 0. Type 3 fills a third of the vocabulary and bears no token
        UNUSED,
        # 1. A type this loader has never seen must be skipped, not guessed at
        99,
    ],
)
def test_skips_entries_that_bear_no_token(
    make_dictionary: Callable, rollup_entry: Callable, unused_type: int
) -> None:
    """An entry of an unused type reaches no lookup at all."""
    rollup = [
        rollup_entry("SNOMED/1", CODE),
        rollup_entry("SNOMED/2", unused_type),
    ]
    path = make_dictionary(ontology_rollup=rollup)

    table = load_token_table(path, vocab_size=len(rollup))

    assert table.code_tokens == {"SNOMED/1": 0}
    assert table.text_tokens.height == 0
    assert table.numeric_tokens.height == 0


def test_indices_count_skipped_entries(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """A skipped entry still owns its embedding row, so it still shifts the ones after.

    This is the failure a running counter over kept entries would produce, and it
    cannot be caught downstream: every index stays inside the table and names a real
    but wrong concept.
    """
    rollup = [
        rollup_entry("SNOMED/1", CODE),
        rollup_entry("SNOMED/unused", UNUSED),
        rollup_entry("SNOMED/3", CODE),
        rollup_entry("SNOMED/4", TEXT, text_string="Y"),
    ]
    path = make_dictionary(ontology_rollup=rollup)

    table = load_token_table(path, vocab_size=len(rollup))

    assert table.code_tokens == {"SNOMED/1": 0, "SNOMED/3": 2}
    assert table.text_tokens["text_indices"].to_list() == [3]


def test_keeps_one_numeric_row_per_bin(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """A code owns roughly ten bins, each a token of its own, in rollup order."""
    rollup = [
        rollup_entry("LOINC/2160-0", NUMERIC, val_start=LOWEST, val_end=0.6),
        rollup_entry("LOINC/2160-0", NUMERIC, val_start=0.6, val_end=1.03),
        rollup_entry("LOINC/2160-0", NUMERIC, val_start=1.03, val_end=HIGHEST),
    ]
    path = make_dictionary(ontology_rollup=rollup)

    table = load_token_table(path, vocab_size=len(rollup))

    assert_frame_equal(
        table.numeric_tokens,
        pl.DataFrame(
            {
                "code": ["LOINC/2160-0"] * 3,
                "val_start": [LOWEST, 0.6, 1.03],
                "val_end": [0.6, 1.03, HIGHEST],
                "numeric_indices": [0, 1, 2],
            },
            schema=NUMERIC_SCHEMA,
        ),
    )


def test_frames_are_typed_even_when_empty(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """Leaf assignment joins on these, and a Null column would match nothing."""
    path = make_dictionary(ontology_rollup=[rollup_entry("SNOMED/1", CODE)])

    table = load_token_table(path, vocab_size=1)

    assert table.numeric_tokens.height == 0
    assert table.text_tokens.height == 0
    assert dict(table.numeric_tokens.schema) == NUMERIC_SCHEMA
    assert dict(table.text_tokens.schema) == TEXT_SCHEMA


def test_text_tokens_are_keyed_by_code_and_value(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """One code carries several text tokens, so the value is half the key."""
    rollup = [
        rollup_entry("SNOMED/228490006", TEXT, text_string="N"),
        rollup_entry("SNOMED/228490006", TEXT, text_string="Y"),
    ]
    path = make_dictionary(ontology_rollup=rollup)

    table = load_token_table(path, vocab_size=len(rollup))

    assert_frame_equal(
        table.text_tokens,
        pl.DataFrame(
            {
                "code": ["SNOMED/228490006"] * 2,
                "text_string": ["N", "Y"],
                "text_indices": [0, 1],
            },
            schema=TEXT_SCHEMA,
        ),
    )


def test_a_code_may_hold_tokens_of_several_types(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """A lab carries both an identity token and value bins; both must survive.

    The resolution order falls back from the bin to the identity token, so a loader
    that let one type overwrite the other would silently drop that fallback.
    """
    rollup = [
        rollup_entry("LOINC/2160-0", CODE),
        rollup_entry("LOINC/2160-0", NUMERIC, val_start=0.0, val_end=1.0),
    ]
    path = make_dictionary(ontology_rollup=rollup)

    table = load_token_table(path, vocab_size=len(rollup))

    assert table.code_tokens == {"LOINC/2160-0": 0}
    assert table.numeric_tokens["code"].to_list() == ["LOINC/2160-0"]


def test_truncates_at_vocab_size(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """Entries past the cut own no embedding row, however real they look."""
    rollup = [
        rollup_entry("SNOMED/kept", CODE),
        rollup_entry("LOINC/dropped", NUMERIC),
        rollup_entry("SNOMED/dropped", TEXT, text_string="N"),
    ]
    path = make_dictionary(ontology_rollup=rollup)

    table = load_token_table(path, vocab_size=1)

    assert table.code_tokens == {"SNOMED/kept": 0}
    assert table.text_tokens.height == 0
    assert table.numeric_tokens.height == 0


def test_reads_the_age_statistics(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """Ages are z-scored against the checkpoint's own stats, not the batch's."""
    path = make_dictionary(
        ontology_rollup=[rollup_entry("SNOMED/1", CODE)],
        age_stats={"mean": 13540.859046944894, "std": 9335.632701762013},
    )

    table = load_token_table(path, vocab_size=1)

    assert table.age_mean == 13540.859046944894
    assert table.age_std == 9335.632701762013


def test_returns_the_whole_ancestor_map(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """Expansion walks ancestors outside the vocabulary, so the map is not filtered."""
    all_parents = {
        "SNOMED/1": ("SNOMED/1", "SNOMED/root"),
        "SNOMED/absent": ("SNOMED/absent", "SNOMED/root"),
    }
    path = make_dictionary(
        ontology_rollup=[rollup_entry("SNOMED/1", CODE)], all_parents=all_parents
    )

    table = load_token_table(path, vocab_size=1)

    assert table.all_parents == all_parents


@pytest.mark.parametrize(
    "vocab_size",
    [
        # 0. Zero would silently yield empty lookups
        0,
        # 1. Negative sizes are meaningless
        -3,
    ],
)
def test_rejects_non_positive_vocab_size(
    make_dictionary: Callable, rollup_entry: Callable, vocab_size: int
) -> None:
    """Asserts the vocabulary needs at least one token."""
    path = make_dictionary(ontology_rollup=[rollup_entry("SNOMED/1", CODE)])

    with pytest.raises(
        ValueError, match=f"The minimum vocabulary size is 1. Received {vocab_size}"
    ):
        load_token_table(path, vocab_size=vocab_size)


def test_rejects_vocab_size_beyond_the_dictionary(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """A size the dictionary cannot honour means the rows would not line up."""
    path = make_dictionary(ontology_rollup=[rollup_entry("SNOMED/1", CODE)])

    with pytest.raises(ValueError, match="exceeds the 1 entries"):
        load_token_table(path, vocab_size=2)
