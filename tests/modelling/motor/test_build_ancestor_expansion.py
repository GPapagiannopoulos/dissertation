"""Testing suite for the precomputed ancestor expansion.

This table decides how many embedding rows each token is worth, and every way it
can be wrong is quiet. Too many rows and a concept is counted twice in the sum;
too few and the generalisation MOTOR was pretrained to rely on silently goes
missing. Neither changes a shape, raises, or moves a number outside its range.

The asymmetry the cases keep returning to was read out of femr's
`FeatureLookup::get_feature_codes`: the value-less branch copies a variable-length
vector of ancestors, while the numeric and text branches allocate four bytes and
return a single token.
"""

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.modelling.motor.tokenizer import build_ancestor_expansion

EXPANSION_SCHEMA = {"index": pl.UInt32, "token": pl.UInt32}
NUMERIC_SCHEMA = {
    "code": pl.String,
    "val_start": pl.Float64,
    "val_end": pl.Float64,
    "numeric_indices": pl.UInt32,
}


def _no_bins() -> pl.DataFrame:
    """A vocabulary whose codes carry no value bins at all."""
    return pl.DataFrame(schema=NUMERIC_SCHEMA)


def _expansion_of(frame: pl.DataFrame, index: int) -> list[int]:
    """Returns the tokens one leaf contributes."""
    return frame.filter(pl.col("index") == index)["token"].to_list()


def test_a_code_contributes_itself_and_its_ancestors(
    make_token_table: Callable, make_texts: Callable
) -> None:
    """The whole point: a rare code is embedded near the parents it shares."""
    table = make_token_table(
        code_tokens={"SNOMED/leaf": 0, "SNOMED/middle": 1, "SNOMED/root": 2},
        numeric_tokens=_no_bins(),
        text_tokens=make_texts(),
        all_parents={"SNOMED/leaf": ("SNOMED/leaf", "SNOMED/middle", "SNOMED/root")},
    )

    result = build_ancestor_expansion(table).collect()

    assert _expansion_of(result, 0) == [0, 1, 2]


def test_ancestors_outside_the_vocabulary_are_dropped(
    make_token_table: Callable, make_texts: Callable
) -> None:
    """The ontology is far larger than the 65,536 tokens that own a row.

    An ancestor the dictionary did not keep has no embedding to contribute, and
    passing its code on would index the table with a number that means something
    else entirely.
    """
    table = make_token_table(
        code_tokens={"SNOMED/leaf": 0, "SNOMED/kept": 1},
        numeric_tokens=_no_bins(),
        text_tokens=make_texts(),
        all_parents={"SNOMED/leaf": ("SNOMED/leaf", "SNOMED/absent", "SNOMED/kept")},
    )

    result = build_ancestor_expansion(table).collect()

    assert _expansion_of(result, 0) == [0, 1]


def test_a_value_bin_contributes_only_itself(
    make_token_table: Callable, make_bins: Callable, make_texts: Callable
) -> None:
    """The numeric branch in femr allocates four bytes and returns one token.

    A bin already names one code at one value range, so there is nothing to
    generalise toward. Giving it the code's ancestors would embed valued events
    in a context pretraining never produced.
    """
    table = make_token_table(
        code_tokens={"LOINC/lab": 0, "LOINC/parent": 1},
        numeric_tokens=make_bins("LOINC/lab", [1.0], first_index=2),
        text_tokens=make_texts(),
        all_parents={"LOINC/lab": ("LOINC/lab", "LOINC/parent")},
    )

    result = build_ancestor_expansion(table).collect()

    assert _expansion_of(result, 2) == [2]
    assert _expansion_of(result, 3) == [3]


def test_a_text_token_contributes_only_itself(
    make_token_table: Callable, make_texts: Callable
) -> None:
    """The text branch is keyed on (code, value) and returns one token too."""
    table = make_token_table(
        code_tokens={"SNOMED/culture": 0, "SNOMED/parent": 1},
        numeric_tokens=_no_bins(),
        text_tokens=make_texts(("SNOMED/culture", "N", 2)),
        all_parents={"SNOMED/culture": ("SNOMED/culture", "SNOMED/parent")},
    )

    result = build_ancestor_expansion(table).collect()

    assert _expansion_of(result, 2) == [2]


@pytest.mark.parametrize(
    "parents",
    [
        # 0. The dictionary lists a code among its own parents
        ("SNOMED/leaf", "SNOMED/leaf", "SNOMED/root"),
        # 1. An ancestor reachable by two paths through the ontology
        ("SNOMED/leaf", "SNOMED/root", "SNOMED/root"),
    ],
)
def test_a_repeated_ancestor_contributes_one_row(
    make_token_table: Callable,
    make_texts: Callable,
    parents: tuple[str, ...],
) -> None:
    """A duplicate row adds that concept's vector twice and skews the position.

    Nothing downstream can see it: the sum stays finite, the shape is unchanged,
    and the only symptom is a concept weighted double.
    """
    table = make_token_table(
        code_tokens={"SNOMED/leaf": 0, "SNOMED/root": 1},
        numeric_tokens=_no_bins(),
        text_tokens=make_texts(),
        all_parents={"SNOMED/leaf": parents},
    )

    result = build_ancestor_expansion(table).collect()

    assert _expansion_of(result, 0) == [0, 1]


def test_a_code_missing_from_the_ontology_still_contributes_itself(
    make_token_table: Callable, make_texts: Callable
) -> None:
    """Two of our 8,923 codes are absent from all_parents; they still embed."""
    table = make_token_table(
        code_tokens={"SNOMED/orphan": 0},
        numeric_tokens=_no_bins(),
        text_tokens=make_texts(),
        all_parents={},
    )

    result = build_ancestor_expansion(table).collect()

    assert _expansion_of(result, 0) == [0]


def test_every_token_in_the_vocabulary_is_named(
    make_token_table: Callable, make_bins: Callable, make_texts: Callable
) -> None:
    """A leaf absent from this table embeds as zero rather than raising."""
    table = make_token_table(
        code_tokens={"SNOMED/plain": 0},
        numeric_tokens=make_bins("LOINC/lab", [1.0], first_index=1),
        text_tokens=make_texts(("SNOMED/text", "N", 3)),
        all_parents={},
    )

    result = build_ancestor_expansion(table).collect()

    assert sorted(result["index"].unique().to_list()) == [0, 1, 2, 3]


def test_is_sorted_and_typed_for_the_join(
    make_token_table: Callable, make_texts: Callable
) -> None:
    """Dict iteration order is not an artifact contract, and the join needs UInt32."""
    table = make_token_table(
        code_tokens={"SNOMED/b": 2, "SNOMED/a": 0, "SNOMED/root": 1},
        numeric_tokens=_no_bins(),
        text_tokens=make_texts(),
        all_parents={
            "SNOMED/a": ("SNOMED/a", "SNOMED/root"),
            "SNOMED/b": ("SNOMED/b", "SNOMED/root"),
        },
    )

    result = build_ancestor_expansion(table).collect()

    assert_frame_equal(
        result,
        pl.DataFrame(
            {"index": [0, 0, 1, 2, 2], "token": [0, 1, 1, 1, 2]},
            schema=EXPANSION_SCHEMA,
        ),
    )


def test_an_empty_vocabulary_yields_a_typed_empty_table(
    make_token_table: Callable, make_texts: Callable
) -> None:
    """The join downstream would match nothing against a Null column."""
    table = make_token_table(
        code_tokens={},
        numeric_tokens=_no_bins(),
        text_tokens=make_texts(),
        all_parents={},
    )

    result = build_ancestor_expansion(table).collect()

    assert result.height == 0
    assert dict(result.schema) == EXPANSION_SCHEMA
