"""Testing suite for the direct vocabulary resolver."""

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.modelling.etl_pipeline.coverage import resolve_direct


def _expected(codes: list[str]) -> pl.LazyFrame:
    """Builds the frame a resolver is expected to emit for codes it resolved."""
    return pl.LazyFrame(
        {
            "code": pl.Series(codes, dtype=pl.String),
            "target": pl.Series(codes, dtype=pl.String),
            "method": pl.Series(["direct"] * len(codes), dtype=pl.String),
        }
    )


@pytest.mark.parametrize(
    "codes, resolved",
    [
        # 0. A code carrying an identity token resolves to itself
        (["SNOMED/1"], ["SNOMED/1"]),
        # 1. A code known only by its value bins still resolves
        (["LOINC/2"], ["LOINC/2"]),
        # 2. A code known only by its categorical values still resolves
        (["SNOMED/3"], ["SNOMED/3"]),
        # 3. Codes outside the vocabulary are left for the next layer
        (["SNOMED/1", "MIMIC_IV_LABITEM/50912"], ["SNOMED/1"]),
        # 4. Nothing in the vocabulary yields nothing
        (["MIMIC_IV_LABITEM/50912"], []),
        # 5. Null codes can never be tokens
        ([None, "SNOMED/1"], ["SNOMED/1"]),
    ],
)
def test_resolves_codes_present_in_the_vocabulary(
    make_vocab: Callable, codes: list[str], resolved: list[str]
) -> None:
    """Asserts only codes the vocabulary holds are emitted, mapped to themselves."""
    frame = pl.LazyFrame({"code": pl.Series(codes, dtype=pl.String)})

    assert_frame_equal(resolve_direct(frame, make_vocab()), _expected(resolved))


def test_drops_columns_the_contract_does_not_carry(make_vocab: Callable) -> None:
    """The inventory's counts must not leak into the layered output."""
    frame = pl.LazyFrame(
        {
            "code": pl.Series(["SNOMED/1"], dtype=pl.String),
            "count": pl.Series([7], dtype=pl.UInt32),
        }
    )

    assert_frame_equal(resolve_direct(frame, make_vocab()), _expected(["SNOMED/1"]))


def test_empty_input_returns_an_empty_frame(make_vocab: Callable) -> None:
    """An empty inventory resolves to nothing, with the schema intact."""
    frame = pl.LazyFrame({"code": pl.Series([], dtype=pl.String)})

    assert_frame_equal(resolve_direct(frame, make_vocab()), _expected([]))


def test_an_empty_vocabulary_resolves_nothing(make_vocab: Callable) -> None:
    """Guards the membership test against an empty token set."""
    frame = pl.LazyFrame({"code": pl.Series(["SNOMED/1"], dtype=pl.String)})
    empty = make_vocab(
        code_tokens=frozenset(), numeric_codes=frozenset(), text_codes=frozenset()
    )

    assert_frame_equal(resolve_direct(frame, empty), _expected([]))
