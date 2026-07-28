"""Testing suite for the SSSOM parent-code resolver."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.modelling.etl_pipeline.coverage import resolve_sssom


def _metadata(parents: dict[str, list[str] | None]) -> pl.LazyFrame:
    """Builds a codes.parquet-shaped frame, description column included."""
    return pl.LazyFrame(
        {
            "code": pl.Series(list(parents), dtype=pl.String),
            "description": pl.Series(
                [f"description of {code}" for code in parents], dtype=pl.String
            ),
            "parent_codes": pl.Series(list(parents.values()), dtype=pl.List(pl.String)),
        }
    )


def _inventory(codes: list[str]) -> pl.LazyFrame:
    """Builds the code inventory side of the join."""
    return pl.LazyFrame({"code": pl.Series(codes, dtype=pl.String)})


def _expected(pairs: list[tuple[str, str]]) -> pl.LazyFrame:
    """Builds the frame the resolver is expected to emit."""
    return pl.LazyFrame(
        {
            "code": pl.Series([code for code, _ in pairs], dtype=pl.String),
            "target": pl.Series([target for _, target in pairs], dtype=pl.String),
            "method": pl.Series(["sssom"] * len(pairs), dtype=pl.String),
        }
    )


@pytest.mark.parametrize(
    "parents, codes, expected",
    [
        # 0. A code with one parent maps to it
        (
            {"MIMIC_IV_LABITEM/50801": ["LOINC/19991-9"]},
            ["MIMIC_IV_LABITEM/50801"],
            [("MIMIC_IV_LABITEM/50801", "LOINC/19991-9")],
        ),
        # 1. An unmapped code falls through to the next layer
        (
            {"MIMIC_IV_ITEM/220045": None},
            ["MIMIC_IV_ITEM/220045"],
            [],
        ),
        # 2. Only the mapped code of the pair is emitted
        (
            {
                "MIMIC_IV_LABITEM/50801": ["LOINC/19991-9"],
                "MIMIC_IV_ITEM/220045": None,
            },
            ["MIMIC_IV_LABITEM/50801", "MIMIC_IV_ITEM/220045"],
            [("MIMIC_IV_LABITEM/50801", "LOINC/19991-9")],
        ),
        # 3. A code we never observed is not invented from the metadata
        (
            {"MIMIC_IV_LABITEM/50801": ["LOINC/19991-9"]},
            ["MIMIC_IV_ITEM/220045"],
            [],
        ),
        # 4. A code absent from the metadata resolves to nothing
        (
            {"MIMIC_IV_LABITEM/50801": ["LOINC/19991-9"]},
            ["MIMIC_IV_LABITEM/50801", "MIMIC_IV_LABITEM/99999"],
            [("MIMIC_IV_LABITEM/50801", "LOINC/19991-9")],
        ),
        # 5. Nothing observed resolves to nothing
        (
            {"MIMIC_IV_LABITEM/50801": ["LOINC/19991-9"]},
            [],
            [],
        ),
    ],
)
def test_resolves_codes_to_their_parents(
    parents: dict[str, list[str] | None],
    codes: list[str],
    expected: list[tuple[str, str]],
) -> None:
    """Asserts observed codes map to the parents the metadata declares."""
    resolved = resolve_sssom(_inventory(codes), _metadata(parents))

    assert_frame_equal(resolved, _expected(expected))


def test_emits_a_row_per_parent() -> None:
    """The column is a list, so a code may carry more than one parent."""
    parents = {"MIMIC_IV_LABITEM/50801": ["LOINC/19991-9", "SNOMED/12345"]}

    resolved = resolve_sssom(_inventory(list(parents)), _metadata(parents))

    assert_frame_equal(
        resolved,
        _expected(
            [
                ("MIMIC_IV_LABITEM/50801", "LOINC/19991-9"),
                ("MIMIC_IV_LABITEM/50801", "SNOMED/12345"),
            ]
        ),
    )


def test_drops_null_parents_from_a_populated_list() -> None:
    """A null element is dropped without taking its siblings with it."""
    parents = {"MIMIC_IV_LABITEM/50801": ["LOINC/19991-9", None]}

    resolved = resolve_sssom(_inventory(list(parents)), _metadata(parents))

    assert_frame_equal(
        resolved, _expected([("MIMIC_IV_LABITEM/50801", "LOINC/19991-9")])
    )


def test_drops_columns_the_contract_does_not_carry() -> None:
    """The metadata's descriptions must not leak into the layered output."""
    parents = {"MIMIC_IV_LABITEM/50801": ["LOINC/19991-9"]}

    resolved = resolve_sssom(_inventory(list(parents)), _metadata(parents))

    assert resolved.collect_schema().names() == ["code", "target", "method"]
