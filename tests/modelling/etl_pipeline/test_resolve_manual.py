"""Testing suite for the curated manual resolver."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.modelling.etl_pipeline.codes import MANUAL_CONCEPT_MAP
from thesis.modelling.etl_pipeline.coverage import resolve_manual

MAPPING = {
    "MIMIC_IV_Gender/M": "Gender/M",
    "MIMIC_IV_Race/WHITE": "Race/5",
    "MIMIC_IV_Race/HISPANIC OR LATINO": "Ethnicity/Hispanic",
    "MIMIC_IV_Admission/EW EMER.": "Visit/ERIP",
}


def _inventory(codes: list[str]) -> pl.LazyFrame:
    """Builds the code inventory side of the join."""
    return pl.LazyFrame({"code": pl.Series(codes, dtype=pl.String)})


def _expected(pairs: list[tuple[str, str]]) -> pl.LazyFrame:
    """Builds the frame the resolver is expected to emit."""
    return pl.LazyFrame(
        {
            "code": pl.Series([code for code, _ in pairs], dtype=pl.String),
            "target": pl.Series([target for _, target in pairs], dtype=pl.String),
            "method": pl.Series(["manual"] * len(pairs), dtype=pl.String),
        }
    )


@pytest.mark.parametrize(
    "codes, expected",
    [
        # 0. A curated code resolves to the concept the table names
        (["MIMIC_IV_Gender/M"], [("MIMIC_IV_Gender/M", "Gender/M")]),
        # 1. One MIMIC family may target two OMOP vocabularies
        (
            ["MIMIC_IV_Race/WHITE", "MIMIC_IV_Race/HISPANIC OR LATINO"],
            [
                ("MIMIC_IV_Race/WHITE", "Race/5"),
                ("MIMIC_IV_Race/HISPANIC OR LATINO", "Ethnicity/Hispanic"),
            ],
        ),
        # 2. A value deliberately left out of the table resolves to nothing
        (["MIMIC_IV_Race/UNKNOWN"], []),
        # 3. A family with no curated entries at all resolves to nothing
        (["MIMIC_IV_Transfer/Emergency Department"], []),
        # 4. Only the curated code of the pair is emitted
        (
            ["MIMIC_IV_Gender/M", "MIMIC_IV_Race/OTHER"],
            [("MIMIC_IV_Gender/M", "Gender/M")],
        ),
        # 5. Nothing observed resolves to nothing
        ([], []),
    ],
)
def test_resolves_codes_the_table_curates(
    codes: list[str], expected: list[tuple[str, str]]
) -> None:
    """Asserts observed codes map to the concepts the curated table names."""
    assert_frame_equal(resolve_manual(_inventory(codes), MAPPING), _expected(expected))


def test_does_not_invent_codes_the_events_never_carried() -> None:
    """The output is keyed on what we observed, not on what the table knows."""
    resolved = resolve_manual(_inventory(["MIMIC_IV_Gender/M"]), MAPPING)

    assert_frame_equal(resolved, _expected([("MIMIC_IV_Gender/M", "Gender/M")]))


def test_an_empty_mapping_resolves_nothing() -> None:
    """An empty table still yields the shared schema rather than null columns."""
    resolved = resolve_manual(_inventory(["MIMIC_IV_Gender/M"]), {})

    assert_frame_equal(resolved, _expected([]))


def test_drops_columns_the_contract_does_not_carry() -> None:
    """The inventory's counts must not leak into the layered output."""
    frame = pl.LazyFrame(
        {
            "code": pl.Series(["MIMIC_IV_Gender/M"], dtype=pl.String),
            "count": pl.Series([7], dtype=pl.UInt32),
        }
    )

    resolved = resolve_manual(frame, MAPPING)

    assert_frame_equal(resolved, _expected([("MIMIC_IV_Gender/M", "Gender/M")]))


def test_resolves_against_the_real_curated_table() -> None:
    """Guards the table itself: the anchors we reasoned about must still resolve."""
    codes = [
        "MIMIC_IV_Gender/F",
        "MIMIC_IV_Race/BLACK/AFRICAN AMERICAN",
        "MIMIC_IV_Admission/EW EMER.",
        "MIMIC_IV_Race/PORTUGUESE",
    ]

    resolved = resolve_manual(_inventory(codes), MANUAL_CONCEPT_MAP)

    assert_frame_equal(
        resolved,
        _expected(
            [
                ("MIMIC_IV_Gender/F", "Gender/F"),
                ("MIMIC_IV_Race/BLACK/AFRICAN AMERICAN", "Race/3"),
                ("MIMIC_IV_Admission/EW EMER.", "Visit/ERIP"),
            ]
        ),
    )
