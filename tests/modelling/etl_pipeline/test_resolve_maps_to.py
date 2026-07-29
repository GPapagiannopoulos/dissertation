"""Testing suite for the Athena 'Maps to' resolver."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.modelling.etl_pipeline.coverage import resolve_maps_to


def _concept(concepts: dict[str, str]) -> pl.LazyFrame:
    """Builds a CONCEPT.csv-shaped frame, every column a String as Athena scans it."""
    vocabularies, concept_codes = zip(
        *(code.split("/", 1) for code in concepts), strict=True
    )
    return pl.LazyFrame(
        {
            "concept_id": pl.Series(list(concepts.values()), dtype=pl.String),
            "concept_name": pl.Series(
                [f"name of {code}" for code in concepts], dtype=pl.String
            ),
            "vocabulary_id": pl.Series(vocabularies, dtype=pl.String),
            "concept_code": pl.Series(concept_codes, dtype=pl.String),
        }
    )


def _relationship(edges: list[tuple[str, str, str, str | None]]) -> pl.LazyFrame:
    """Builds a CONCEPT_RELATIONSHIP.csv-shaped frame from (c1, c2, kind, invalid)."""
    return pl.LazyFrame(
        {
            "concept_id_1": pl.Series([edge[0] for edge in edges], dtype=pl.String),
            "concept_id_2": pl.Series([edge[1] for edge in edges], dtype=pl.String),
            "relationship_id": pl.Series([edge[2] for edge in edges], dtype=pl.String),
            "invalid_reason": pl.Series([edge[3] for edge in edges], dtype=pl.String),
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
            "method": pl.Series(["maps_to"] * len(pairs), dtype=pl.String),
        }
    )


CONCEPTS = {
    "ICD10CM/I50.84": "45591000",
    "SNOMED/443253003": "4229440",
    "SNOMED/84114007": "316139",
    "ICD10PCS/0T773DZ": "2007216",
    "NDC/00121085340": "44923712",
}


@pytest.mark.parametrize(
    "edges, codes, expected",
    [
        # 0. A non-standard code resolves to its standard concept
        (
            [("45591000", "4229440", "Maps to", None)],
            ["ICD10CM/I50.84"],
            [("ICD10CM/I50.84", "SNOMED/443253003")],
        ),
        # 1. A standard code maps to itself, so the climb has a starting point
        (
            [("2007216", "2007216", "Maps to", None)],
            ["ICD10PCS/0T773DZ"],
            [("ICD10PCS/0T773DZ", "ICD10PCS/0T773DZ")],
        ),
        # 2. Relationships other than 'Maps to' are not mappings
        (
            [("45591000", "4229440", "Is a", None)],
            ["ICD10CM/I50.84"],
            [],
        ),
        # 3. The reverse edge OMOP also stores must not be followed
        (
            [("45591000", "4229440", "Mapped from", None)],
            ["ICD10CM/I50.84"],
            [],
        ),
        # 4. A deprecated mapping is not a mapping
        (
            [("45591000", "4229440", "Maps to", "D")],
            ["ICD10CM/I50.84"],
            [],
        ),
        # 5. A code Athena carries but never maps resolves to nothing
        (
            [],
            ["ICD10CM/I50.84"],
            [],
        ),
        # 6. A code Athena has never heard of resolves to nothing
        (
            [("45591000", "4229440", "Maps to", None)],
            ["MIMIC_IV_LABITEM/50912"],
            [],
        ),
        # 7. A mapping whose target Athena cannot name is dropped
        (
            [("45591000", "9999999", "Maps to", None)],
            ["ICD10CM/I50.84"],
            [],
        ),
        # 8. Mappings for codes we never observed are not invented
        (
            [("44923712", "4229440", "Maps to", None)],
            ["ICD10CM/I50.84"],
            [],
        ),
        # 9. Nothing observed resolves to nothing
        (
            [("45591000", "4229440", "Maps to", None)],
            [],
            [],
        ),
    ],
)
def test_resolves_codes_through_maps_to(
    edges: list[tuple[str, str, str, str | None]],
    codes: list[str],
    expected: list[tuple[str, str]],
) -> None:
    """Asserts only observed codes carrying a valid 'Maps to' edge are emitted."""
    resolved = resolve_maps_to(
        _inventory(codes), _concept(CONCEPTS), _relationship(edges)
    )

    assert_frame_equal(resolved, _expected(expected))


def test_emits_a_row_per_standard_concept() -> None:
    """Combination codes map to several standard concepts, and all of them count."""
    edges = [
        ("45591000", "4229440", "Maps to", None),
        ("45591000", "316139", "Maps to", None),
    ]

    resolved = resolve_maps_to(
        _inventory(["ICD10CM/I50.84"]), _concept(CONCEPTS), _relationship(edges)
    )

    assert_frame_equal(
        resolved,
        _expected(
            [
                ("ICD10CM/I50.84", "SNOMED/443253003"),
                ("ICD10CM/I50.84", "SNOMED/84114007"),
            ]
        ),
    )


def test_drops_columns_the_contract_does_not_carry() -> None:
    """Athena's own columns must not leak into the layered output."""
    edges = [("45591000", "4229440", "Maps to", None)]

    resolved = resolve_maps_to(
        _inventory(["ICD10CM/I50.84"]), _concept(CONCEPTS), _relationship(edges)
    )

    assert resolved.collect_schema().names() == ["code", "target", "method"]


def test_keeps_the_inventory_free_of_athena_row_counts() -> None:
    """A code observed once is emitted once per mapping, not once per Athena row."""
    edges = [
        ("45591000", "4229440", "Maps to", None),
        ("2007216", "2007216", "Maps to", None),
    ]

    resolved = resolve_maps_to(
        _inventory(["ICD10CM/I50.84"]), _concept(CONCEPTS), _relationship(edges)
    )

    assert_frame_equal(resolved, _expected([("ICD10CM/I50.84", "SNOMED/443253003")]))
