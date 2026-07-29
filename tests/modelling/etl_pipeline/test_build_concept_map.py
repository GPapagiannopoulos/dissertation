"""Testing suite for the concept map assembly."""

from collections.abc import Callable, Mapping

import polars as pl
from polars.testing import assert_frame_equal

from thesis.modelling.etl_pipeline.coverage import MotorVocab, build_concept_map

# Three tokens, and two concepts that must climb to reach one. ORPHAN resolves but
# sits under a concept MOTOR dropped, so it is placeable by no layer
ALL_PARENTS = {
    "SNOMED/T": ("SNOMED/T",),
    "SNOMED/S": ("SNOMED/S",),
    "SNOMED/N": ("SNOMED/N",),
    "RxNorm/R": ("RxNorm/R",),
    "SNOMED/C": ("SNOMED/C", "SNOMED/T"),
    "SNOMED/D": ("SNOMED/D", "SNOMED/C", "SNOMED/T"),
    "SNOMED/ORPHAN": ("SNOMED/ORPHAN", "SNOMED/UNKEPT"),
    "SNOMED/UNKEPT": ("SNOMED/UNKEPT",),
}
# N is near but says little, T is informative but two layers up: the pair exists so
# that ranking on hops and ranking on weight cannot agree by accident
WEIGHTS = {"SNOMED/T": -0.5, "SNOMED/S": -0.9, "SNOMED/N": -0.05, "RxNorm/R": -0.2}

# Athena keys on integer ids, so every code the tests mention needs one
CONCEPT_IDS = {
    "ICD10CM/A": "1",
    "ICD10CM/B": "2",
    "ICD10CM/ORPH": "3",
    "SNOMED/T": "4",
    "SNOMED/S": "5",
    "SNOMED/C": "6",
    "SNOMED/D": "7",
    "SNOMED/ORPHAN": "8",
    "MIMIC_IV_Gender/M": "9",
    "RxNorm/R": "10",
    "SNOMED/N": "11",
    "MIMIC_IV_LABITEM/1": "12",
}


def _vocab(make_vocab: Callable) -> MotorVocab:
    """Builds the vocabulary the world above is written against."""
    return make_vocab(
        code_tokens=frozenset(WEIGHTS),
        numeric_codes=frozenset(),
        text_codes=frozenset(),
        weights=dict(WEIGHTS),
        all_parents=dict(ALL_PARENTS),
    )


def _inventory(counts: dict[str | None, int]) -> pl.LazyFrame:
    """Builds code_inventory's output, counts included."""
    return pl.LazyFrame(
        {
            "code": pl.Series(list(counts), dtype=pl.String),
            "count": pl.Series(list(counts.values()), dtype=pl.UInt32),
        }
    )


def _concept() -> pl.LazyFrame:
    """Builds a CONCEPT.csv-shaped frame holding every code the tests name."""
    vocabularies, concept_codes = zip(
        *(code.split("/", 1) for code in CONCEPT_IDS), strict=True
    )
    return pl.LazyFrame(
        {
            "concept_id": pl.Series(list(CONCEPT_IDS.values()), dtype=pl.String),
            "vocabulary_id": pl.Series(vocabularies, dtype=pl.String),
            "concept_code": pl.Series(concept_codes, dtype=pl.String),
        }
    )


def _relationship(edges: list[tuple[str, str]]) -> pl.LazyFrame:
    """Builds a CONCEPT_RELATIONSHIP.csv-shaped frame of valid 'Maps to' edges."""
    return pl.LazyFrame(
        {
            "concept_id_1": pl.Series(
                [CONCEPT_IDS[source] for source, _ in edges], dtype=pl.String
            ),
            "concept_id_2": pl.Series(
                [CONCEPT_IDS[target] for _, target in edges], dtype=pl.String
            ),
            "relationship_id": pl.Series(["Maps to"] * len(edges), dtype=pl.String),
            "invalid_reason": pl.Series([None] * len(edges), dtype=pl.String),
        }
    )


def _metadata(parents: dict[str, list[str] | None]) -> pl.LazyFrame:
    """Builds a codes.parquet-shaped frame of SSSOM parent codes."""
    return pl.LazyFrame(
        {
            "code": pl.Series(list(parents), dtype=pl.String),
            "parent_codes": pl.Series(list(parents.values()), dtype=pl.List(pl.String)),
        }
    )


def _expected(rows: list[tuple[str, int, str, str, str, int, float]]) -> pl.LazyFrame:
    """Builds the artefact the assembly is expected to emit."""
    columns = list(zip(*rows, strict=True)) if rows else [[]] * 7
    return pl.LazyFrame(
        {
            "code": pl.Series(columns[0], dtype=pl.String),
            "count": pl.Series(columns[1], dtype=pl.UInt32),
            "target": pl.Series(columns[2], dtype=pl.String),
            "method": pl.Series(columns[3], dtype=pl.String),
            "vocab_code": pl.Series(columns[4], dtype=pl.String),
            "hops": pl.Series(columns[5], dtype=pl.UInt32),
            "weights": pl.Series(columns[6], dtype=pl.Float64),
        }
    )


def _run(
    make_vocab: Callable,
    counts: dict[str | None, int],
    *,
    edges: list[tuple[str, str]] | None = None,
    parents: dict[str, list[str] | None] | None = None,
    manual: Mapping[str, str] | None = None,
) -> pl.LazyFrame:
    """Assembles a map over the shared world, each layer's data optional."""
    return build_concept_map(
        _inventory(counts),
        _vocab(make_vocab),
        _metadata(parents or {}),
        _concept(),
        _relationship(edges or []),
        manual or {},
    )


def test_places_a_code_through_the_athena_bridge(make_vocab: Callable) -> None:
    """The ordinary path: a non-standard code maps to a concept, which climbs."""
    result = _run(make_vocab, {"ICD10CM/A": 7}, edges=[("ICD10CM/A", "SNOMED/C")])

    assert_frame_equal(
        result,
        _expected([("ICD10CM/A", 7, "SNOMED/C", "maps_to", "SNOMED/T", 1, -0.5)]),
    )


def test_a_token_is_claimed_by_the_first_layer_only(make_vocab: Callable) -> None:
    """A code two layers could resolve is resolved once, by the more reliable one."""
    result = _run(make_vocab, {"SNOMED/T": 3}, edges=[("SNOMED/T", "SNOMED/S")])

    assert_frame_equal(
        result, _expected([("SNOMED/T", 3, "SNOMED/T", "direct", "SNOMED/T", 0, -0.5)])
    )


def test_a_crosswalked_itemid_never_reaches_athena(make_vocab: Callable) -> None:
    """SSSOM is published, 'Maps to' is derived, so the crosswalk wins the code.

    The Athena route is given the better-ranked target on purpose: what decides the
    row is the chain's precedence, not the ranking that settles fan-out within a layer.
    """
    result = _run(
        make_vocab,
        {"MIMIC_IV_LABITEM/1": 4},
        parents={"MIMIC_IV_LABITEM/1": ["SNOMED/D"]},
        edges=[("MIMIC_IV_LABITEM/1", "SNOMED/N")],
    )

    assert_frame_equal(
        result,
        _expected(
            [("MIMIC_IV_LABITEM/1", 4, "SNOMED/D", "sssom", "SNOMED/T", 2, -0.5)]
        ),
    )


def test_the_manual_table_is_the_last_resort(make_vocab: Callable) -> None:
    """Hand-written entries yield to anything the vocabulary can say for itself.

    The hand-written target is the more informative of the two, so a run that kept it
    would be honouring the ranking over the chain, which is the failure to catch.
    """
    result = _run(
        make_vocab,
        {"MIMIC_IV_Gender/M": 2},
        edges=[("MIMIC_IV_Gender/M", "SNOMED/T")],
        manual={"MIMIC_IV_Gender/M": "SNOMED/S"},
    )

    assert_frame_equal(
        result,
        _expected(
            [("MIMIC_IV_Gender/M", 2, "SNOMED/T", "maps_to", "SNOMED/T", 0, -0.5)]
        ),
    )


def test_reaches_codes_no_other_layer_can(make_vocab: Callable) -> None:
    """Running last is not the same as running rarely: it owns what nothing reaches."""
    result = _run(
        make_vocab, {"MIMIC_IV_Race/1": 5}, manual={"MIMIC_IV_Race/1": "SNOMED/T"}
    )

    assert_frame_equal(
        result,
        _expected([("MIMIC_IV_Race/1", 5, "SNOMED/T", "manual", "SNOMED/T", 0, -0.5)]),
    )


def test_keeps_the_nearest_of_several_targets(make_vocab: Callable) -> None:
    """Hops outrank weight: a distant token is a vaguer description of the code."""
    result = _run(
        make_vocab,
        {"ICD10CM/A": 9},
        edges=[("ICD10CM/A", "SNOMED/D"), ("ICD10CM/A", "SNOMED/N")],
    )

    assert_frame_equal(
        result,
        _expected([("ICD10CM/A", 9, "SNOMED/N", "maps_to", "SNOMED/N", 0, -0.05)]),
    )


def test_keeps_the_most_informative_of_equally_near_targets(
    make_vocab: Callable,
) -> None:
    """Within a layer the lowest weight wins, weight being negative entropy."""
    result = _run(
        make_vocab,
        {"ICD10CM/A": 9},
        edges=[("ICD10CM/A", "SNOMED/T"), ("ICD10CM/A", "SNOMED/S")],
    )

    assert_frame_equal(
        result,
        _expected([("ICD10CM/A", 9, "SNOMED/S", "maps_to", "SNOMED/S", 0, -0.9)]),
    )


def test_prefers_a_climbable_target_over_a_dead_one(make_vocab: Callable) -> None:
    """The reason fan-out waits: a dead target must not outrank a surviving sibling."""
    result = _run(
        make_vocab,
        {"ICD10CM/A": 1},
        edges=[("ICD10CM/A", "SNOMED/ORPHAN"), ("ICD10CM/A", "SNOMED/D")],
    )

    assert_frame_equal(
        result,
        _expected([("ICD10CM/A", 1, "SNOMED/D", "maps_to", "SNOMED/T", 2, -0.5)]),
    )


def test_drops_a_code_whose_every_target_dies(make_vocab: Callable) -> None:
    """Resolved is not placed; the report recovers these by anti-join, per method."""
    result = _run(
        make_vocab, {"ICD10CM/ORPH": 6}, edges=[("ICD10CM/ORPH", "SNOMED/ORPHAN")]
    )

    assert_frame_equal(result, _expected([]))


def test_drops_a_code_no_layer_resolves(make_vocab: Callable) -> None:
    """An unmappable family emits no row rather than a null target."""
    result = _run(make_vocab, {"MIMIC_IV_Service/CMED": 8})

    assert_frame_equal(result, _expected([]))


def test_drops_null_codes(make_vocab: Callable) -> None:
    """Null-coded events are real and stay in the denominator, but map to nothing."""
    result = _run(make_vocab, {None: 11, "SNOMED/T": 3})

    assert_frame_equal(
        result, _expected([("SNOMED/T", 3, "SNOMED/T", "direct", "SNOMED/T", 0, -0.5)])
    )


def test_carries_the_event_counts_through(make_vocab: Callable) -> None:
    """Coverage is weighted by events, so the counts must survive the assembly."""
    result = _run(
        make_vocab,
        {"SNOMED/T": 100, "ICD10CM/A": 5},
        edges=[("ICD10CM/A", "SNOMED/C")],
    ).collect()

    assert dict(zip(result["code"], result["count"], strict=True)) == {
        "SNOMED/T": 100,
        "ICD10CM/A": 5,
    }


def test_sorts_by_code(make_vocab: Callable) -> None:
    """The map is a build artefact, so its row order must not drift between runs."""
    result = _run(
        make_vocab,
        {"SNOMED/T": 1, "ICD10CM/A": 1, "MIMIC_IV_Race/1": 1},
        edges=[("ICD10CM/A", "SNOMED/C")],
        manual={"MIMIC_IV_Race/1": "RxNorm/R"},
    ).collect()

    assert result["code"].to_list() == ["ICD10CM/A", "MIMIC_IV_Race/1", "SNOMED/T"]


def test_an_empty_inventory_yields_an_empty_map(make_vocab: Callable) -> None:
    """Nothing in, nothing out, with the schema intact for the parquet write."""
    assert_frame_equal(_run(make_vocab, {}), _expected([]))
