"""Testing suite for the climb from resolved concepts to MOTOR tokens."""

from collections.abc import Callable, Mapping

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.modelling.etl_pipeline.coverage import MotorVocab, climb_to_vocab

# T is the only token. C sits one layer below it, G two. ORPHAN's one ancestor is a
# concept MOTOR dropped, so it is resolvable but unplaceable
CHAIN = {
    "T": ("T",),
    "C": ("C", "T"),
    "G": ("G", "C", "T"),
    "ORPHAN": ("ORPHAN", "UNKEPT"),
    "UNKEPT": ("UNKEPT",),
}


def _vocab(make_vocab: Callable) -> MotorVocab:
    """Builds the vocabulary the chain above is written against."""
    return make_vocab(
        code_tokens=frozenset({"T"}),
        numeric_codes=frozenset(),
        text_codes=frozenset(),
        weights={"T": -0.5},
        all_parents=dict(CHAIN),
    )


def _expected(rows: list[tuple[str, str, int, float]]) -> pl.LazyFrame:
    """Builds the frame the climb is expected to emit for the targets it placed."""
    return pl.LazyFrame(
        {
            "target": pl.Series([row[0] for row in rows], dtype=pl.String),
            "vocab_code": pl.Series([row[1] for row in rows], dtype=pl.String),
            "hops": pl.Series([row[2] for row in rows], dtype=pl.UInt32),
            "weights": pl.Series([row[3] for row in rows], dtype=pl.Float64),
        }
    )


@pytest.mark.parametrize(
    "targets, expected",
    [
        # 0. A target that is already a token is placed at zero hops
        (["T"], [("T", "T", 0, -0.5)]),
        # 1. One layer of climbing
        (["C"], [("C", "T", 1, -0.5)]),
        # 2. Two layers, through a concept outside the vocabulary
        (["G"], [("G", "T", 2, -0.5)]),
        # 3. A target with no token above it emits no row
        (["ORPHAN"], []),
        # 4. Nor does one the closure has never heard of
        (["SNOMED/unheard-of"], []),
        # 5. Placed and unplaceable targets travel together
        (["C", "ORPHAN"], [("C", "T", 1, -0.5)]),
        # 6. Nulls cannot be climbed
        ([None, "C"], [("C", "T", 1, -0.5)]),
        # 7. Nothing in yields nothing out, with the schema intact
        ([], []),
    ],
)
def test_places_each_target_at_its_nearest_token(
    make_vocab: Callable,
    targets: list[str | None],
    expected: list[tuple[str, str, int, float]],
) -> None:
    """Asserts the shared output schema, one row per target the climb could place."""
    frame = pl.LazyFrame({"target": pl.Series(targets, dtype=pl.String)})

    assert_frame_equal(climb_to_vocab(frame, _vocab(make_vocab)), _expected(expected))


def test_collapses_repeated_targets(make_vocab: Callable) -> None:
    """Thousands of source codes share a target; the climb runs once per concept."""
    frame = pl.LazyFrame({"target": pl.Series(["C", "C", "C"], dtype=pl.String)})

    assert_frame_equal(
        climb_to_vocab(frame, _vocab(make_vocab)), _expected([("C", "T", 1, -0.5)])
    )


def test_sorts_by_target(make_vocab: Callable) -> None:
    """The concept map is a build artefact, so its row order must not drift."""
    frame = pl.LazyFrame({"target": pl.Series(["T", "G", "C"], dtype=pl.String)})

    assert_frame_equal(
        climb_to_vocab(frame, _vocab(make_vocab)),
        _expected([("C", "T", 1, -0.5), ("G", "T", 2, -0.5), ("T", "T", 0, -0.5)]),
    )


def test_drops_columns_the_contract_does_not_carry(make_vocab: Callable) -> None:
    """The resolvers' own columns are joined back on later, not carried through."""
    frame = pl.LazyFrame(
        {
            "code": pl.Series(["ICD10CM/I50.84"], dtype=pl.String),
            "target": pl.Series(["C"], dtype=pl.String),
            "method": pl.Series(["maps_to"], dtype=pl.String),
        }
    )

    assert_frame_equal(
        climb_to_vocab(frame, _vocab(make_vocab)), _expected([("C", "T", 1, -0.5)])
    )


def test_shares_one_walk_across_targets(make_vocab: Callable) -> None:
    """Targets on one branch answer consistently, whatever order they arrive in."""
    frame = pl.LazyFrame({"target": pl.Series(["G", "C", "T"], dtype=pl.String)})
    reversed_frame = pl.LazyFrame(
        {"target": pl.Series(["T", "C", "G"], dtype=pl.String)}
    )
    vocab = _vocab(make_vocab)

    assert_frame_equal(
        climb_to_vocab(frame, vocab), climb_to_vocab(reversed_frame, vocab)
    )


@pytest.mark.parametrize(
    "all_parents, weights, expected",
    [
        # 0. Hops outrank weight: the nearer token wins though it says less
        (
            {
                "X": ("X", "NEAR", "MID", "FAR"),
                "NEAR": ("NEAR",),
                "MID": ("MID", "FAR"),
                "FAR": ("FAR",),
            },
            {"NEAR": -0.1, "FAR": -0.9},
            [("X", "NEAR", 1, -0.1)],
        ),
        # 1. Within a layer the most informative token wins
        (
            {"X": ("X", "DULL", "SHARP"), "DULL": ("DULL",), "SHARP": ("SHARP",)},
            {"DULL": -0.1, "SHARP": -0.9},
            [("X", "SHARP", 1, -0.9)],
        ),
        # 2. A full tie falls back to the code, so a run cannot drift from a rerun
        (
            {"X": ("X", "B", "A"), "A": ("A",), "B": ("B",)},
            {"A": -0.5, "B": -0.5},
            [("X", "A", 1, -0.5)],
        ),
    ],
)
def test_ranks_candidate_tokens(
    make_vocab: Callable,
    all_parents: Mapping[str, tuple[str, ...]],
    weights: Mapping[str, float],
    expected: list[tuple[str, str, int, float]],
) -> None:
    """Asserts the ranking the search applies survives into the emitted frame."""
    vocab = make_vocab(
        code_tokens=frozenset(weights),
        numeric_codes=frozenset(),
        text_codes=frozenset(),
        weights=dict(weights),
        all_parents=dict(all_parents),
    )
    frame = pl.LazyFrame({"target": pl.Series(["X"], dtype=pl.String)})

    assert_frame_equal(climb_to_vocab(frame, vocab), _expected(expected))
