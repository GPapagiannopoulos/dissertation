"""Testing suite for the ancestor search behind the climb."""

from collections.abc import Callable, Mapping

import pytest

from thesis.modelling.etl_pipeline.coverage import MotorVocab, _best_ancestor


def _vocab(
    make_vocab: Callable,
    all_parents: Mapping[str, tuple[str, ...]],
    weights: Mapping[str, float],
) -> MotorVocab:
    """Builds a vocabulary whose tokens are exactly the codes weights names."""
    return make_vocab(
        code_tokens=frozenset(weights),
        numeric_codes=frozenset(),
        text_codes=frozenset(),
        weights=dict(weights),
        all_parents=dict(all_parents),
    )


# A chain of concepts under a single token: T is in the vocabulary, C sits one layer
# below it and G one below that. ORPHAN has an ancestor, but not one MOTOR kept
CHAIN = {
    "T": ("T",),
    "C": ("C", "T"),
    "G": ("G", "C", "T"),
    "ORPHAN": ("ORPHAN", "UNKEPT"),
    "UNKEPT": ("UNKEPT",),
}


@pytest.mark.parametrize(
    "code, expected",
    [
        # 0. A token answers itself without climbing, carrying its own weight
        ("T", ("T", 0, -0.5)),
        # 1. One layer below its token
        ("C", ("T", 1, -0.5)),
        # 2. Two layers, through a concept MOTOR does not hold
        ("G", ("T", 2, -0.5)),
        # 3. An ancestor exists, but no token above it
        ("ORPHAN", None),
        # 4. A root MOTOR dropped has nothing above it at all
        ("UNKEPT", None),
        # 5. A concept the closure never mentions is a dead end, not an error
        ("SNOMED/unheard-of", None),
    ],
)
def test_climbs_to_the_nearest_token(
    make_vocab: Callable, code: str, expected: tuple[str, int, float] | None
) -> None:
    """Asserts the search reports the token, the layers taken and its weight."""
    vocab = _vocab(make_vocab, CHAIN, {"T": -0.5})

    assert _best_ancestor(code, vocab, {}, {}) == expected


def test_reports_the_weight_of_the_token_it_lands_on(make_vocab: Callable) -> None:
    """The weight belongs to the token reached, not to the code that climbed."""
    vocab = _vocab(make_vocab, CHAIN, {"T": -0.5, "C": -0.2})

    assert _best_ancestor("G", vocab, {}, {}) == ("C", 1, -0.2)
    assert _best_ancestor("C", vocab, {}, {}) == ("C", 0, -0.2)


def test_prefers_the_nearer_token_over_the_more_informative_one(
    make_vocab: Callable,
) -> None:
    """Hops rank ahead of weight: a distant token is a vaguer description."""
    all_parents = {
        "X": ("X", "NEAR", "MID", "FAR"),
        "NEAR": ("NEAR",),
        "MID": ("MID", "FAR"),
        "FAR": ("FAR",),
    }
    vocab = _vocab(make_vocab, all_parents, {"NEAR": -0.1, "FAR": -0.9})

    assert _best_ancestor("X", vocab, {}, {}) == ("NEAR", 1, -0.1)


def test_prefers_the_most_informative_token_within_a_layer(
    make_vocab: Callable,
) -> None:
    """Weight is negative entropy, so the lowest of two equally near tokens wins."""
    all_parents = {"X": ("X", "DULL", "SHARP"), "DULL": ("DULL",), "SHARP": ("SHARP",)}
    vocab = _vocab(make_vocab, all_parents, {"DULL": -0.1, "SHARP": -0.9})

    assert _best_ancestor("X", vocab, {}, {}) == ("SHARP", 1, -0.9)


def test_breaks_a_full_tie_on_the_code_itself(make_vocab: Callable) -> None:
    """Equal hops and equal weight must still pick the same token every run."""
    all_parents = {"X": ("X", "B", "A"), "A": ("A",), "B": ("B",)}
    vocab = _vocab(make_vocab, all_parents, {"A": -0.5, "B": -0.5})

    assert _best_ancestor("X", vocab, {}, {}) == ("A", 1, -0.5)


def test_answers_from_the_memo_without_climbing(make_vocab: Callable) -> None:
    """A code already answered is never walked again, however wrong the answer."""
    vocab = _vocab(make_vocab, CHAIN, {"T": -0.5})
    memo: dict[str, tuple[str, int, float] | None] = {"G": ("PLANTED", 9, -0.7)}

    assert _best_ancestor("G", vocab, {}, memo) == ("PLANTED", 9, -0.7)


def test_records_every_code_it_walks(make_vocab: Callable) -> None:
    """Both caches fill with the ancestors met on the way, not just the target."""
    vocab = _vocab(make_vocab, CHAIN, {"T": -0.5})
    parents: dict[str, tuple[str, ...]] = {}
    memo: dict[str, tuple[str, int, float] | None] = {}

    _best_ancestor("G", vocab, parents, memo)

    assert memo == {"G": ("T", 2, -0.5), "C": ("T", 1, -0.5), "T": ("T", 0, -0.5)}
    assert parents == {"G": ("C",), "C": ("T",)}


def test_records_a_dead_end_so_it_is_not_retried(make_vocab: Callable) -> None:
    """Failure is an answer too, and the expensive half is proving there is none."""
    vocab = _vocab(make_vocab, CHAIN, {"T": -0.5})
    memo: dict[str, tuple[str, int, float] | None] = {}

    _best_ancestor("ORPHAN", vocab, {}, memo)

    assert memo == {"ORPHAN": None, "UNKEPT": None}


def test_survives_a_cycle_in_the_closure(make_vocab: Callable) -> None:
    """The real closure is a DAG, but a cycle must degrade rather than recurse away."""
    all_parents = {"A": ("A", "B"), "B": ("B", "A")}
    vocab = _vocab(make_vocab, all_parents, {"T": -0.5})

    assert _best_ancestor("A", vocab, {}, {}) is None
