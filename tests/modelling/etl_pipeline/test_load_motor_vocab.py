"""Testing suite for the MOTOR dictionary extraction helper."""

from collections.abc import Callable

import pytest

from thesis.modelling.etl_pipeline.coverage import load_motor_vocab

CODE, NUMERIC, TEXT, UNUSED = 0, 1, 2, 3


@pytest.mark.parametrize(
    "types, expected",
    [
        # 0. Each type lands in its own set
        (
            {"SNOMED/1": CODE, "LOINC/2": NUMERIC, "SNOMED/3": TEXT},
            {
                "code_tokens": {"SNOMED/1"},
                "numeric_codes": {"LOINC/2"},
                "text_codes": {"SNOMED/3"},
            },
        ),
        # 1. Unused entries reach no set at all
        (
            {"SNOMED/1": CODE, "SNOMED/2": UNUSED},
            {
                "code_tokens": {"SNOMED/1"},
                "numeric_codes": set(),
                "text_codes": set(),
            },
        ),
        # 2. An unrecognised type is skipped rather than raising
        (
            {"SNOMED/1": CODE, "SNOMED/2": 99},
            {
                "code_tokens": {"SNOMED/1"},
                "numeric_codes": set(),
                "text_codes": set(),
            },
        ),
    ],
)
def test_partitions_entries_by_type(
    make_dictionary: Callable,
    rollup_entry: Callable,
    types: dict[str, int],
    expected: dict[str, set[str]],
) -> None:
    """Asserts each entry type populates only its own token set."""
    rollup = [rollup_entry(code, kind) for code, kind in types.items()]
    path = make_dictionary(ontology_rollup=rollup)

    vocab = load_motor_vocab(path, vocab_size=len(rollup))

    assert vocab.code_tokens == expected["code_tokens"]
    assert vocab.numeric_codes == expected["numeric_codes"]
    assert vocab.text_codes == expected["text_codes"]


def test_truncates_at_vocab_size(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """Entries past the cut are not tokens, however real they look."""
    rollup = [
        rollup_entry("SNOMED/kept", CODE),
        rollup_entry("SNOMED/dropped", CODE),
    ]
    path = make_dictionary(ontology_rollup=rollup)

    vocab = load_motor_vocab(path, vocab_size=1)

    assert vocab.code_tokens == {"SNOMED/kept"}


def test_collapses_repeated_code_strings(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """A code binned into several numeric entries is still one member."""
    rollup = [
        rollup_entry("LOINC/2160-0", NUMERIC, val_start=0.0, val_end=1.0),
        rollup_entry("LOINC/2160-0", NUMERIC, val_start=1.0, val_end=2.0),
        rollup_entry("LOINC/2160-0", NUMERIC, val_start=2.0, val_end=3.0),
    ]
    path = make_dictionary(ontology_rollup=rollup)

    vocab = load_motor_vocab(path, vocab_size=len(rollup))

    assert vocab.numeric_codes == {"LOINC/2160-0"}


def test_code_may_hold_tokens_of_several_types(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """The sets overlap: a lab carries both an identity token and value bins."""
    rollup = [
        rollup_entry("LOINC/2160-0", CODE),
        rollup_entry("LOINC/2160-0", NUMERIC),
    ]
    path = make_dictionary(ontology_rollup=rollup)

    vocab = load_motor_vocab(path, vocab_size=len(rollup))

    assert "LOINC/2160-0" in vocab.code_tokens
    assert "LOINC/2160-0" in vocab.numeric_codes


def test_records_the_weight_of_every_token(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """Every token carries a weight, whichever set it landed in."""
    rollup = [
        rollup_entry("SNOMED/1", CODE, weight=-0.4),
        rollup_entry("LOINC/2", NUMERIC, weight=-0.2),
        rollup_entry("SNOMED/3", TEXT, weight=-0.1),
    ]
    path = make_dictionary(ontology_rollup=rollup)

    vocab = load_motor_vocab(path, vocab_size=len(rollup))

    assert vocab.weights == {"SNOMED/1": -0.4, "LOINC/2": -0.2, "SNOMED/3": -0.1}


@pytest.mark.parametrize(
    "weights",
    [
        # 0. The best entry arrives last
        [-0.1, -0.2, -0.9],
        # 1. The best entry arrives first, so it must survive the later ones
        [-0.9, -0.2, -0.1],
        # 2. The best entry is buried mid-list
        [-0.2, -0.9, -0.1],
    ],
)
def test_keeps_the_most_informative_entry_per_code(
    make_dictionary: Callable, rollup_entry: Callable, weights: list[float]
) -> None:
    """A code repeats once per numeric bin; weight is negative, so min() is best."""
    rollup = [
        rollup_entry("LOINC/2160-0", NUMERIC, weight=weight) for weight in weights
    ]
    path = make_dictionary(ontology_rollup=rollup)

    vocab = load_motor_vocab(path, vocab_size=len(rollup))

    assert vocab.weights == {"LOINC/2160-0": -0.9}


def test_records_a_zero_weight(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """Zero is a legal weight -- a concept firing exactly when its parent does."""
    path = make_dictionary(ontology_rollup=[rollup_entry("SNOMED/1", CODE, weight=0.0)])

    vocab = load_motor_vocab(path, vocab_size=1)

    assert vocab.weights == {"SNOMED/1": 0.0}


def test_ignores_the_weight_of_entries_that_bear_no_token(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """An unused slot must not win the tie-break for a code it never tokenised."""
    rollup = [
        rollup_entry("SNOMED/1", CODE, weight=-0.1),
        rollup_entry("SNOMED/1", UNUSED, weight=-0.9),
        rollup_entry("SNOMED/2", UNUSED, weight=-0.5),
    ]
    path = make_dictionary(ontology_rollup=rollup)

    vocab = load_motor_vocab(path, vocab_size=len(rollup))

    assert vocab.weights == {"SNOMED/1": -0.1}


def test_ignores_the_weight_of_entries_past_the_cut(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """Truncation binds the weights too: a better entry outside the vocab is not one."""
    rollup = [
        rollup_entry("SNOMED/1", CODE, weight=-0.1),
        rollup_entry("SNOMED/1", CODE, weight=-0.9),
    ]
    path = make_dictionary(ontology_rollup=rollup)

    vocab = load_motor_vocab(path, vocab_size=1)

    assert vocab.weights == {"SNOMED/1": -0.1}


def test_weights_cover_exactly_the_vocabulary(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """The climb looks weights up for anything in tokens, so the two must agree."""
    rollup = [
        rollup_entry("SNOMED/1", CODE),
        rollup_entry("LOINC/2", NUMERIC),
        rollup_entry("SNOMED/3", TEXT),
        rollup_entry("SNOMED/4", UNUSED),
    ]
    path = make_dictionary(ontology_rollup=rollup)

    vocab = load_motor_vocab(path, vocab_size=len(rollup))

    assert vocab.weights.keys() == vocab.tokens


def test_returns_the_whole_ancestor_map(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """The climb walks codes outside the vocabulary, so the map is not filtered."""
    all_parents = {
        "SNOMED/1": ("SNOMED/1", "SNOMED/root"),
        "SNOMED/absent": ("SNOMED/absent", "SNOMED/root"),
    }
    path = make_dictionary(
        ontology_rollup=[rollup_entry("SNOMED/1", CODE)], all_parents=all_parents
    )

    vocab = load_motor_vocab(path, vocab_size=1)

    assert vocab.all_parents == all_parents
    assert vocab.all_parents["SNOMED/1"] == ("SNOMED/1", "SNOMED/root")


@pytest.mark.parametrize(
    "vocab_size",
    [
        # 0. Zero would silently yield empty sets
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
        load_motor_vocab(path, vocab_size=vocab_size)


def test_rejects_vocab_size_beyond_the_dictionary(
    make_dictionary: Callable, rollup_entry: Callable
) -> None:
    """A size the dictionary cannot honour means the tokens would not line up."""
    path = make_dictionary(ontology_rollup=[rollup_entry("SNOMED/1", CODE)])

    with pytest.raises(ValueError, match="exceeds the 1 entries"):
        load_motor_vocab(path, vocab_size=2)
