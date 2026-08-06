"""Testing suite for the fan-out into sparse embedding pairs.

This is the last step before the encoder, and it is where a position stops being
a row in a frame and becomes a place in a sequence the transformer attends over.
The failure to fear is a position that shifts: the join returns hash-probe order,
and a frame that comes back reordered hands every event another event's tokens
while staying a perfectly well-formed table of real concepts.

The last case runs the pairs through `HierarchicalEmbedding` itself, because the
contract being tested is ultimately about which vectors get summed together.
"""

from collections.abc import Callable

import polars as pl
import pytest
import torch
from polars.testing import assert_frame_equal

from thesis.modelling.motor.layers import HierarchicalEmbedding
from thesis.modelling.motor.tokenizer import expand_leaf_tokens

PAIR_SCHEMA = {"token": pl.UInt32, "position": pl.UInt32}


@pytest.fixture
def make_expansion() -> Callable:
    """Returns a factory for an expansion table, from leaf -> tokens mappings."""

    def _make(**chains: list[int]) -> pl.LazyFrame:
        rows = [
            (int(leaf), token) for leaf, tokens in chains.items() for token in tokens
        ]
        return pl.DataFrame(
            {
                "index": [leaf for leaf, _ in rows],
                "token": [token for _, token in rows],
            },
            schema={"index": pl.UInt32, "token": pl.UInt32},
        ).lazy()

    return _make


@pytest.fixture
def make_sequence() -> Callable:
    """Returns a factory for a tokenised sequence: leaf indices in event order."""

    def _make(*indices: int) -> pl.LazyFrame:
        return pl.DataFrame(
            {"index": list(indices)}, schema={"index": pl.UInt32}
        ).lazy()

    return _make


def test_a_bare_code_fans_out_and_a_valued_one_does_not(
    make_sequence: Callable, make_expansion: Callable
) -> None:
    """The asymmetry the expansion table encodes, seen end to end."""
    expansion = make_expansion(**{"0": [0, 8, 9], "5": [5]})
    events = make_sequence(0, 5)

    result = expand_leaf_tokens(events, expansion).collect()

    assert_frame_equal(
        result,
        pl.DataFrame(
            {"token": [0, 8, 9, 5], "position": [0, 0, 0, 1]}, schema=PAIR_SCHEMA
        ),
    )


def test_a_position_is_the_place_in_the_sequence_not_the_token(
    make_sequence: Callable, make_expansion: Callable
) -> None:
    """The two are both small integers, and confusing them costs nothing visible.

    Here the third event holds token 0, so a fan-out that wrote the token id into
    the position column would produce the same set of numbers in a different
    arrangement -- and silently sum the third event's rows into the first.
    """
    expansion = make_expansion(**{"0": [0], "7": [7]})
    events = make_sequence(7, 7, 0)

    result = expand_leaf_tokens(events, expansion).collect()

    assert result["position"].to_list() == [0, 1, 2]
    assert result["token"].to_list() == [7, 7, 0]


def test_the_same_code_twice_keeps_two_positions(
    make_sequence: Callable, make_expansion: Callable
) -> None:
    """A subject records the same concept repeatedly and each is its own event.

    Collapsing them would shorten the sequence and silently drop the repetition,
    which stage 2.6 kept on purpose as a signal about recording frequency.
    """
    expansion = make_expansion(**{"4": [4, 9]})
    events = make_sequence(4, 4)

    result = expand_leaf_tokens(events, expansion).collect()

    assert result["position"].to_list() == [0, 0, 1, 1]


def test_positions_come_back_in_order(
    make_sequence: Callable, make_expansion: Callable
) -> None:
    """A join returns hash-probe order, and position is assigned by row order.

    This is the same trap that made stage 2.6 and the labeller sort before
    writing; here it would misalign every event with its own tokens.
    """
    expansion = make_expansion(
        **{"0": [0, 1, 2, 3], "1": [1], "2": [2], "3": [3], "4": [4]}
    )
    events = make_sequence(4, 3, 0, 2, 1, 0, 4)

    result = expand_leaf_tokens(events, expansion).collect()

    positions = result["position"].to_list()
    assert positions == sorted(positions)
    assert result.group_by("position").len().sort("position")["len"].to_list() == [
        1,
        1,
        4,
        1,
        1,
        4,
        1,
    ]


def test_a_leaf_the_expansion_does_not_name_contributes_nothing(
    make_sequence: Callable, make_expansion: Callable
) -> None:
    """A miss returns an empty vector in femr, and the position embeds as zero.

    An inner join drops the row rather than carrying a null token into a lookup
    that would index the embedding table with it.
    """
    expansion = make_expansion(**{"0": [0]})
    events = make_sequence(0, 99)

    result = expand_leaf_tokens(events, expansion).collect()

    assert result["position"].to_list() == [0]


def test_an_empty_sequence_yields_typed_empty_pairs(
    make_sequence: Callable, make_expansion: Callable
) -> None:
    """A subject can hold no tokenisable event at all; 48k of ours nearly don't."""
    result = expand_leaf_tokens(make_sequence(), make_expansion(**{"0": [0]})).collect()

    assert result.height == 0
    assert dict(result.schema) == PAIR_SCHEMA


def test_carries_nothing_but_the_pair(
    make_sequence: Callable, make_expansion: Callable
) -> None:
    """The events frame is 22 columns wide and none of them belong in the pairs."""
    events = make_sequence(0).with_columns(
        subject_id=pl.lit(1, dtype=pl.Int64), code=pl.lit("SNOMED/plain")
    )

    result = expand_leaf_tokens(events, make_expansion(**{"0": [0, 7]})).collect()

    assert result.columns == ["token", "position"]


def test_the_pairs_sum_the_right_rows(
    make_sequence: Callable, make_expansion: Callable
) -> None:
    """The contract, asserted through the module that consumes it.

    The table is rigged so row t holds [t, -t], making each position's vector the
    sum of the token ids that reached it.
    """
    embedding = HierarchicalEmbedding(vocab_size=10, hidden_size=2)
    embedding.load_haiku(torch.tensor([[float(t), -float(t)] for t in range(10)]))
    expansion = make_expansion(**{"1": [1, 2, 3], "5": [5]})
    events = make_sequence(1, 5, 1)

    pairs = expand_leaf_tokens(events, expansion).collect()
    embedded = embedding(torch.from_numpy(pairs.to_numpy().astype("int64")), seq_len=3)

    torch.testing.assert_close(
        embedded, torch.tensor([[6.0, -6.0], [5.0, -5.0], [6.0, -6.0]])
    )
