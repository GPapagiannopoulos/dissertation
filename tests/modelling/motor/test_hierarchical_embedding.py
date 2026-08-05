"""Tests the ancestor-sum embedding against the real embedded sequence.

This is the only ported piece that reaches the oracle bit-exactly: it gathers rows
and adds them, with no matmul to accumulate in a different order. So the oracle
assertion below runs at zero tolerance, and any drift at all is a bug rather than
arithmetic.
"""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from numpy.lib.npyio import NpzFile

from thesis.modelling.motor.layers import HierarchicalEmbedding

VOCAB = 65536
HIDDEN = 768
SEQ_LEN = 64


@pytest.fixture
def loaded(oracle_param: Callable[[str], np.ndarray]) -> HierarchicalEmbedding:
    """The embedding carrying the released table."""
    module = HierarchicalEmbedding(VOCAB, HIDDEN)
    module.load_haiku(torch.from_numpy(oracle_param("embed::embeddings")))
    return module


@pytest.fixture
def sparse_pairs(jax_oracle: NpzFile) -> torch.Tensor:
    """The oracle batch's (read, write) pairs, cast off uint32 so torch can index."""
    return torch.from_numpy(jax_oracle["batch_sparse_token_indices"].astype(np.int64))


@pytest.fixture
def tiny() -> HierarchicalEmbedding:
    """A four-row table whose rows are trivially recognisable in a sum."""
    module = HierarchicalEmbedding(4, 2)
    module.load_haiku(torch.tensor([[1.0, 0.0], [0.0, 1.0], [10.0, 0.0], [0.0, 10.0]]))
    return module


def test_matches_jax_oracle(
    loaded: HierarchicalEmbedding, sparse_pairs: torch.Tensor, jax_oracle: NpzFile
) -> None:
    """Gathers and adds, so it must reproduce `embedded` to the bit."""
    result = loaded(sparse_pairs, SEQ_LEN)

    torch.testing.assert_close(
        result, torch.from_numpy(jax_oracle["embedded"]), rtol=0, atol=0
    )


def test_pair_order_does_not_change_the_result(
    loaded: HierarchicalEmbedding, sparse_pairs: torch.Tensor
) -> None:
    """The control: the sum is an accumulation, so the pairs' order is irrelevant.

    femr emits them sorted by the write index, but nothing here depends on that. An
    implementation that consumed the sortedness — bagging by contiguous runs, or
    taking the last write — would break. Not exact, because reordering a float sum
    reassociates it.
    """
    shuffled = sparse_pairs[torch.randperm(sparse_pairs.shape[0])]

    torch.testing.assert_close(
        loaded(shuffled, SEQ_LEN), loaded(sparse_pairs, SEQ_LEN), rtol=0, atol=1e-6
    )


def test_ancestors_are_summed_not_overwritten(tiny: HierarchicalEmbedding) -> None:
    """Two pairs naming one position contribute both rows, not the last."""
    pairs = torch.tensor([[0, 0], [1, 0]])

    torch.testing.assert_close(tiny(pairs, 1), torch.tensor([[1.0, 1.0]]))


def test_positions_named_by_no_pair_are_zero(tiny: HierarchicalEmbedding) -> None:
    """seq_len is an argument, so leading and trailing gaps both survive."""
    pairs = torch.tensor([[3, 1]])

    torch.testing.assert_close(
        tiny(pairs, 3), torch.tensor([[0.0, 0.0], [0.0, 10.0], [0.0, 0.0]])
    )


def test_columns_are_not_interchangeable(tiny: HierarchicalEmbedding) -> None:
    """Column 0 reads from the table, column 1 writes to a position.

    Pinned because both columns are integers of the same dtype, so a swap is not a
    type error and would only surface as wrong numbers.
    """
    torch.testing.assert_close(
        tiny(torch.tensor([[2, 1]]), 3),
        torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]]),
    )


def test_gradients_reach_the_table(tiny: HierarchicalEmbedding) -> None:
    """The index_add_ into a fresh buffer must stay in the autograd graph.

    Otherwise training over this backbone would silently see no embedding gradient.
    """
    tiny(torch.tensor([[0, 0], [1, 0]]), 1).sum().backward()

    assert tiny.embeddings.weight.grad is not None
    assert tiny.embeddings.weight.grad[2:].eq(0).all()


def test_load_haiku_owns_its_table(
    oracle_param: Callable[[str], np.ndarray],
) -> None:
    """Copied, not aliased: torch.from_numpy shares the checkpoint's memory."""
    table = torch.from_numpy(oracle_param("embed::embeddings"))
    module = HierarchicalEmbedding(VOCAB, HIDDEN)
    module.load_haiku(table)
    before = module.embeddings.weight[0, 0].item()

    table[0, 0] = -999.0

    assert module.embeddings.weight[0, 0].item() == before
    assert module.embeddings.weight.requires_grad


@pytest.mark.parametrize(
    "table",
    [
        pytest.param(torch.ones(VOCAB + 1, HIDDEN), id="vocab"),
        pytest.param(torch.ones(VOCAB, HIDDEN - 1), id="width"),
    ],
)
def test_load_haiku_rejects_a_mismatched_table(table: torch.Tensor) -> None:
    """A table from a differently shaped model must not partially load."""
    with pytest.raises(ValueError):
        HierarchicalEmbedding(VOCAB, HIDDEN).load_haiku(table)


@pytest.mark.parametrize(
    "pairs",
    [
        pytest.param(torch.tensor([[4, 0]]), id="token-past-the-table"),
        pytest.param(torch.tensor([[-1, 0]]), id="token-below-the-table"),
        pytest.param(torch.tensor([[0, 3]]), id="position-past-the-sequence"),
        pytest.param(torch.tensor([[0, -1]]), id="position-below-the-sequence"),
    ],
)
def test_rejects_indices_outside_the_table_or_the_sequence(
    tiny: HierarchicalEmbedding, pairs: torch.Tensor
) -> None:
    """Out-of-range indices raise here, where femr silently absorbs them.

    Its gather fills bad reads with zero and its scatter drops bad writes, which
    would turn a broken tokeniser into quietly wrong vectors.
    """
    with pytest.raises(ValueError):
        tiny(pairs, 3)


@pytest.mark.parametrize(
    "pairs",
    [
        pytest.param(torch.tensor([0, 1]), id="one-dimensional"),
        pytest.param(torch.tensor([[0, 1, 2]]), id="three-columns"),
        pytest.param(torch.tensor([[0.0, 1.0]]), id="float"),
    ],
)
def test_rejects_misshapen_pairs(
    tiny: HierarchicalEmbedding, pairs: torch.Tensor
) -> None:
    """The shape contract raises ValueError rather than asserting.

    python -O strips asserts, and these guards are load-bearing.
    """
    with pytest.raises(ValueError):
        tiny(pairs, 3)


def test_empty_pairs_give_a_zero_sequence(tiny: HierarchicalEmbedding) -> None:
    """A sequence with no tokens at all embeds to zeros.

    Pinned because the range guards reduce over the pairs, and min() over an empty
    tensor raises.
    """
    torch.testing.assert_close(
        tiny(torch.zeros(0, 2, dtype=torch.long), 2), torch.zeros(2, 2)
    )


def test_probe_arrays_are_what_the_oracle_dumped(jax_oracle: NpzFile) -> None:
    """Guards the shape assumptions the oracle test above depends on."""
    assert jax_oracle["embedded"].shape == (SEQ_LEN, HIDDEN)
    assert jax_oracle["batch_sparse_token_indices"].shape[1] == 2
