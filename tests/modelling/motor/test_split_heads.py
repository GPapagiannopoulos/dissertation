"""Tests the head reshape, the third layout in this port that fails silently.

Reading the sequence axis as heads returns the same shape as reading the channel axis
as heads, so only the values distinguish them.
"""

import numpy as np
import pytest
import torch
from numpy.lib.npyio import NpzFile

from thesis.modelling.motor.layers import split_heads

HIDDEN = 768
N_HEADS = 12
HEAD_DIM = HIDDEN // N_HEADS
SEQ_LEN = 64


@pytest.fixture
def x() -> torch.Tensor:
    """A (seq_len, hidden_size) tensor with a distinct value per channel."""
    return torch.arange(SEQ_LEN * HIDDEN, dtype=torch.float32).reshape(SEQ_LEN, HIDDEN)


def test_head_h_owns_a_contiguous_block_of_channels(x: torch.Tensor) -> None:
    """The whole point: head h holds channels h * 64 to (h + 1) * 64, all positions.

    This is the assertion that separates the correct reshape from the transposed one.
    """
    heads = split_heads(x, N_HEADS)

    for h in range(N_HEADS):
        assert torch.equal(heads[h], x[:, h * HEAD_DIM : (h + 1) * HEAD_DIM])


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((SEQ_LEN, HIDDEN), id="unbatched"),
        pytest.param((3, SEQ_LEN, HIDDEN), id="batched"),
    ],
)
def test_leading_dimensions_pass_through(shape: tuple[int, ...]) -> None:
    """Negative axes keep a batch dimension in front of the heads."""
    result = split_heads(torch.zeros(shape), N_HEADS)

    assert result.shape == (*shape[:-2], N_HEADS, SEQ_LEN, HEAD_DIM)


def test_moves_no_numbers(x: torch.Tensor) -> None:
    """A reshape and a transpose, so the multiset of values is untouched."""
    assert torch.equal(split_heads(x, N_HEADS).flatten().sort().values, x.flatten())


def test_rejects_an_indivisible_head_count(x: torch.Tensor) -> None:
    """768 channels over 5 heads has no answer; torch's reshape would raise later."""
    with pytest.raises(ValueError, match="divide"):
        split_heads(x, 5)


def test_matches_jax_oracle(jax_oracle: NpzFile) -> None:
    """Reproduces femr's `move_to_batch` on the real layer-0 queries, exactly."""
    q = torch.from_numpy(jax_oracle["layer_00_middle"][:, :HIDDEN])

    result = split_heads(q, N_HEADS)

    assert torch.equal(result, torch.from_numpy(jax_oracle["layer_00_q_heads"]))


def test_probe_arrays_are_what_the_oracle_dumped(jax_oracle: NpzFile) -> None:
    """Guards the oracle's shape assumptions the test above depends on."""
    assert jax_oracle["layer_00_middle"].dtype == np.float32
    assert jax_oracle["layer_00_q_heads"].shape == (N_HEADS, SEQ_LEN, HEAD_DIM)
