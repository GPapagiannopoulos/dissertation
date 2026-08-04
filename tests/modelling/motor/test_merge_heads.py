"""Tests that the head merge is exactly the inverse of the split.

`split_heads` is pinned against the oracle, so defining this one as its inverse is
enough; a second oracle target would only restate the same fact.
"""

import pytest
import torch

from thesis.modelling.motor.layers import merge_heads, split_heads

SEQ_LEN = 64
HIDDEN = 768
N_HEADS = 12
HEAD_DIM = HIDDEN // N_HEADS


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((SEQ_LEN, HIDDEN), id="unbatched"),
        pytest.param((3, SEQ_LEN, HIDDEN), id="batched"),
    ],
)
def test_round_trips_with_split_heads(shape: tuple[int, ...]) -> None:
    """A reshape and a transpose either way, so the round trip is exact."""
    x = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32).reshape(
        shape
    )

    assert torch.equal(merge_heads(split_heads(x, N_HEADS)), x)


def test_channels_land_where_the_heads_were() -> None:
    """Head h's values reappear at channels h * head_dim onwards, not interleaved."""
    heads = torch.arange(N_HEADS * SEQ_LEN * HEAD_DIM, dtype=torch.float32).reshape(
        N_HEADS, SEQ_LEN, HEAD_DIM
    )

    merged = merge_heads(heads)

    for h in range(N_HEADS):
        assert torch.equal(merged[:, h * HEAD_DIM : (h + 1) * HEAD_DIM], heads[h])


def test_shape_collapses_the_head_axis() -> None:
    """(..., heads, seq, head_dim) -> (..., seq, heads * head_dim)."""
    assert merge_heads(torch.zeros(3, N_HEADS, SEQ_LEN, HEAD_DIM)).shape == (
        3,
        SEQ_LEN,
        HIDDEN,
    )


def test_rejects_too_few_dimensions() -> None:
    """Without a head axis the transpose would silently reorder something else."""
    with pytest.raises(ValueError, match="three dimensions"):
        merge_heads(torch.zeros(SEQ_LEN, HEAD_DIM))
