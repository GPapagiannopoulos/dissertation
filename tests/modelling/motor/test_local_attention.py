"""Tests masked attention against a spelled-out softmax and against the JAX oracle."""

import pytest
import torch
from numpy.lib.npyio import NpzFile

from thesis.modelling.motor.layers import local_attention, local_attention_mask

SEQ_LEN = 64
HEAD_DIM = 64
N_HEADS = 12
WIDTH = 16
SEGMENT_LENGTH = 32

# torch's fused kernel reassociates the softmax; femr's fallback does not
KERNEL_ATOL = 1e-5


@pytest.fixture
def qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independently drawn q, k and v.

    Reusing one tensor would make the logits symmetric and hide a transposed mask.
    """
    generator = torch.Generator().manual_seed(0)
    return tuple(
        torch.randn(N_HEADS, SEQ_LEN, HEAD_DIM, generator=generator) for _ in range(3)
    )


@pytest.fixture
def mask() -> torch.Tensor:
    """The probe's mask: two segments of 32, window 16."""
    return local_attention_mask(torch.arange(SEQ_LEN) // SEGMENT_LENGTH, WIDTH)


def reference_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Femr's fallback, written out: scale, mask to -inf, softmax, weight the values."""
    logits = queries @ keys.transpose(-1, -2) / (keys.shape[-1] ** 0.5)
    return torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1) @ values


def test_matches_a_spelled_out_softmax(
    qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor], mask: torch.Tensor
) -> None:
    """Pins the scale as 1 / sqrt(head_dim), which torch supplies by default."""
    result = local_attention(*qkv, mask)

    torch.testing.assert_close(
        result, reference_attention(*qkv, mask), rtol=0, atol=KERNEL_ATOL
    )


def test_masked_keys_have_no_influence(
    qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor], mask: torch.Tensor
) -> None:
    """Rewriting a masked position must not move the queries that cannot see it.

    Position 0 is in the first segment, so nothing from position 32 onwards may reach
    it, and neither may anything more than 16 positions back within its own segment.
    """
    queries, keys, values = qkv
    before = local_attention(queries, keys, values, mask)

    keys[:, SEGMENT_LENGTH:] += 100.0
    values[:, SEGMENT_LENGTH:] += 100.0

    after = local_attention(queries, keys, values, mask)
    torch.testing.assert_close(
        after[:, :SEGMENT_LENGTH], before[:, :SEGMENT_LENGTH], rtol=0, atol=KERNEL_ATOL
    )


def test_output_is_a_convex_combination_of_visible_values(
    qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor], mask: torch.Tensor
) -> None:
    """Softmax weights sum to one, so no output can leave the visible values' range."""
    queries, keys, values = qkv

    result = local_attention(queries, keys, values, mask)

    assert (result.amax(dim=-2) <= values.amax(dim=-2) + KERNEL_ATOL).all()
    assert (result.amin(dim=-2) >= values.amin(dim=-2) - KERNEL_ATOL).all()


def test_a_self_only_mask_returns_the_values(
    qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Width 0 leaves one key per query, so the softmax is 1 and the output is v."""
    queries, keys, values = qkv
    self_only = local_attention_mask(torch.zeros(SEQ_LEN, dtype=torch.long), 0)

    result = local_attention(queries, keys, values, self_only)

    torch.testing.assert_close(result, values, rtol=0, atol=KERNEL_ATOL)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((N_HEADS, SEQ_LEN, HEAD_DIM), id="per_head"),
        pytest.param((2, N_HEADS, SEQ_LEN, HEAD_DIM), id="batched"),
    ],
)
def test_shape_is_preserved(mask: torch.Tensor, shape: tuple[int, ...]) -> None:
    """The mask broadcasts over every leading dimension."""
    x = torch.zeros(shape)

    assert local_attention(x, x, x, mask).shape == shape


def test_rejects_a_non_boolean_mask(
    qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor], mask: torch.Tensor
) -> None:
    """A float mask is ADDED to the logits by torch, not applied as a predicate.

    Passing one silently turns every masked position into a small bias instead of an
    exclusion.
    """
    with pytest.raises(ValueError, match="boolean"):
        local_attention(*qkv, mask.float())


def test_rejects_mismatched_inputs(mask: torch.Tensor) -> None:
    """q, k and v come from one projection, so a shape disagreement is a bug."""
    x = torch.zeros(N_HEADS, SEQ_LEN, HEAD_DIM)

    with pytest.raises(ValueError, match="one shape"):
        local_attention(x, x, torch.zeros(N_HEADS, SEQ_LEN, HEAD_DIM + 1), mask)


def test_matches_jax_oracle(jax_oracle: NpzFile) -> None:
    """Reproduces the released implementation on its own probe.

    On CPU, XLA compiles femr's `local_attention_fallback`, so this is the semantics
    the model runs, not a reference path beside it.
    """
    queries = torch.from_numpy(jax_oracle["attention_probe_q"])
    keys = torch.from_numpy(jax_oracle["attention_probe_k"])
    values = torch.from_numpy(jax_oracle["attention_probe_v"])
    segment_ids = torch.arange(SEQ_LEN) & int(jax_oracle["attention_probe_length_mask"])
    mask = local_attention_mask(segment_ids, int(jax_oracle["attention_probe_width"]))

    result = local_attention(queries, keys, values, mask)

    torch.testing.assert_close(
        result,
        torch.from_numpy(jax_oracle["attention_probe_out"]),
        rtol=0,
        atol=KERNEL_ATOL,
    )


def test_probe_arrays_are_what_the_oracle_dumped(jax_oracle: NpzFile) -> None:
    """Guards the oracle's shape assumptions the test above depends on."""
    assert jax_oracle["attention_probe_q"].shape == (N_HEADS, SEQ_LEN, HEAD_DIM)
    assert jax_oracle["attention_probe_out"].shape == (N_HEADS, SEQ_LEN, HEAD_DIM)
