"""Tests the block's four front projections against the real layer-0 activations.

The oracle comparison checks the weights, their orientation, the biases and
the split boundaries in a single assertion, because a column out of place puts
the wrong numbers in the wrong projection.
"""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from numpy.lib.npyio import NpzFile

from thesis.modelling.motor.layers import InputProjection

HIDDEN = 768
INTERMEDIATE = 3072
IN_FEATURES = 770
SEQ_LEN = 64

# fp32 matmuls over 770 terms, accumulated in a different order by JAX and torch.
# Measured at 2.4e-06 against values reaching 4.3, so this leaves 4x headroom.
GEMM_ATOL = 1e-5


@pytest.fixture
def projection() -> InputProjection:
    """A projection at MOTOR's real widths, still randomly initialised."""
    return InputProjection(IN_FEATURES, HIDDEN, INTERMEDIATE)


@pytest.fixture
def loaded(
    projection: InputProjection, oracle_param: Callable[[str], np.ndarray]
) -> InputProjection:
    """The same projection carrying layer 0's released weights."""
    projection.load_fused(
        torch.from_numpy(oracle_param("loop_0/TransformerBlock/~/linear::w")),
        torch.from_numpy(oracle_param("loop_0/TransformerBlock/~/linear::b")),
    )
    return projection


def test_submodules_are_named_for_peft(projection: InputProjection) -> None:
    """LoRA targets modules by name, so these four strings are part of the contract.

    Renaming them silently changes which weights an adapter trains.
    """
    assert [name for name, _module in projection.named_children()] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "ff_proj",
    ]


def test_returns_four_tensors_of_the_configured_widths(
    projection: InputProjection,
) -> None:
    """q, k and v come back at the model width; ff at the expansion width."""
    q, k, v, ff = projection(torch.zeros(SEQ_LEN, IN_FEATURES))

    assert q.shape == k.shape == v.shape == (SEQ_LEN, HIDDEN)
    assert ff.shape == (SEQ_LEN, INTERMEDIATE)


def test_leading_dimensions_pass_through(projection: InputProjection) -> None:
    """torch.nn.Linear maps over any leading axes, so a batch costs nothing."""
    q, _k, _v, ff = projection(torch.zeros(3, SEQ_LEN, IN_FEATURES))

    assert q.shape == (3, SEQ_LEN, HIDDEN)
    assert ff.shape == (3, SEQ_LEN, INTERMEDIATE)


def test_load_fused_owns_its_weights(
    projection: InputProjection, oracle_param: Callable[[str], np.ndarray]
) -> None:
    """Copied, not aliased: mutating the checkpoint buffer must not reach into it."""
    weight = torch.from_numpy(oracle_param("loop_0/TransformerBlock/~/linear::w"))
    bias = torch.from_numpy(oracle_param("loop_0/TransformerBlock/~/linear::b"))
    projection.load_fused(weight, bias)
    before = projection.q_proj.weight[0, 0].item()

    weight[0, 0] = -999.0

    assert projection.q_proj.weight[0, 0].item() == before
    assert projection.q_proj.weight.requires_grad


@pytest.mark.parametrize(
    ("weight", "bias"),
    [
        pytest.param(torch.ones(IN_FEATURES + 1, 5376), torch.ones(5376), id="in"),
        pytest.param(torch.ones(IN_FEATURES, 5000), torch.ones(5000), id="out"),
    ],
)
def test_load_fused_rejects_a_mismatched_checkpoint(
    projection: InputProjection, weight: torch.Tensor, bias: torch.Tensor
) -> None:
    """A checkpoint from a differently shaped model must not partially load."""
    with pytest.raises(ValueError):
        projection.load_fused(weight, bias)


def test_matches_jax_oracle(loaded: InputProjection, jax_oracle: NpzFile) -> None:
    """The four outputs, re-concatenated, are the block's fused `middle`.

    Fed the oracle's own `x_with_ages`, so a discrepancy belongs to this module and
    not to the norm or the age concat upstream.
    """
    x = torch.from_numpy(jax_oracle["layer_00_x_with_ages"])

    result = torch.cat(loaded(x), dim=-1)

    torch.testing.assert_close(
        result,
        torch.from_numpy(jax_oracle["layer_00_middle"]),
        rtol=0,
        atol=GEMM_ATOL,
    )


def test_probe_arrays_are_what_the_oracle_dumped(jax_oracle: NpzFile) -> None:
    """Guards the oracle's shape assumptions the test above depends on."""
    assert jax_oracle["layer_00_x_with_ages"].shape == (SEQ_LEN, IN_FEATURES)
    assert jax_oracle["layer_00_middle"].shape == (
        SEQ_LEN,
        3 * HIDDEN + INTERMEDIATE,
    )
