"""Tests one whole block against layer 0 of the released model.

This is the port's first end-to-end assertion: the comparison runs the norm, the age
concat, all four projections, the head split, both rotary applications, masked
attention, the merge, the GELU and the output projection, and it is fed nothing but
the oracle's own layer-0 inputs.
"""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from numpy.lib.npyio import NpzFile

from thesis.modelling.motor.constants import MOTOR_AGE_FEATURES
from thesis.modelling.motor.layers import MotorBlock, local_attention_mask

HIDDEN = 768
INTERMEDIATE = 3072
N_HEADS = 12
SEQ_LEN = 64
ATTENTION_WIDTH = 496

# Chained fp32 matmuls, accumulated in a different order by JAX and torch. Measured at
# 9.5e-07 against outputs reaching 4.2, so this leaves 10x headroom -- and the wrong
# GELU misses by 4.0e-04, forty times the bound.
BLOCK_ATOL = 1e-5


@pytest.fixture
def block() -> MotorBlock:
    """A block at the released model's widths, still randomly initialised."""
    return MotorBlock(HIDDEN, INTERMEDIATE, N_HEADS)


@pytest.fixture
def loaded(block: MotorBlock, oracle_param: Callable[[str], np.ndarray]) -> MotorBlock:
    """The same block carrying layer 0's released parameters."""
    prefix = "loop_0/TransformerBlock/~/"
    block.load_haiku(
        *(
            torch.from_numpy(oracle_param(prefix + name))
            for name in (
                "rms_norm::scale",
                "linear::w",
                "linear::b",
                "linear_1::w",
                "linear_1::b",
            )
        )
    )
    return block


@pytest.fixture
def block_inputs(jax_oracle: NpzFile) -> dict[str, torch.Tensor]:
    """Layer 0's real inputs: the stream, the ages and the rotary tables.

    The mask is rebuilt rather than dumped: at width 496 over 64 positions in one
    packed sequence it is plain causal, which is exactly the degeneracy the separate
    attention probe exists to cover.
    """
    return {
        "x": torch.from_numpy(jax_oracle["layer_00_input"]),
        "normed_ages": torch.from_numpy(jax_oracle["batch_normalized_ages"]),
        "sin": torch.from_numpy(jax_oracle["rotary_sin"]),
        "cos": torch.from_numpy(jax_oracle["rotary_cos"]),
        "mask": local_attention_mask(
            torch.zeros(SEQ_LEN, dtype=torch.long), ATTENTION_WIDTH
        ),
    }


def test_matches_jax_oracle(
    loaded: MotorBlock, block_inputs: dict[str, torch.Tensor], jax_oracle: NpzFile
) -> None:
    """Reproduces layer 0's output, residual included."""
    result = loaded(**block_inputs)

    torch.testing.assert_close(
        result,
        torch.from_numpy(jax_oracle["layer_00_output"]),
        rtol=0,
        atol=BLOCK_ATOL,
    )


def test_returns_the_residual_added(
    loaded: MotorBlock, block_inputs: dict[str, torch.Tensor], jax_oracle: NpzFile
) -> None:
    """Femr's caller adds the residual under lax.scan; the port's block does it itself.

    Pinning both halves separately means a sign or a doubled residual cannot hide
    inside the total.
    """
    result = loaded(**block_inputs)

    delta = result - block_inputs["x"]
    torch.testing.assert_close(
        delta, torch.from_numpy(jax_oracle["layer_00_delta"]), rtol=0, atol=BLOCK_ATOL
    )


def test_uses_the_tanh_gelu(
    loaded: MotorBlock,
    block_inputs: dict[str, torch.Tensor],
    jax_oracle: NpzFile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """jax.nn.gelu defaults to the tanh approximation; torch's default is the erf.

    Forcing the exact variant is the only mutation this file cannot express as a
    wrong argument, and it is the one most likely to be introduced by someone
    "simplifying" the call. It misses by 400x the correct block's error.
    """
    expected = torch.from_numpy(jax_oracle["layer_00_output"])
    tanh_error = (loaded(**block_inputs) - expected).abs().max()

    exact = torch.nn.functional.gelu
    monkeypatch.setattr(
        torch.nn.functional, "gelu", lambda x, approximate="none": exact(x)
    )
    erf_error = (loaded(**block_inputs) - expected).abs().max()

    assert tanh_error < BLOCK_ATOL
    assert erf_error > tanh_error * 100


def test_age_columns_reach_the_projection(
    loaded: MotorBlock, block_inputs: dict[str, torch.Tensor]
) -> None:
    """The ages are concatenated, not ignored, so changing them changes the output."""
    baseline = loaded(**block_inputs)

    moved = dict(block_inputs, normed_ages=block_inputs["normed_ages"] + 1.0)

    assert not torch.allclose(loaded(**moved), baseline)


def test_the_projection_is_two_columns_wider_than_the_model(block: MotorBlock) -> None:
    """The age and its square are what widen it, and the checkpoint agrees."""
    assert block.input_proj.q_proj.in_features == HIDDEN + MOTOR_AGE_FEATURES


def test_rejects_ages_of_another_dtype(
    block: MotorBlock, block_inputs: dict[str, torch.Tensor]
) -> None:
    """fp32 ages against an fp16 stream would promote the whole block silently."""
    inputs = dict(block_inputs, normed_ages=block_inputs["normed_ages"].double())

    with pytest.raises(ValueError, match="dtype"):
        block(**inputs)


def test_load_haiku_rejects_a_mismatched_scale(block: MotorBlock) -> None:
    """A norm scale from another width must not partially load."""
    with pytest.raises(ValueError, match="scale"):
        block.load_haiku(
            torch.zeros(HIDDEN + 1),
            torch.zeros(HIDDEN + MOTOR_AGE_FEATURES, 3 * HIDDEN + INTERMEDIATE),
            torch.zeros(3 * HIDDEN + INTERMEDIATE),
            torch.zeros(HIDDEN + INTERMEDIATE, HIDDEN),
            torch.zeros(HIDDEN),
        )


def test_probe_arrays_are_what_the_oracle_dumped(jax_oracle: NpzFile) -> None:
    """Guards the oracle's shape assumptions the tests above depend on."""
    assert jax_oracle["layer_00_input"].shape == (SEQ_LEN, HIDDEN)
    assert jax_oracle["layer_00_output"].shape == (SEQ_LEN, HIDDEN)
    assert jax_oracle["batch_normalized_ages"].shape == (SEQ_LEN,)
