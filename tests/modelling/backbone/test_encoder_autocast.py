"""Tests that the encoder runs under `torch.autocast`, which training needs.

Full fine-tuning wants bf16 matmuls for speed but fp32 parameters, because bf16 keeps
about eight mantissa bits and an AdamW update at lr 1e-5 is smaller than that
resolution -- in pure bf16 most of every update rounds straight back onto the weight
it came from. `torch.autocast` is what buys both, so the stack has to survive it.

These use a small random encoder rather than the checkpoint: what is under test is the
plumbing of dtypes through the stack, which does not depend on the weights, and the
oracle fixtures skip on a clean checkout.
"""

import pytest
import torch

from thesis.modelling.backbone.model import MotorEncoder

HIDDEN = 32
N_HEADS = 4
SEQ_LEN = 8
BATCH = 2

# bf16 carries ~8 mantissa bits, so ~4e-3 relative; over four small blocks the
# accumulated difference from the fp32 path lands well inside this.
BF16_ATOL = 5e-2


@pytest.fixture
def encoder() -> MotorEncoder:
    """A small stack at the real proportions, in float32."""
    torch.manual_seed(0)
    return MotorEncoder(
        vocab_size=64,
        hidden_size=HIDDEN,
        intermediate_size=4 * HIDDEN,
        n_heads=N_HEADS,
        n_layers=4,
        attention_width=4,
    )


@pytest.fixture
def batch() -> dict[str, torch.Tensor | int]:
    """One batch of two sequences, every position holding one token."""
    torch.manual_seed(1)
    n_positions = BATCH * SEQ_LEN
    ages = torch.rand(BATCH, SEQ_LEN).cumsum(-1) * 100.0
    return {
        "indices": torch.stack(
            [torch.randint(0, 64, (n_positions,)), torch.arange(n_positions)], dim=1
        ),
        "seq_len": SEQ_LEN,
        "ages": ages,
        "normed_ages": (ages - 13540.859) / 9335.633,
        "valid_tokens": torch.ones(BATCH, SEQ_LEN, dtype=torch.bool),
        "segment_ids": torch.zeros(BATCH, SEQ_LEN, dtype=torch.long),
    }


@pytest.mark.parametrize("cast_to", [torch.bfloat16, torch.float16])
def test_runs_under_autocast(
    encoder: MotorEncoder, batch: dict[str, torch.Tensor | int], cast_to: torch.dtype
) -> None:
    """The rotary tables meet the queries, so `apply_rotary`'s guard is satisfied.

    Before the tables followed the autocast dtype this raised ValueError from
    `apply_rotary`, which is the regression this test pins.
    """
    with torch.autocast("cpu", dtype=cast_to):
        features = encoder(**batch)

    assert features.shape == (BATCH, SEQ_LEN, HIDDEN)
    assert torch.isfinite(features).all()


def test_residual_stream_stays_float32(
    encoder: MotorEncoder, batch: dict[str, torch.Tensor | int]
) -> None:
    """Each block adds a bf16 delta onto an fp32 stream, which promotes back to fp32.

    This is what keeps `MotorBlock`'s age-dtype guard satisfied at every block rather
    than only the first, so it is a property of the design and not an accident.
    """
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert encoder(**batch).dtype is torch.float32


def test_agrees_with_the_float32_path(
    encoder: MotorEncoder, batch: dict[str, torch.Tensor | int]
) -> None:
    """Autocast is a precision change, not a semantic one."""
    expected = encoder(**batch)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = encoder(**batch)

    torch.testing.assert_close(actual, expected, rtol=0, atol=BF16_ATOL)


def test_parameters_are_untouched_by_autocast(
    encoder: MotorEncoder, batch: dict[str, torch.Tensor | int]
) -> None:
    """The point of autocast over `half_stack`: the master weights stay float32."""
    with torch.autocast("cpu", dtype=torch.bfloat16):
        encoder(**batch)

    assert all(parameter.dtype is torch.float32 for parameter in encoder.parameters())


def test_gradients_reach_the_parameters_in_float32(
    encoder: MotorEncoder, batch: dict[str, torch.Tensor | int]
) -> None:
    """Autocast casts activations, not parameters, so AdamW still sees fp32 grads."""
    with torch.autocast("cpu", dtype=torch.bfloat16):
        encoder(**batch).sum().backward()

    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(grad.dtype is torch.float32 for grad in grads)
