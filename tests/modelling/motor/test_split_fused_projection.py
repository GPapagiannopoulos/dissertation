"""Tests the fused-weight split against the maths and the real checkpoint.

The split has to be exactly reversible: every number in the checkpoint must land in
one slice, in the same order, transposed into torch's layout.
"""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from thesis.modelling.motor.layers import split_fused_projection

HIDDEN = 768
INTERMEDIATE = 3072
IN_FEATURES = 770

# a small stand-in for the real (770, 5376): 3 * 2 + 4 output columns
SMALL_HIDDEN = 2
SMALL_IN = 5
SMALL_OUT = 3 * SMALL_HIDDEN + 4


@pytest.fixture
def fused() -> tuple[torch.Tensor, torch.Tensor]:
    """A fused weight whose every entry is distinct, so misplaced slices show up."""
    weight = torch.arange(SMALL_IN * SMALL_OUT, dtype=torch.float32).reshape(
        SMALL_IN, SMALL_OUT
    )
    return weight, torch.arange(SMALL_OUT, dtype=torch.float32)


def test_slices_are_transposed_into_torch_layout(
    fused: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Haiku stores (in, out); torch.nn.Linear stores (out, in)."""
    parts = split_fused_projection(*fused, SMALL_HIDDEN)

    assert parts["q_proj"][0].shape == (SMALL_HIDDEN, SMALL_IN)
    assert parts["ff_proj"][0].shape == (SMALL_OUT - 3 * SMALL_HIDDEN, SMALL_IN)


@pytest.mark.parametrize(
    ("name", "low", "high"),
    [
        ("q_proj", 0, SMALL_HIDDEN),
        ("k_proj", SMALL_HIDDEN, 2 * SMALL_HIDDEN),
        ("v_proj", 2 * SMALL_HIDDEN, 3 * SMALL_HIDDEN),
        ("ff_proj", 3 * SMALL_HIDDEN, SMALL_OUT),
    ],
)
def test_each_slice_reads_its_own_columns(
    fused: tuple[torch.Tensor, torch.Tensor], name: str, low: int, high: int
) -> None:
    """Pins the boundaries: q, k, v in that order, then the feed-forward remainder."""
    weight, bias = fused

    part_weight, part_bias = split_fused_projection(weight, bias, SMALL_HIDDEN)[name]

    assert torch.equal(part_weight, weight[:, low:high].T)
    assert torch.equal(part_bias, bias[low:high])


def test_reconcatenating_recovers_the_checkpoint(
    fused: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Nothing is dropped, duplicated or reordered.

    Fails on any off-by-one boundary, which slicing tests alone can miss when two
    adjacent slices are the same width.
    """
    weight, bias = fused
    parts = split_fused_projection(weight, bias, SMALL_HIDDEN)

    order = ["q_proj", "k_proj", "v_proj", "ff_proj"]
    assert torch.equal(torch.cat([parts[n][0].T for n in order], dim=1), weight)
    assert torch.equal(torch.cat([parts[n][1] for n in order]), bias)


def test_slices_own_their_memory(fused: tuple[torch.Tensor, torch.Tensor]) -> None:
    """A view would alias the checkpoint buffer and carry a transposed stride."""
    weight, bias = fused
    parts = split_fused_projection(weight, bias, SMALL_HIDDEN)

    weight[0, 0] = -999.0
    bias[0] = -999.0

    assert parts["q_proj"][0].is_contiguous()
    assert parts["q_proj"][0][0, 0] != -999.0
    assert parts["q_proj"][1][0] != -999.0


@pytest.mark.parametrize(
    ("weight", "bias", "hidden", "match"),
    [
        pytest.param(torch.ones(6), torch.ones(6), 1, "two dimensional", id="1d"),
        pytest.param(torch.ones(5, 10), torch.ones(9), 2, "bias", id="bias_width"),
        pytest.param(torch.ones(5, 6), torch.ones(6), 2, "wider", id="no_ff_columns"),
        pytest.param(torch.ones(5, 4), torch.ones(4), 2, "wider", id="too_narrow"),
    ],
)
def test_rejects_malformed_checkpoints(
    weight: torch.Tensor, bias: torch.Tensor, hidden: int, match: str
) -> None:
    """A wrong hidden_size would otherwise return an empty feed-forward slice."""
    with pytest.raises(ValueError, match=match):
        split_fused_projection(weight, bias, hidden)


def test_splits_the_real_checkpoint(
    oracle_param: Callable[[str], np.ndarray],
) -> None:
    """The released (770, 5376) weight yields exactly MOTOR's four projections."""
    weight = torch.from_numpy(oracle_param("loop_0/TransformerBlock/~/linear::w"))
    bias = torch.from_numpy(oracle_param("loop_0/TransformerBlock/~/linear::b"))

    parts = split_fused_projection(weight, bias, HIDDEN)

    assert {name: tuple(w.shape) for name, (w, _b) in parts.items()} == {
        "q_proj": (HIDDEN, IN_FEATURES),
        "k_proj": (HIDDEN, IN_FEATURES),
        "v_proj": (HIDDEN, IN_FEATURES),
        "ff_proj": (INTERMEDIATE, IN_FEATURES),
    }
