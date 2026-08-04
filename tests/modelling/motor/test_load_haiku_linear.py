"""Tests the haiku-to-torch linear loader against the real output projection."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from thesis.modelling.motor.layers import load_haiku_linear

HIDDEN = 768
COMBINED = 3840  # hidden_size + intermediate_size, the output projection's input


@pytest.fixture
def module() -> torch.nn.Linear:
    """A small linear whose in and out widths differ, so an orientation error shows."""
    return torch.nn.Linear(5, 3)


def test_transposes_into_torch_layout(module: torch.nn.Linear) -> None:
    """Haiku's (in, out) becomes torch's (out, in)."""
    weight = torch.arange(15, dtype=torch.float32).reshape(5, 3)

    load_haiku_linear(module, weight, torch.zeros(3))

    assert torch.equal(module.weight, weight.T)


def test_computes_what_haiku_computed(module: torch.nn.Linear) -> None:
    """The whole point of the transpose: x @ w + b must survive it."""
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(5, 3, generator=generator)
    bias = torch.randn(3, generator=generator)
    x = torch.randn(7, 5, generator=generator)

    load_haiku_linear(module, weight, bias)

    torch.testing.assert_close(module(x), x @ weight + bias, rtol=0, atol=1e-6)


def test_owns_its_parameters(module: torch.nn.Linear) -> None:
    """Copied, not aliased, so the checkpoint buffer can be freed or reused."""
    weight = torch.ones(5, 3)
    load_haiku_linear(module, weight, torch.zeros(3))

    weight[0, 0] = -999.0

    assert module.weight[0, 0] == 1.0
    assert module.weight.is_contiguous()


@pytest.mark.parametrize(
    ("weight", "bias", "match"),
    [
        pytest.param(torch.zeros(3, 5), torch.zeros(3), "haiku", id="already_torch"),
        pytest.param(torch.zeros(5, 4), torch.zeros(4), "haiku", id="wrong_out"),
        pytest.param(torch.zeros(5, 3), torch.zeros(4), "bias", id="wrong_bias"),
    ],
)
def test_rejects_a_mismatched_checkpoint(
    module: torch.nn.Linear, weight: torch.Tensor, bias: torch.Tensor, match: str
) -> None:
    """A pre-transposed weight is the likely mistake, and it must not load."""
    with pytest.raises(ValueError, match=match):
        load_haiku_linear(module, weight, bias)


def test_loads_the_real_output_projection(
    oracle_param: Callable[[str], np.ndarray],
) -> None:
    """The released (3840, 768) `linear_1` fits a torch Linear the other way round."""
    module = torch.nn.Linear(COMBINED, HIDDEN)

    load_haiku_linear(
        module,
        torch.from_numpy(oracle_param("loop_0/TransformerBlock/~/linear_1::w")),
        torch.from_numpy(oracle_param("loop_0/TransformerBlock/~/linear_1::b")),
    )

    assert module.weight.shape == (HIDDEN, COMBINED)
