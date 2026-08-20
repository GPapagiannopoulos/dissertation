"""Tests the helper that tells the rotary tables which dtype the queries will be.

The function exists because `apply_rotary` refuses a table whose dtype differs from
its input, and under `torch.autocast` a Linear emits the autocast dtype rather than
the dtype of the stream that entered it. That guard is load bearing -- it is what
catches a batched table silently rotating by another sequence's clock -- so the fix
is to build the tables in the right dtype, not to relax the guard.

Autocast is queried per device, and these tests run the CPU one because the GPU may
not be present. The helper reads whichever device the tensor it is handed sits on,
so the behaviour is the same either way.
"""

import pytest
import torch

from thesis.modelling.backbone.layers import projected_dtype

DTYPES = [torch.float32, torch.float64, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
def test_returns_the_input_dtype_outside_autocast(dtype: torch.dtype) -> None:
    """Without autocast a Linear returns what it was given, and so does this."""
    assert projected_dtype(torch.zeros(2, 2, dtype=dtype)) is dtype


@pytest.mark.parametrize("cast_to", [torch.bfloat16, torch.float16])
def test_returns_the_autocast_dtype_inside_autocast(cast_to: torch.dtype) -> None:
    """Inside autocast the answer is the autocast dtype, not the stream's."""
    x = torch.zeros(2, 2, dtype=torch.float32)

    with torch.autocast("cpu", dtype=cast_to):
        assert projected_dtype(x) is cast_to


@pytest.mark.parametrize("cast_to", [torch.bfloat16, torch.float16])
def test_agrees_with_what_a_linear_actually_emits(cast_to: torch.dtype) -> None:
    """The contract, asserted against the thing it is predicting rather than restated.

    A test that only compared the helper against `torch.get_autocast_dtype` would
    pass even if torch changed which ops autocast covers.
    """
    x = torch.zeros(3, 8, dtype=torch.float32)
    linear = torch.nn.Linear(8, 8)

    with torch.autocast("cpu", dtype=cast_to):
        assert projected_dtype(x) is linear(x).dtype

    assert projected_dtype(x) is linear(x).dtype


def test_reads_the_device_of_the_tensor_it_is_given() -> None:
    """A CPU tensor is unaffected by autocast enabled on another device."""
    x = torch.zeros(2, 2, dtype=torch.float32)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert projected_dtype(x) is torch.float32


def test_is_not_fooled_by_a_disabled_autocast_region() -> None:
    """`enabled=False` is how the head opts out, so it must report the real dtype."""
    x = torch.zeros(2, 2, dtype=torch.float32)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=False):
            assert projected_dtype(x) is torch.float32
