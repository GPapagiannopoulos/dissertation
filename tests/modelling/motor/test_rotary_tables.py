"""Tests the rotary sin/cos tables against the maths and against the JAX oracle.

The oracle test is deliberately loose. jaxlib's CPU sin reduces huge arguments sloppily.
Measured against a float64 reference at MOTOR's real ages, torch errs by 6e-08 and JAX
by 1.6e-03. Torch is the accurate one, so the tables cannot agree to better than JAX's
own error.
"""

import numpy as np
import pytest
import torch
from numpy.lib.npyio import NpzFile

from thesis.modelling.motor.layers import rotary_tables

HEAD_DIM = 64

# jaxlib 0.4.7's CPU sin errs by up to 1.6e-03 on arguments the size of an age in
# minutes; anything tighter would fail on a correct implementation.
LIBM_ATOL = 5e-3


@pytest.fixture
def small_ages() -> torch.Tensor:
    """Ages small enough that fp32 and fp64 angles agree.

    Real ages run to 4.2e7 minutes, where one fp32 step is already ~4 radians, so a
    float64 reference computed end to end would disagree for reasons that have nothing
    to do with this function. Keeping the angles small isolates the schedule itself.
    """
    return torch.tensor([0.0, 0.5, 1.0, 3.25, 7.5, 10.0])


def test_matches_float64_reference(small_ages: torch.Tensor) -> None:
    """Reproduces the definition computed end to end in double precision."""
    inv_freq = 1.0 / (10000 ** torch.linspace(0, 2, HEAD_DIM // 2, dtype=torch.float64))
    angles = small_ages.double().unsqueeze(1) * inv_freq.unsqueeze(0)

    sin, cos = rotary_tables(small_ages, HEAD_DIM)

    torch.testing.assert_close(
        sin, angles.sin().repeat_interleave(2, dim=-1).float(), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        cos, angles.cos().repeat_interleave(2, dim=-1).float(), rtol=0, atol=1e-6
    )


def test_frequency_schedule_spans_zero_to_two(small_ages: torch.Tensor) -> None:
    """Pins the exponent range: the endpoints are 10000**0 and 10000**2.

    Ages reach the model in days, so the first channel pair turns once per 6.3 days
    and the last once per 172 years -- a lifetime without wraparound. Standard RoPE
    floors at 1e-4, whose slowest clock wraps every 172 days, and would fail here.
    """
    sin, cos = rotary_tables(small_ages, HEAD_DIM)

    assert torch.equal(sin[:, 0], small_ages.sin())
    assert torch.equal(cos[:, 0], small_ages.cos())
    torch.testing.assert_close(sin[:, -1], (small_ages * 1e-8).sin(), rtol=0, atol=1e-6)
    torch.testing.assert_close(cos[:, -1], (small_ages * 1e-8).cos(), rtol=0, atol=1e-6)


def test_tables_are_pairwise_interleaved(small_ages: torch.Tensor) -> None:
    """Each angle occupies two ADJACENT channels, because the rotation is 2-D.

    Rotating the pair (a, b) uses one theta for both components, so the tables must
    read [t0, t0, t1, t1, ...].
    """
    sin, cos = rotary_tables(small_ages, HEAD_DIM)

    assert torch.equal(sin[:, 0::2], sin[:, 1::2])
    assert torch.equal(cos[:, 0::2], cos[:, 1::2])
    # and the halves are NOT equal, which is what the wrong layout would produce
    assert not torch.equal(sin[:, : HEAD_DIM // 2], sin[:, HEAD_DIM // 2 :])


@pytest.mark.parametrize("dim", [16, 64, 768])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_shape_and_dtype(
    small_ages: torch.Tensor, dim: int, dtype: torch.dtype
) -> None:
    """Returns (seq_len, dim) in the requested dtype, whatever the head width."""
    sin, cos = rotary_tables(small_ages, dim, dtype=dtype)

    assert sin.shape == cos.shape == (small_ages.shape[0], dim)
    assert sin.dtype == cos.dtype == dtype


@pytest.mark.parametrize(
    "ages",
    [
        pytest.param(torch.tensor([1.0, 2.0], dtype=torch.float16), id="fp16"),
        pytest.param(torch.tensor([1.0, 2.0], dtype=torch.float64), id="fp64"),
        pytest.param(torch.ones(2, 3, 4), id="three_dimensional"),
    ],
)
def test_rejects_malformed_ages(ages: torch.Tensor) -> None:
    """Refuses anything but a float32 tensor of one or two dimensions.

    fp16's largest value is 65,504, so every age past 45 days overflows to inf
    and the tables come back as nan.
    """
    with pytest.raises(ValueError):
        rotary_tables(ages, HEAD_DIM)


def test_matches_jax_oracle(jax_oracle: NpzFile) -> None:
    """Agrees with the released model's own tables to within JAX's sin error."""
    ages = torch.from_numpy(jax_oracle["batch_ages"])
    sin, cos = rotary_tables(ages, HEAD_DIM)

    torch.testing.assert_close(
        sin, torch.from_numpy(jax_oracle["rotary_sin"]), rtol=0, atol=LIBM_ATOL
    )
    torch.testing.assert_close(
        cos, torch.from_numpy(jax_oracle["rotary_cos"]), rtol=0, atol=LIBM_ATOL
    )


def test_torch_is_more_accurate_than_the_oracle(jax_oracle: NpzFile) -> None:
    """Documents why the oracle bound above is loose.

    If a future torch drops accuracy on large arguments, the loose bound stops being
    justified and this fails.
    """
    ages = torch.from_numpy(jax_oracle["batch_ages"])
    sin, _cos = rotary_tables(ages, HEAD_DIM)

    inv_freq = 1.0 / (10000 ** torch.linspace(0, 2, HEAD_DIM // 2))
    angles = ages.unsqueeze(1) * inv_freq.unsqueeze(0)
    exact = torch.sin(angles.double()).float().repeat_interleave(2, dim=-1)

    torch_error = (sin - exact).abs().max()
    jax_error = (torch.from_numpy(jax_oracle["rotary_sin"]) - exact).abs().max()

    assert torch_error < 1e-6
    assert jax_error > torch_error * 100


def test_reference_arrays_are_what_the_oracle_dumped(jax_oracle: NpzFile) -> None:
    """Guards the oracle's own shape assumptions the tests above depend on."""
    assert jax_oracle["batch_ages"].dtype == np.float32
    assert jax_oracle["rotary_sin"].shape == (64, HEAD_DIM)
    assert jax_oracle["rotary_cos"].shape == (64, HEAD_DIM)


def test_a_batch_builds_one_table_per_sequence() -> None:
    """Every sequence carries its own ages, so the tables cannot be shared.

    A table computed from the batch as a whole -- flattened, or taken from row zero
    -- still has the right shape and finite values, and rotates most positions by
    another subject's clock.
    """
    first = torch.tensor([0.0, 10.0, 100.0])
    second = torch.tensor([5.0, 50.0, 5000.0])

    sin, cos = rotary_tables(torch.stack((first, second)), HEAD_DIM)

    for row, ages in enumerate((first, second)):
        alone = rotary_tables(ages, HEAD_DIM)
        torch.testing.assert_close(sin[row], alone[0], rtol=0, atol=0)
        torch.testing.assert_close(cos[row], alone[1], rtol=0, atol=0)


def test_a_batch_keeps_the_sequence_and_head_axes_in_order() -> None:
    """The batch axis is prepended, not folded into the sequence.

    Both layouts hold the same numbers, so only the shape distinguishes them.
    """
    sin, cos = rotary_tables(torch.ones(4, 7), HEAD_DIM)

    assert sin.shape == cos.shape == (4, 7, HEAD_DIM)
