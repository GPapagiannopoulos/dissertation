"""Tests the rotary rotation against the maths and against the JAX oracle.

Two conventions exist for this operation and both produce well-shaped, finite output:
MOTOR rotates ADJACENT channel pairs (x0,x1), (x2,x3), ... while HuggingFace's
`rotate_half` rotates halves, (x0,x32), (x1,x33), .... Copying the wrong one is silent,
so most of what follows pins the layout rather than the arithmetic.
"""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from numpy.lib.npyio import NpzFile

from thesis.modelling.motor.layers import apply_rotary, rotary_tables

HEAD_DIM = 64
N_HEADS = 12


@pytest.fixture
def ages() -> torch.Tensor:
    """Small ages, so a float64 reference stays comparable in float32.

    Real ages run to 4.2e7 minutes, where one float32 step is already ~4 radians on the
    fastest channel; the disagreement that would cause belongs to `rotary_tables`, which
    tests it, not here.
    """
    return torch.tensor([0.0, 0.5, 1.0, 3.25, 7.5, 10.0])


@pytest.fixture
def tables(ages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The sin/cos pair for `ages` at MOTOR's head width."""
    return rotary_tables(ages, HEAD_DIM)


@pytest.fixture
def make_x() -> Callable[..., torch.Tensor]:
    """Builds a deterministic activation tensor of the requested shape."""

    def _make(*shape: int, seed: int = 0) -> torch.Tensor:
        return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))

    return _make


def rotate_reference(x: torch.Tensor, ages: torch.Tensor, dim: int) -> torch.Tensor:
    """Applies one explicit 2x2 rotation matrix per channel pair, in float64.

    The definition the port compresses into two multiplies, written the slow way:
    pair i of the vector at position p turns by ages[p] * inv_freq[i].
    """
    inv_freq = 1.0 / (10000 ** torch.linspace(0, 2, dim // 2, dtype=torch.float64))
    x = x.double()
    out = torch.empty_like(x)
    for pos, age in enumerate(ages.double()):
        for pair, freq in enumerate(inv_freq):
            theta = age * freq
            first, second = x[..., pos, 2 * pair], x[..., pos, 2 * pair + 1]
            out[..., pos, 2 * pair] = first * theta.cos() - second * theta.sin()
            out[..., pos, 2 * pair + 1] = first * theta.sin() + second * theta.cos()
    return out


def test_matches_explicit_rotation_matrices(
    ages: torch.Tensor,
    tables: tuple[torch.Tensor, torch.Tensor],
    make_x: Callable[..., torch.Tensor],
) -> None:
    """Reproduces a literal per-pair 2x2 rotation.

    This is the test the half-split convention fails: it rotates (x0,x32) rather than
    (x0,x1), which is a different linear map with the same shape.
    """
    x = make_x(N_HEADS, ages.shape[0], HEAD_DIM)

    result = apply_rotary(x, *tables)

    torch.testing.assert_close(
        result, rotate_reference(x, ages, HEAD_DIM).float(), rtol=0, atol=1e-6
    )


def test_zero_ages_are_the_identity(make_x: Callable[..., torch.Tensor]) -> None:
    """Age 0 means angle 0 on every channel, so sin=0, cos=1 and nothing moves.

    Exact equality is the point: x*1 + rotated*0 has no rounding to hide a sign error.
    """
    zeros = torch.zeros(5)
    x = make_x(N_HEADS, 5, HEAD_DIM)

    result = apply_rotary(x, *rotary_tables(zeros, HEAD_DIM))

    assert torch.equal(result, x)


def test_preserves_pairwise_norms(
    ages: torch.Tensor,
    tables: tuple[torch.Tensor, torch.Tensor],
    make_x: Callable[..., torch.Tensor],
) -> None:
    """A rotation is orthogonal, so it turns queries and keys without rescaling them.

    Checked per pair rather than per vector: scaling one pair up and another down by
    the same factor would leave the whole-vector norm intact.
    """
    x = make_x(N_HEADS, ages.shape[0], HEAD_DIM)

    result = apply_rotary(x, *tables)

    before = x.unflatten(-1, (HEAD_DIM // 2, 2)).norm(dim=-1)
    after = result.unflatten(-1, (HEAD_DIM // 2, 2)).norm(dim=-1)
    torch.testing.assert_close(after, before, rtol=0, atol=1e-6)


def test_logits_depend_only_on_the_age_difference(
    ages: torch.Tensor, make_x: Callable[..., torch.Tensor]
) -> None:
    """The reason rotary exists: q.k carries elapsed time, not absolute age.

    R(a)q . R(b)k = q . R(b-a)k, so shifting every age by a constant must leave the
    attention logits untouched. This is the property MOTOR relies on to represent time
    without a timestamp feature.

    The tolerance is loose because the shifted angles are rounded in float32 before the
    difference is taken; the logits themselves are order 1.
    """
    shift = 3.0
    q = make_x(N_HEADS, ages.shape[0], HEAD_DIM, seed=1)
    k = make_x(N_HEADS, ages.shape[0], HEAD_DIM, seed=2)

    def logits(at: torch.Tensor) -> torch.Tensor:
        sin, cos = rotary_tables(at, HEAD_DIM)
        return apply_rotary(q, sin, cos) @ apply_rotary(k, sin, cos).transpose(-1, -2)

    torch.testing.assert_close(logits(ages + shift), logits(ages), rtol=0, atol=1e-4)


def test_absolute_ages_do_move_the_vectors(
    ages: torch.Tensor, make_x: Callable[..., torch.Tensor]
) -> None:
    """Guards the test above from passing on a function that rotates by nothing.

    Invariant logits are only meaningful if the vectors themselves did change.
    """
    x = make_x(N_HEADS, ages.shape[0], HEAD_DIM)

    shifted = apply_rotary(x, *rotary_tables(ages + 3.0, HEAD_DIM))

    assert not torch.allclose(shifted, apply_rotary(x, *rotary_tables(ages, HEAD_DIM)))


def test_heads_are_independent_and_share_one_table(
    ages: torch.Tensor,
    tables: tuple[torch.Tensor, torch.Tensor],
    make_x: Callable[..., torch.Tensor],
) -> None:
    """Broadcasting a (seq, dim) table over (heads, seq, dim) rotates each head alike.

    MOTOR has no per-head frequencies, so batching must equal a loop over heads.
    """
    x = make_x(N_HEADS, ages.shape[0], HEAD_DIM)

    result = apply_rotary(x, *tables)

    per_head = torch.stack([apply_rotary(head, *tables) for head in x])
    torch.testing.assert_close(result, per_head)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((6, HEAD_DIM), id="unbatched"),
        pytest.param((N_HEADS, 6, HEAD_DIM), id="per_head"),
        pytest.param((2, N_HEADS, 6, HEAD_DIM), id="extra_leading_dim"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_shape_and_dtype_are_preserved(
    ages: torch.Tensor,
    make_x: Callable[..., torch.Tensor],
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    """Returns x's own shape and dtype, whatever leads the table's two dimensions."""
    x = make_x(*shape).to(dtype)

    result = apply_rotary(x, *rotary_tables(ages, HEAD_DIM, dtype=dtype))

    assert result.shape == x.shape
    assert result.dtype == dtype


def test_rejects_mismatched_dtypes(
    ages: torch.Tensor, make_x: Callable[..., torch.Tensor]
) -> None:
    """float16 activations against float32 tables must not silently promote.

    torch would return float32 here and widen every tensor downstream of the block,
    which is a precision bug that runs perfectly well.
    """
    x = make_x(N_HEADS, ages.shape[0], HEAD_DIM).half()

    with pytest.raises(ValueError, match="dtypes"):
        apply_rotary(x, *rotary_tables(ages, HEAD_DIM))


@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        pytest.param((N_HEADS, 6, 32), HEAD_DIM, id="wrong_head_dim"),
        pytest.param((N_HEADS, 5, HEAD_DIM), HEAD_DIM, id="wrong_seq_len"),
        pytest.param((HEAD_DIM,), HEAD_DIM, id="one_dimensional"),
    ],
)
def test_rejects_mismatched_shapes(
    ages: torch.Tensor,
    make_x: Callable[..., torch.Tensor],
    shape: tuple[int, ...],
    dim: int,
) -> None:
    """Refuses anything the tables do not describe.

    Broadcasting would otherwise accept a wrong sequence length of 1 and a head width
    that divides cleanly, producing output of the wrong length in silence.
    """
    with pytest.raises(ValueError, match="shapes"):
        apply_rotary(make_x(*shape), *rotary_tables(ages, dim))


def test_matches_jax_oracle(jax_oracle: NpzFile) -> None:
    """Rotates the released model's own probe the way the released model does.

    The oracle's tables are fed in rather than rebuilt, so a discrepancy here is this
    function's and not `rotary_tables`'.
    """
    probe = torch.from_numpy(jax_oracle["rotary_probe"])
    sin = torch.from_numpy(jax_oracle["rotary_sin"])
    cos = torch.from_numpy(jax_oracle["rotary_cos"])

    result = apply_rotary(probe, sin, cos)

    torch.testing.assert_close(
        result, torch.from_numpy(jax_oracle["rotary_probe_rotated"]), rtol=0, atol=1e-6
    )


def test_probe_arrays_are_what_the_oracle_dumped(jax_oracle: NpzFile) -> None:
    """Guards the oracle's shape assumptions the test above depends on."""
    assert jax_oracle["rotary_probe"].dtype == np.float32
    assert jax_oracle["rotary_probe"].shape == (N_HEADS, 64, HEAD_DIM)
    assert jax_oracle["rotary_probe_rotated"].shape == (N_HEADS, 64, HEAD_DIM)


def test_a_batch_rotates_each_sequence_by_its_own_table() -> None:
    """Batched queries meet batched tables, one clock per subject.

    Queries arrive at (batch, heads, seq, dim) and the tables at
    (batch, 1, seq, dim): the heads share a clock, the subjects do not.
    """
    batch, seq = 2, 5
    ages = torch.stack((torch.arange(seq).float(), torch.arange(seq).float() * 3))
    sin, cos = rotary_tables(ages, HEAD_DIM)
    x = torch.randn(batch, N_HEADS, seq, HEAD_DIM)

    rotated = apply_rotary(x, sin.unsqueeze(-3), cos.unsqueeze(-3))

    for row in range(batch):
        alone = apply_rotary(x[row], sin[row], cos[row])
        torch.testing.assert_close(rotated[row], alone, rtol=0, atol=0)


def test_a_batched_table_must_carry_its_head_axis() -> None:
    """The trap this guard exists for, and the only shape where it bites.

    A (batch, seq, dim) table against (batch, heads, seq, dim) queries aligns from
    the right, so it broadcasts silently whenever batch happens to equal the head
    count -- rotating every head by a different subject's clock while returning a
    correctly shaped, finite tensor.
    """
    sin, cos = rotary_tables(torch.ones(N_HEADS, 5), HEAD_DIM)
    x = torch.randn(N_HEADS, N_HEADS, 5, HEAD_DIM)

    with pytest.raises(ValueError, match="carry every axis"):
        apply_rotary(x, sin, cos)


def test_a_shared_table_still_broadcasts_over_a_batch() -> None:
    """One sequence's tables may serve a whole batch, which is the B=1 case."""
    sin, cos = rotary_tables(torch.arange(5).float(), HEAD_DIM)
    x = torch.randn(3, N_HEADS, 5, HEAD_DIM)

    rotated = apply_rotary(x, sin, cos)

    for row in range(3):
        torch.testing.assert_close(
            rotated[row], apply_rotary(x[row], sin, cos), rtol=0, atol=0
        )
