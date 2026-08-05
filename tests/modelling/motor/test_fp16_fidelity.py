"""The end-of-port check: the encoder in the precision real inference runs in.

The rest of the suite develops against the float32 dump, where one float16 ULP is
0.0039 and would give a ~1e-2 noise floor over twelve layers -- deep enough to hide
a real bug. This module is the other half of that bargain: float16 is what the
released model actually runs, so the port has to be checked there once, and nothing
below asserts anything tighter than float16 can represent.

Bounds are stated in ULP at each tensor's own scale rather than as a fixed atol,
because activations grow from ~4 at layer 0 to ~20 at layer 11 and a constant would
be simultaneously too tight at the top and meaningless at the bottom.
"""

import numpy as np
import pytest
import torch
from numpy.lib.npyio import NpzFile

from thesis.modelling.motor.model import MotorEncoder

VOCAB = 65536
HIDDEN = 768
INTERMEDIATE = 3072
N_HEADS = 12
N_LAYERS = 12
ATTENTION_WIDTH = 496
SEQ_LEN = 64

# Measured: every checkpoint lands within 1.5 ULP of the oracle, the largest being
# layer 11 at 0.023 against a 0.0156 ULP. Two leaves headroom.
#
# What this module cannot catch, by construction: the wrong GELU misses layer 0 by
# 4.0e-04, which is a tenth of one float16 ULP there. Substituting the exact erf
# passes every test below and fails sixteen in the float32 suite. That is the whole
# argument for developing against fp32 -- these bounds prove the port survives
# float16, not that it is correct.
ULP_BUDGET = 2.0


def two_ulp(expected: torch.Tensor) -> float:
    """The tolerance for one tensor: ULP_BUDGET ULP at its own largest magnitude.

    Args:
        expected (torch.Tensor): The oracle tensor being compared against.

    Returns:
        float: An absolute tolerance.
    """
    scale = expected.abs().max().to(torch.float32).item()
    return ULP_BUDGET * torch.finfo(torch.float16).eps * scale


@pytest.fixture(scope="session")
def encoder(haiku_params_fp16: dict[str, torch.Tensor]) -> MotorEncoder:
    """The released encoder in inference precision: float16 bar the table."""
    module = MotorEncoder(
        VOCAB, HIDDEN, INTERMEDIATE, N_HEADS, N_LAYERS, ATTENTION_WIDTH
    )
    module.load_haiku(haiku_params_fp16)
    return module.half_stack().eval()


@pytest.fixture(scope="session")
def batch(jax_oracle_fp16: NpzFile) -> dict[str, torch.Tensor | int]:
    """The oracle's batch. The inputs stay float32 -- only parameters were cast."""
    return {
        "indices": torch.from_numpy(
            jax_oracle_fp16["batch_sparse_token_indices"].astype(np.int64)
        ),
        "seq_len": SEQ_LEN,
        "ages": torch.from_numpy(jax_oracle_fp16["batch_ages"]),
        "normed_ages": torch.from_numpy(jax_oracle_fp16["batch_normalized_ages"]),
        "valid_tokens": torch.from_numpy(jax_oracle_fp16["batch_valid_tokens"]),
        "segment_ids": torch.zeros(SEQ_LEN, dtype=torch.long),
    }


@pytest.fixture(scope="session")
def traced(
    encoder: MotorEncoder, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One float16 forward pass with every dumped intermediate captured.

    No rotary substitution here, unlike the float32 stack test: jaxlib's sin error
    of 1.6e-03 is below one ULP at these magnitudes, so it no longer dominates.
    """
    captured: dict[str, torch.Tensor] = {}
    handles = [
        module.register_forward_hook(
            lambda _module, _args, output, key=name: captured.__setitem__(
                key, output.detach().clone()
            )
        )
        for name, module in [("in_norm", encoder.in_norm)]
        + [(f"layer_{index:02d}", block) for index, block in enumerate(encoder.blocks)]
    ]

    try:
        with torch.no_grad():
            final = encoder(**batch)
    finally:
        for handle in handles:
            handle.remove()

    return final, captured


def test_final_features_match_the_oracle(
    traced: tuple[torch.Tensor, dict[str, torch.Tensor]], jax_oracle_fp16: NpzFile
) -> None:
    """The whole port, in the precision it will be served in."""
    final, _captured = traced
    expected = torch.from_numpy(jax_oracle_fp16["final"])

    torch.testing.assert_close(final, expected, rtol=0, atol=two_ulp(expected))


def test_the_oracle_decomposition_is_the_real_model(jax_oracle_fp16: NpzFile) -> None:
    """`final` is bit-identical to the unmodified EHRTransformer's own output.

    Which is what makes the assertion above a comparison against the released model
    rather than against the dump script's reading of it. The float32 dump cannot
    make this claim: its reference is float16 and differs by the rounding the
    float32 dump exists to escape.
    """
    assert np.array_equal(
        jax_oracle_fp16["final"], jax_oracle_fp16["reference_features"]
    )


def test_the_features_are_float16(
    traced: tuple[torch.Tensor, dict[str, torch.Tensor]],
) -> None:
    """A silent promotion to float32 would double the memory and still pass above."""
    final, _captured = traced

    assert final.dtype == torch.float16


def test_the_input_norm_runs_in_float32(
    traced: tuple[torch.Tensor, dict[str, torch.Tensor]], jax_oracle_fp16: NpzFile
) -> None:
    """The norm above a float32 table is a float32 norm, whatever its scale holds.

    haiku casts the scale to the input's dtype; torch would instead compute in the
    weight's, which is why HaikuRMSNorm exists. Letting torch choose costs 8.4e-04
    here, and all twelve blocks inherit it.
    """
    _final, captured = traced

    assert captured["in_norm"].dtype == torch.float32
    torch.testing.assert_close(
        captured["in_norm"],
        torch.from_numpy(jax_oracle_fp16["after_in_norm"]),
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.parametrize("layer", range(N_LAYERS))
def test_every_layer_stays_within_the_float16_floor(
    traced: tuple[torch.Tensor, dict[str, torch.Tensor]],
    jax_oracle_fp16: NpzFile,
    layer: int,
) -> None:
    """Error must stay at the rounding floor rather than compounding down the stack.

    An end-to-end check alone cannot distinguish the two: a stack that drifts and a
    stack that rounds can land at the same final number.
    """
    _final, captured = traced
    expected = torch.from_numpy(jax_oracle_fp16[f"layer_{layer:02d}_output"])
    result = captured[f"layer_{layer:02d}"]

    torch.testing.assert_close(result, expected, rtol=0, atol=two_ulp(expected))
    assert result.dtype == torch.float16


def test_no_activation_overflows(
    traced: tuple[torch.Tensor, dict[str, torch.Tensor]],
) -> None:
    """float16 saturates at 65504, and attention logits are the usual casualty.

    Asserted separately from the tolerances because an inf compares unequal and would
    be reported as a numerical difference rather than as the overflow it is.
    """
    final, captured = traced

    assert torch.isfinite(final).all()
    for name, activation in captured.items():
        assert torch.isfinite(activation).all(), name


def test_half_stack_leaves_the_embedding_table_in_float32(
    encoder: MotorEncoder,
) -> None:
    """The released conversion skips the table, and the port must skip it too."""
    assert encoder.embedding.embeddings.weight.dtype == torch.float32
    assert encoder.compute_dtype == torch.float16


def test_half_then_float_is_not_a_substitute_for_half_stack(
    haiku_params_fp16: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> None:
    """`half()` then `embedding.float()` is lossy, and this is why half_stack exists.

    The round trip rounds the table to float16 and upcasting cannot undo it. An
    embedded row is a sum over ontology ancestors, so every summand is rounded, and
    the damage lands upstream of everything.
    """
    module = MotorEncoder(
        VOCAB, HIDDEN, INTERMEDIATE, N_HEADS, N_LAYERS, ATTENTION_WIDTH
    )
    module.load_haiku(haiku_params_fp16)
    module.half()
    module.embedding.float()

    assert module.embedding.embeddings.weight.dtype == torch.float32
    assert not torch.equal(
        module.embedding.embeddings.weight,
        haiku_params_fp16["embed::embeddings"],
    )


def test_probe_arrays_are_what_the_oracle_dumped(jax_oracle_fp16: NpzFile) -> None:
    """Guards the dump's shape and dtype assumptions the tests above depend on."""
    assert jax_oracle_fp16["final"].dtype == np.float16
    assert jax_oracle_fp16["final"].shape == (SEQ_LEN, HIDDEN)
    assert jax_oracle_fp16["after_in_norm"].dtype == np.float32
    assert (
        jax_oracle_fp16[
            "param::EHRTransformer/~/TransformerFeaturizer/~/Transformer/~/"
            "embed::embeddings"
        ].dtype
        == np.float32
    )
