"""Tests the assembled encoder against every checkpoint the oracle dumps.

The stack is asserted layer by layer rather than only end to end. A single
end-to-end comparison would tell you that something drifted over twelve blocks
without telling you which one, and the blocks hold twelve independent sets of
weights, so loading them out of order is exactly the kind of bug that produces a
plausible-looking number.

The traced forward substitutes the oracle's own rotary tables for the ones the
encoder computes due to a known failure of jax's sin calculation. Substituting it
away leaves the assertions measuring the stack. The encoder's own tables are then
checked separately, end to end, at jaxlib's error.

The heavy fixtures are session scoped: the encoder is 135 M parameters, and the
per-layer assertions all read one traced forward rather than running twelve.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from numpy.lib.npyio import NpzFile

from thesis.modelling.motor import model as motor_model
from thesis.modelling.motor.model import MotorEncoder

VOCAB = 65536
HIDDEN = 768
INTERMEDIATE = 3072
N_HEADS = 12
N_LAYERS = 12
ATTENTION_WIDTH = 496
SEQ_LEN = 64

# On the oracle's own rotary tables: one block reproduces layer 0 at 9.5e-07, and
# twelve accumulate that GEMM ordering difference to 8.6e-06 by layer 11. The wrong
# GELU misses layer 0 by 4.0e-04, twenty times this bound.
STACK_ATOL = 2e-05

# End to end on the encoder's own rotary tables. jaxlib's sin errs by 1.6e-03 at
# MOTOR's real ages, which reaches 3.0e-04 at the final norm; this bounds the wiring,
# not the arithmetic.
LIBM_ATOL = 5e-03


@pytest.fixture(scope="session")
def encoder(haiku_params: dict[str, torch.Tensor]) -> MotorEncoder:
    """The released encoder, in float32."""
    module = MotorEncoder(
        VOCAB, HIDDEN, INTERMEDIATE, N_HEADS, N_LAYERS, ATTENTION_WIDTH
    )
    module.load_haiku(haiku_params)
    return module.eval()


@pytest.fixture(scope="session")
def batch(jax_oracle: NpzFile) -> dict[str, torch.Tensor | int]:
    """The oracle's synthetic batch as forward's keyword arguments.

    `segment_ids` is all zeros because the dump packs one subject.

    femr's bitmask ~(64 - 1) puts all 64 positions in one segment.
    Segmentation is exercised by the attention probe instead, where
    it actually binds.
    """
    return {
        "indices": torch.from_numpy(
            jax_oracle["batch_sparse_token_indices"].astype(np.int64)
        ),
        "seq_len": SEQ_LEN,
        "ages": torch.from_numpy(jax_oracle["batch_ages"]),
        "normed_ages": torch.from_numpy(jax_oracle["batch_normalized_ages"]),
        "valid_tokens": torch.from_numpy(jax_oracle["batch_valid_tokens"]),
        "segment_ids": torch.zeros(SEQ_LEN, dtype=torch.long),
    }


@contextmanager
def jax_rotary_tables(jax_oracle: NpzFile) -> Iterator[None]:
    """Substitutes the oracle's rotary tables for the ones the encoder computes.

    A context manager rather than a fixture, so the substitution cannot outlive the
    one forward pass that wants it and silently reach the unpatched test below.
    """
    tables = (
        torch.from_numpy(jax_oracle["rotary_sin"]),
        torch.from_numpy(jax_oracle["rotary_cos"]),
    )
    original = motor_model.rotary_tables
    motor_model.rotary_tables = lambda *_args, **_kwargs: tables
    try:
        yield
    finally:
        motor_model.rotary_tables = original


@pytest.fixture(scope="session")
def traced(
    encoder: MotorEncoder, batch: dict[str, torch.Tensor], jax_oracle: NpzFile
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One forward pass, with every intermediate the oracle also dumped captured.

    Hooks rather than a `return_hidden_states` flag, so the assertions cost the
    module nothing: an inference API that carries a debugging switch invites callers
    to depend on it.
    """
    captured: dict[str, torch.Tensor] = {}

    def record(name: str) -> Callable[..., None]:
        def _hook(_module: torch.nn.Module, _args: tuple, output: object) -> None:
            assert isinstance(output, torch.Tensor)
            captured[name] = output.detach().clone()

        return _hook

    handles = [encoder.in_norm.register_forward_hook(record("in_norm"))]
    handles += [
        block.register_forward_hook(record(f"layer_{index:02d}"))
        for index, block in enumerate(encoder.blocks)
    ]

    try:
        with jax_rotary_tables(jax_oracle), torch.no_grad():
            final = encoder(**batch)
    finally:
        for handle in handles:
            handle.remove()

    return final, captured


@pytest.fixture
def tiny() -> MotorEncoder:
    """A two-layer encoder at toy widths, for the guards and the fill rule."""
    return MotorEncoder(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        n_heads=2,
        n_layers=2,
        attention_width=4,
    )


@pytest.fixture
def tiny_batch() -> dict[str, torch.Tensor | int]:
    """A three-position batch matching `tiny`."""
    return {
        "indices": torch.tensor([[0, 0], [3, 1], [5, 2]]),
        "seq_len": 3,
        "ages": torch.tensor([0.0, 60.0, 4800.0]),
        "normed_ages": torch.tensor([-1.0, 0.0, 1.0]),
        "valid_tokens": torch.ones(3, dtype=torch.bool),
        "segment_ids": torch.zeros(3, dtype=torch.long),
    }


def test_matches_jax_oracle(
    traced: tuple[torch.Tensor, dict[str, torch.Tensor]], jax_oracle: NpzFile
) -> None:
    """The features the encoder returns are the oracle's `final`."""
    final, _captured = traced

    torch.testing.assert_close(
        final, torch.from_numpy(jax_oracle["final"]), rtol=0, atol=STACK_ATOL
    )


def test_its_own_rotary_tables_are_wired_correctly(
    encoder: MotorEncoder, batch: dict[str, torch.Tensor], jax_oracle: NpzFile
) -> None:
    """The unsubstituted run, bounded by jaxlib's sin rather than by the stack.

    This is what proves forward builds the tables from the real ages at the head
    width: the wrong width, the normalised ages, or a rotary schedule off by a factor
    all miss by far more than jaxlib's own 1.6e-03. It cannot be tighter, because
    torch is the more accurate of the two.
    """
    with torch.no_grad():
        final = encoder(**batch)

    torch.testing.assert_close(
        final, torch.from_numpy(jax_oracle["final"]), rtol=0, atol=LIBM_ATOL
    )


def test_in_norm_matches_the_oracle(
    traced: tuple[torch.Tensor, dict[str, torch.Tensor]], jax_oracle: NpzFile
) -> None:
    """The embedding, the ones-fill and the input norm, before any block runs.

    Separated from the stack because everything below it inherits an error here.
    """
    _final, captured = traced

    torch.testing.assert_close(
        captured["in_norm"],
        torch.from_numpy(jax_oracle["after_in_norm"]),
        rtol=0,
        atol=STACK_ATOL,
    )


@pytest.mark.parametrize("layer", range(N_LAYERS))
def test_every_layer_matches_the_oracle(
    traced: tuple[torch.Tensor, dict[str, torch.Tensor]],
    jax_oracle: NpzFile,
    layer: int,
) -> None:
    """Each block's output, so drift localises to one layer instead of the stack.

    This is also what pins the load order: the twelve blocks carry independent
    weights, so a permuted load still produces well-shaped finite features and only
    shows up as the wrong numbers, here.
    """
    _final, captured = traced

    torch.testing.assert_close(
        captured[f"layer_{layer:02d}"],
        torch.from_numpy(jax_oracle[f"layer_{layer:02d}_output"]),
        rtol=0,
        atol=STACK_ATOL,
    )


def test_blocks_are_loaded_in_checkpoint_order(
    encoder: MotorEncoder, haiku_params: dict[str, torch.Tensor]
) -> None:
    """Each block's norm scale identifies which checkpoint layer it came from.

    A direct check, so a reversed or off-by-one load fails here with a clear cause
    rather than as twelve simultaneously wrong activations.
    """
    for index, block in enumerate(encoder.blocks):
        expected = haiku_params[f"loop_{index}/TransformerBlock/~/rms_norm::scale"]

        torch.testing.assert_close(block.norm.weight, expected, rtol=0, atol=0)


def test_the_two_outer_norms_are_not_swapped(
    encoder: MotorEncoder, haiku_params: dict[str, torch.Tensor]
) -> None:
    """`rms_norm` is the input norm and `rms_norm_1` the output one.

    Both are (768, ) scales on the same stream, so swapping them is not a shape error.
    """
    torch.testing.assert_close(
        encoder.in_norm.weight, haiku_params["rms_norm::scale"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        encoder.out_norm.weight, haiku_params["rms_norm_1::scale"], rtol=0, atol=0
    )


def test_load_haiku_refuses_an_incomplete_checkpoint(
    haiku_params: dict[str, torch.Tensor],
) -> None:
    """Checked up front, so a stack cannot end up half-loaded and half-random."""
    incomplete = {
        key: value
        for key, value in haiku_params.items()
        if key != "loop_7/TransformerBlock/~/linear_1::b"
    }
    module = MotorEncoder(VOCAB, HIDDEN, INTERMEDIATE, N_HEADS, N_LAYERS, 496)

    with pytest.raises(KeyError):
        module.load_haiku(incomplete)


def test_invalid_positions_are_filled_with_ones_not_zeros(
    tiny: MotorEncoder, tiny_batch: dict[str, torch.Tensor]
) -> None:
    """An invalid row is replaced with ones before the norm, and ones are load bearing.

    Read off the norm's input, because the fill has no observable consequence
    downstream: RMSNorm's epsilon sits inside the square root, so a zero row
    normalises to zero rather than to NaN and a zero fill would otherwise look
    perfectly healthy. The oracle cannot cover this either -- its batch has every
    position valid.
    """
    tiny_batch["valid_tokens"] = torch.tensor([True, False, False])
    embedded = tiny.embedding(tiny_batch["indices"], tiny_batch["seq_len"])
    captured: dict[str, torch.Tensor] = {}
    handle = tiny.in_norm.register_forward_pre_hook(
        lambda _module, args: captured.__setitem__("x", args[0].detach().clone())
    )

    try:
        tiny(**tiny_batch)
    finally:
        handle.remove()

    torch.testing.assert_close(captured["x"][1:], torch.ones(2, 4))
    torch.testing.assert_close(captured["x"][0], embedded[0], rtol=0, atol=0)


def test_invalid_positions_ignore_their_tokens(
    tiny: MotorEncoder, tiny_batch: dict[str, torch.Tensor]
) -> None:
    """The fill replaces the embedded row outright rather than adding to it.

    So which tokens an invalid position held cannot reach the stack at all, whatever
    they were.
    """
    tiny_batch["valid_tokens"] = torch.zeros(3, dtype=torch.bool)
    first = tiny(**tiny_batch)

    tiny_batch["indices"] = torch.tensor([[7, 0], [7, 1], [7, 2]])

    torch.testing.assert_close(tiny(**tiny_batch), first, rtol=0, atol=0)


def test_compute_dtype_tracks_the_stack_not_the_table(tiny: MotorEncoder) -> None:
    """Released inference casts every parameter to float16 except the embedding.

    `half()` then `embedding.float()` reproduces that, and the property must report
    the stack's dtype so forward casts the activations at the right point.
    """
    tiny.half()
    tiny.embedding.float()

    assert tiny.compute_dtype == torch.float16
    assert tiny.embedding.embeddings.weight.dtype == torch.float32


@pytest.mark.parametrize(
    "field",
    ["ages", "normed_ages", "valid_tokens", "segment_ids"],
)
def test_rejects_per_position_tensors_of_the_wrong_length(
    tiny: MotorEncoder, tiny_batch: dict[str, torch.Tensor], field: str
) -> None:
    """A short per-position tensor would broadcast rather than fail.

    Silently rotating or masking the wrong positions is the failure mode.
    """
    tiny_batch[field] = tiny_batch[field][:2]

    with pytest.raises(ValueError):
        tiny(**tiny_batch)


def test_heads_must_divide_the_model_width() -> None:
    """Otherwise split_heads would reshape into the wrong head layout at run time."""
    with pytest.raises(ValueError):
        MotorEncoder(
            vocab_size=8,
            hidden_size=5,
            intermediate_size=8,
            n_heads=2,
            n_layers=1,
            attention_width=4,
        )


def test_lora_target_names_are_reachable(encoder: MotorEncoder) -> None:
    """LoRA adapters resolve by module path, so these strings are a contract.

    Renaming `blocks` or the projections silently changes which weights an adapter
    trains, without any error.
    """
    names = dict(encoder.named_modules())

    assert "blocks.0.input_proj.q_proj" in names
    assert "blocks.11.input_proj.v_proj" in names


def test_probe_arrays_are_what_the_oracle_dumped(jax_oracle: NpzFile) -> None:
    """Guards the shape assumptions the oracle tests above depend on."""
    assert jax_oracle["final"].shape == (SEQ_LEN, HIDDEN)
    assert jax_oracle["after_in_norm"].shape == (SEQ_LEN, HIDDEN)
    assert jax_oracle["layer_11_output"].shape == (SEQ_LEN, HIDDEN)
    assert bool(jax_oracle["batch_valid_tokens"].all())


# A stack small enough to build without the checkpoint. The batch dimension is a
# structural property, so these need weights, not the RIGHT weights, and they run
# whether or not the oracle has been generated locally.
SMALL = {
    "vocab_size": 32,
    "hidden_size": 8,
    "intermediate_size": 16,
    "n_heads": 2,
    "n_layers": 2,
    "attention_width": 3,
}
SMALL_SEQ = 6

# A batched matmul accumulates in a different order from a 2-D one, which moves the
# features by ~2.4e-07 over two blocks -- the same class of difference as the JAX
# comparisons above. Every failure these tests exist for (a table read from the wrong
# row, a mask aligned against the heads, a mis-flattened write index) substitutes one
# subject's data for another's and moves the output by orders of magnitude more.
BATCH_ATOL = 1e-06


@pytest.fixture
def small_encoder() -> MotorEncoder:
    """A randomly initialised encoder at toy widths."""
    torch.manual_seed(0)
    return MotorEncoder(**SMALL).eval()


def _one_sequence(seed: int) -> dict[str, torch.Tensor]:
    """Builds the forward arguments for a single sequence."""
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        SMALL["vocab_size"], (SMALL_SEQ,), generator=generator, dtype=torch.int64
    )
    ages = torch.sort(
        torch.rand(SMALL_SEQ, generator=generator) * 30_000
    ).values.float()
    valid = torch.ones(SMALL_SEQ, dtype=torch.bool)
    valid[-1] = False  # a padded tail, which every real batch carries
    return {
        "indices": torch.stack((tokens, torch.arange(SMALL_SEQ))).T,
        "seq_len": SMALL_SEQ,
        "ages": ages,
        "normed_ages": (ages - ages.mean()) / ages.std(),
        "valid_tokens": valid,
        "segment_ids": torch.zeros(SMALL_SEQ, dtype=torch.long),
    }


def _as_batch(sequences: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stacks single sequences into one batched call, offsetting the write indices."""
    pairs = torch.cat(
        [
            torch.stack(
                (one["indices"][:, 0], one["indices"][:, 1] + row * SMALL_SEQ)
            ).T
            for row, one in enumerate(sequences)
        ]
    )
    return {
        "indices": pairs,
        "seq_len": SMALL_SEQ,
        **{
            name: torch.stack([one[name] for one in sequences])
            for name in ("ages", "normed_ages", "valid_tokens", "segment_ids")
        },
    }


def test_a_batch_matches_running_the_sequences_alone(
    small_encoder: MotorEncoder,
) -> None:
    """The whole contract of the batch dimension, and it is exact.

    Every per-position tensor gains a leading axis and the write indices become flat
    over the batch, so an off-by-one in that flattening, a table taken from the wrong
    row, or a norm reducing over the wrong axis all show up here.
    """
    sequences = [_one_sequence(seed) for seed in (1, 2, 3)]

    batched = small_encoder(**_as_batch(sequences))

    assert batched.shape == (len(sequences), SMALL_SEQ, SMALL["hidden_size"])
    for row, one in enumerate(sequences):
        torch.testing.assert_close(
            batched[row], small_encoder(**one), rtol=0, atol=BATCH_ATOL
        )


def test_one_sequence_cannot_reach_another(small_encoder: MotorEncoder) -> None:
    """The failure the batch dimension introduces, and the one nothing else catches.

    A mask that broadcasts against the head axis instead of the batch, or an
    embedding reshaped in the wrong order, mixes subjects together while returning
    finite features of the right shape.
    """
    sequences = [_one_sequence(1), _one_sequence(2)]
    before = small_encoder(**_as_batch(sequences))

    disturbed = small_encoder(**_as_batch([sequences[0], _one_sequence(99)]))

    torch.testing.assert_close(disturbed[0], before[0], rtol=0, atol=0)
    assert not torch.equal(disturbed[1], before[1])


def test_a_batch_of_one_matches_the_unbatched_call(
    small_encoder: MotorEncoder,
) -> None:
    """The oracle tests run unbatched, so the two paths must not have diverged."""
    one = _one_sequence(7)

    torch.testing.assert_close(
        small_encoder(**_as_batch([one]))[0],
        small_encoder(**one),
        rtol=0,
        atol=BATCH_ATOL,
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        pytest.param("ages", torch.ones(1, 1, SMALL_SEQ), "shaped", id="three_dims"),
        pytest.param("ages", torch.ones(2, SMALL_SEQ + 1), "positions", id="wrong_len"),
        pytest.param(
            "valid_tokens",
            torch.ones(SMALL_SEQ, dtype=torch.bool),
            "like the ages",
            id="unbatched_companion",
        ),
    ],
)
def test_rejects_per_position_tensors_that_disagree(
    small_encoder: MotorEncoder, field: str, value: torch.Tensor, match: str
) -> None:
    """A companion left unbatched would broadcast rather than fail."""
    arguments = _as_batch([_one_sequence(1), _one_sequence(2)]) | {field: value}

    with pytest.raises(ValueError, match=match):
        small_encoder(**arguments)
