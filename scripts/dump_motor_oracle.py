r"""One-off driver for the MOTOR port: dump a numerical oracle from the JAX model.

Runs in the femr-v1 environment ONLY, which is the only one that can load the
released checkpoint:

    .venv-motor-v1/bin/python scripts/dump_motor_oracle.py --dtype fp32

Reproduces `femr.models.transformer.Transformer.__call__` step by step so every
intermediate can be captured, then writes the inputs, the per-layer hidden states and
the flat parameter tree to a single .npz. The PyTorch port is asserted layer by layer
against that file, which is what makes the port falsifiable.

The batch is synthetic. The oracle exists to assert the arithmetic of the
port. Tokenisation is a separate concern with its own failure modes,
and building a v1 extract just to get an oracle would cost a whole ETL detour.

Two dtypes:
- fp16 reproduces inference exactly (`convert_params` casts every float parameter
  but the embedding table, and the activations are cast right after `in_norm`). It is
  the end-of-port fidelity check.
- fp32 skips both casts. No two implementations are bit-identical in fp16 -- 1 ULP
  is 0.0039 at these magnitudes, compounding to a ~1e-2 noise floor over 12 layers,
  which is deep enough to hide a real bug. fp32 gives a 1e-6 target to develop against.
"""

import argparse
import pickle
import sys
from pathlib import Path

import femr.jax
import haiku as hk
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
from femr.models.transformer import (
    EHRTransformer,
    TransformerBlock,
    apply_rotary_pos_emb,
    convert_params,
    fixed_pos_embedding,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "motor_model" / "model"
DEST_DIR = ROOT / "motor_output"

PREFIX = "EHRTransformer/~/TransformerFeaturizer/~/Transformer/~/"

# must be a multiple of 16 (local_attention's abstract eval) and a power of two (the
# length bitmask only segments cleanly on power-of-two blocks)
SEQ_LEN = 64
SEED = 0

DTYPES = {"fp16": jnp.float16, "fp32": jnp.float32}


def _parse_args() -> argparse.Namespace:
    """Parses the paths and the one knob that decides what the oracle is for."""
    parser = argparse.ArgumentParser(
        description="Dump per-layer activations from the JAX MOTOR checkpoint."
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="fp16",
        help="fp16 reproduces inference; fp32 is the tight-tolerance port target.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Defaults to motor_output/oracle_{dtype}.npz.",
    )

    return parser.parse_args()


def load_config(model_dir: Path) -> dict:
    """Reads the model's own hyperparameters, so nothing here is hardcoded twice."""
    return msgpack.unpackb(model_dir.joinpath("config.msgpack").read_bytes(), raw=False)


def load_params(model_dir: Path) -> dict:
    """Unpickles the haiku parameter tree. Needs a real jax import in scope."""
    with model_dir.joinpath("best").open("rb") as f:
        return pickle.load(f)


def make_batch(config: dict, seq_len: int, seed: int) -> dict:
    """Builds a synthetic transformer batch with the fields the model reads."""
    rng = np.random.default_rng(seed)
    vocab_size = config["transformer"]["vocab_size"]

    # ages are in minutes in femr v1 and strictly increasing within a subject
    ages = np.sort(rng.uniform(0, 60 * 24 * 365 * 80, size=seq_len)).astype(np.float32)
    normalized_ages = ((ages - ages.mean()) / (ages.std() + 1e-6)).astype(np.float32)

    # is_hierarchical: a token is the SUM of its ancestors' embedding rows.
    # indices[:, 0] reads from the embedding, indices[:, 1] writes to a position.
    # Must be sorted by the write index.
    n_ancestors = rng.integers(1, 5, size=seq_len)
    write_idx = np.repeat(np.arange(seq_len), n_ancestors)
    read_idx = rng.integers(0, vocab_size, size=write_idx.shape[0])
    sparse_token_indices = np.stack([read_idx, write_idx], axis=1).astype(np.uint32)

    return {
        "ages": jnp.asarray(ages),
        "normalized_ages": jnp.asarray(normalized_ages),
        "sparse_token_indices": jnp.asarray(sparse_token_indices),
        "valid_tokens": jnp.ones(seq_len, dtype=jnp.bool_),
        "length": jnp.asarray(seq_len, dtype=jnp.uint32),
        # every position is a label, so the featurizer returns the full sequence
        "label_indices": jnp.arange(seq_len, dtype=jnp.uint32),
    }


def rms_norm(x, scale, eps: float = 1e-5):
    """Haiku's hk.RMSNorm(-1) with create_scale=True, spelled out.

    Note the scale is cast to the *input* dtype and eps sits inside the sqrt.
    """
    scale = scale.astype(x.dtype)
    mean_squared = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * scale * jax.lax.rsqrt(mean_squared + eps)


def layer_params(params: dict, i: int) -> dict:
    """Slices the flat checkpoint into one block's params.

    Keyed as the inner hk.transform expects: the `loop_i` prefix hk.lift added is
    stripped.
    """
    scope = f"{PREFIX}loop_{i}/"
    return {k[len(scope) :]: v for k, v in params.items() if k.startswith(scope)}


def cross_check(
    config: dict, raw_params: dict, batch: dict, final: np.ndarray
) -> tuple:
    """Runs the real EHRTransformer and measures the decomposition against it.

    If this drifts, the step-by-step version above is wrong and every downstream
    comparison against the port is measuring the wrong thing.

    Always fp16: `EHRTransformer.__call__` casts the activations to fp16 unconditionally
    (transformer.py:222), so fp32 parameters trip the `x.dtype == sin.dtype` assertion
    in `apply_rotary_pos_emb`. The reference has exactly one precision.
    """
    model = hk.transform(
        lambda b: EHRTransformer(config)(b, is_training=False, no_task=True)
    )
    features, _mask = model.apply(
        convert_params(raw_params, jnp.float16),
        jax.random.PRNGKey(SEED),
        {"transformer": batch, "task": {}},
    )
    ref = np.asarray(features)
    return ref, np.abs(ref.astype(np.float32) - final.astype(np.float32)).max()


def main() -> None:
    """Dumps the oracle at the requested precision."""
    args = _parse_args()
    compute_dtype = DTYPES[args.dtype]
    dest = args.dest or DEST_DIR / f"oracle_{args.dtype}.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(args.model_dir)
    raw_params = load_params(args.model_dir)
    tconfig = config["transformer"]
    batch = make_batch(config, SEQ_LEN, SEED)

    # inference casts every float param to fp16 EXCEPT the embedding table. In fp32
    # the checkpoint is already float32, so there is nothing to convert.
    params = (
        convert_params(raw_params, jnp.float16)
        if compute_dtype == jnp.float16
        else raw_params
    )

    out: dict[str, np.ndarray] = {}

    # --- embedding: sum over ontology ancestors -------------------------------
    embeddings = params[f"{PREFIX}embed"]["embeddings"]
    x = femr.jax.gather_scatter_add(
        embeddings, batch["sparse_token_indices"], batch["ages"].shape[0]
    )
    out["embedded"] = np.asarray(x)

    # invalid positions are overwritten with ones, then normed
    x = jnp.where(
        batch["valid_tokens"].reshape((-1, 1)), x, jnp.ones((1, 1), dtype=x.dtype)
    )
    x = rms_norm(x, params[f"{PREFIX}rms_norm"]["scale"])
    out["after_in_norm"] = np.asarray(x)

    # the whole stack below in_norm runs in fp16 at inference (transformer.py:222)
    x = x.astype(compute_dtype)
    normed_ages = batch["normalized_ages"].astype(x.dtype)

    head_size = tconfig["hidden_size"] // tconfig["n_heads"]
    pos_embed = fixed_pos_embedding(batch["ages"], head_size, x.dtype)
    out["rotary_sin"] = np.asarray(pos_embed[0])
    out["rotary_cos"] = np.asarray(pos_embed[1])

    # --- a rotary probe -------------------------------------------------------
    # q and k never surface from inside TransformerBlock, so apply_rotary would
    # otherwise only be testable at block level, where a layout error is one
    # contribution to a delta among many. Rotating a fixed random tensor of
    # exactly q's post-`move_to_batch` shape gives the port a direct target.
    probe = jax.random.normal(
        jax.random.PRNGKey(SEED + 1), (tconfig["n_heads"], SEQ_LEN, head_size)
    ).astype(x.dtype)
    out["rotary_probe"] = np.asarray(probe)
    out["rotary_probe_rotated"] = np.asarray(apply_rotary_pos_emb(probe, pos_embed))

    # --- the 12 blocks --------------------------------------------------------
    block = hk.transform(lambda *a: TransformerBlock(tconfig)(*a))
    rng = jax.random.PRNGKey(SEED)
    out["layer_00_input"] = np.asarray(x)
    for i in range(tconfig["n_layers"]):
        delta = block.apply(
            layer_params(params, i), rng, x, normed_ages, pos_embed, batch, False
        )
        out[f"layer_{i:02d}_delta"] = np.asarray(delta)
        x = x + delta  # the residual lives in the caller (jax.lax.scan)
        out[f"layer_{i:02d}_output"] = np.asarray(x)

    out["final"] = np.asarray(rms_norm(x, params[f"{PREFIX}rms_norm_1"]["scale"]))

    # --- cross-check against the real module ----------------------------------
    ref, delta = cross_check(config, raw_params, batch, out["final"])
    out["reference_features"] = ref
    print(f"decomposition vs real EHRTransformer: max abs delta = {delta:.3e}")
    if compute_dtype == jnp.float16:
        assert delta == 0.0, f"decomposition drifted from the real module by {delta}"
    else:
        # The reference is fp16 and this dump is fp32, so a nonzero delta here is
        # expected -- it IS the fp16 rounding the fp32 dump exists to escape, and its
        # size is the noise floor a port could otherwise hide a bug under. The fp16
        # run, which asserts an exact match, is what proves the decomposition
        # faithful; this path changes one dtype and nothing else.
        print("  (fp32: nonzero expected -- the reference is fp16 below in_norm)")

    # --- inputs, so the port runs on exactly the same batch -------------------
    for k, v in batch.items():
        out[f"batch_{k}"] = np.asarray(v)

    # --- the flat parameter tree ---------------------------------------------
    for module, leaves in params.items():
        for name, value in leaves.items():
            out[f"param::{module}::{name}"] = np.asarray(value)

    np.savez(dest, **out)
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB), {len(out)} arrays")
    print(
        f"final repr: shape={out['final'].shape} dtype={out['final'].dtype} "
        f"mean={out['final'].mean():.6f} std={out['final'].std():.6f}"
    )


if __name__ == "__main__":
    sys.exit(main())
