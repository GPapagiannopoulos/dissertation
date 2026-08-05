"""Fixtures for testing the MOTOR port functions and layers.

Every test here compares against a dump of the released JAX model, per layer. The
port is developed against motor_output/oracle_fp32.npz, where one float16 ULP would
otherwise hide a real bug; oracle_fp16.npz is the end-of-port fidelity check,
because float16 is the path real inference takes. Both artifacts are gitignored,
so the fixtures skip when they are absent.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.lib.npyio import NpzFile

_ROOT: Path = Path(__file__).resolve().parents[3]
_ORACLE_PATH: Path = _ROOT / "motor_output" / "oracle_fp32.npz"
_ORACLE_FP16_PATH: Path = _ROOT / "motor_output" / "oracle_fp16.npz"

# every parameter key in the dump is "param::{module}::{leaf}", and every module in
# the encoder hangs off this haiku scope
_PREFIX: str = "EHRTransformer/~/TransformerFeaturizer/~/Transformer/~/"


def _open_oracle(path: Path, flag: str) -> NpzFile:
    """Opens a dump, or skips with the command that regenerates it."""
    if not path.exists():
        pytest.skip(
            f"{path} not found; regenerate with '.venv-motor-v1/bin/python "
            f"scripts/dump_motor_oracle.py --dtype {flag}'"
        )
    return np.load(path)


def _encoder_params(oracle: NpzFile) -> dict[str, torch.Tensor]:
    """Pulls every encoder parameter out of a dump, keyed below the haiku scope.

    The task head's parameters hang off a sibling scope and so are excluded, which
    is what the encoder expects.
    """
    prefix = f"param::{_PREFIX}"
    return {
        key.removeprefix(prefix): torch.from_numpy(oracle[key])
        for key in oracle.files
        if key.startswith(prefix)
    }


@pytest.fixture(scope="session")
def jax_oracle() -> NpzFile:
    """Opens the float32 dump, or skips when it has not been generated locally."""
    return _open_oracle(_ORACLE_PATH, "fp32")


@pytest.fixture(scope="session")
def jax_oracle_fp16() -> NpzFile:
    """Opens the float16 dump, the one real inference is run in."""
    return _open_oracle(_ORACLE_FP16_PATH, "fp16")


@pytest.fixture
def oracle_param(jax_oracle: NpzFile) -> Callable[[str], np.ndarray]:
    """Returns a lookup for one checkpoint parameter, keyed below the haiku scope.

    Spares every test the 52-character prefix: `oracle_param("rms_norm::scale")`.
    """

    def _get(name: str) -> np.ndarray:
        return jax_oracle[f"param::{_PREFIX}{name}"]

    return _get


@pytest.fixture(scope="session")
def haiku_params(jax_oracle: NpzFile) -> dict[str, torch.Tensor]:
    """The float32 encoder parameters. Session scoped: the table alone is 200 MB."""
    return _encoder_params(jax_oracle)


@pytest.fixture(scope="session")
def haiku_params_fp16(jax_oracle_fp16: NpzFile) -> dict[str, torch.Tensor]:
    """The released inference parameters: float16 throughout, bar the table.

    Released inference converts every parameter to float16 and skips the embedding
    table, so the table here is bit-identical to the float32 dump's.
    """
    return _encoder_params(jax_oracle_fp16)
