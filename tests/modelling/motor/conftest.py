"""Fixtures for testing the MOTOR port functions and layers.

Every test here compares against motor_output/oracle_fp32.npz, the per-layer dump
of the released JAX model. That artifact is 577 MB and gitignored, so the
fixtures skip when it is absent.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.lib.npyio import NpzFile

_ROOT: Path = Path(__file__).resolve().parents[3]
_ORACLE_PATH: Path = _ROOT / "motor_output" / "oracle_fp32.npz"

# every parameter key in the dump is "param::{module}::{leaf}", and every module in
# the encoder hangs off this haiku scope
_PREFIX: str = "EHRTransformer/~/TransformerFeaturizer/~/Transformer/~/"


@pytest.fixture(scope="session")
def jax_oracle() -> NpzFile:
    """Opens the oracle dump, or skips when it has not been generated locally."""
    if not _ORACLE_PATH.exists():
        pytest.skip(
            f"{_ORACLE_PATH} not found; regenerate with "
            "'.venv-motor-v1/bin/python scripts/dump_motor_oracle.py --dtype fp32'"
        )
    return np.load(_ORACLE_PATH)


@pytest.fixture
def oracle_param(jax_oracle: NpzFile) -> Callable[[str], np.ndarray]:
    """Returns a lookup for one checkpoint parameter, keyed below the haiku scope.

    Spares every test the 52-character prefix: `oracle_param("rms_norm::scale")`.
    """

    def _get(name: str) -> np.ndarray:
        return jax_oracle[f"param::{_PREFIX}{name}"]

    return _get
