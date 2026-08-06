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
import polars as pl
import pytest
import torch
from numpy.lib.npyio import NpzFile

from thesis.modelling.motor.tokenizer import TokenTable

# the open ends of a code's bins, C's DBL_MAX, as the dictionary stores them
LOWEST: float = -1.7976931348623157e308
HIGHEST: float = 1.7976931348623157e308

NUMERIC_SCHEMA: dict[str, pl.DataType] = {
    "code": pl.String,
    "val_start": pl.Float64,
    "val_end": pl.Float64,
    "numeric_indices": pl.UInt32,
}
TEXT_SCHEMA: dict[str, pl.DataType] = {
    "code": pl.String,
    "text_string": pl.String,
    "text_indices": pl.UInt32,
}

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


@pytest.fixture
def make_bins() -> Callable:
    """Returns a factory for one code's numeric bins, from its cut points.

    The bins a code owns partition the whole line, so n cut points give n + 1 bins
    running from -DBL_MAX to +DBL_MAX. Rows are numbered from `first_index`.
    """

    def _make(code: str, cuts: list[float], first_index: int = 0) -> pl.DataFrame:
        edges = [LOWEST, *cuts, HIGHEST]
        return pl.DataFrame(
            {
                "code": [code] * (len(edges) - 1),
                "val_start": edges[:-1],
                "val_end": edges[1:],
                "numeric_indices": range(first_index, first_index + len(edges) - 1),
            },
            schema=NUMERIC_SCHEMA,
        )

    return _make


@pytest.fixture
def make_texts() -> Callable:
    """Returns a factory for text tokens, from (code, value, row) triples."""

    def _make(*rows: tuple[str, str, int]) -> pl.DataFrame:
        codes, values, indices = zip(*rows, strict=False) if rows else ((), (), ())
        return pl.DataFrame(
            {
                "code": list(codes),
                "text_string": list(values),
                "text_indices": list(indices),
            },
            schema=TEXT_SCHEMA,
        )

    return _make


@pytest.fixture
def make_token_table(make_bins: Callable, make_texts: Callable) -> Callable:
    """Returns a factory for TokenTable objects, one token of each kind seeded."""

    def _make(**overrides) -> TokenTable:
        fields = {
            "code_tokens": {"SNOMED/plain": 0},
            "numeric_tokens": make_bins("LOINC/numeric", [1.0], first_index=1),
            "text_tokens": make_texts(("SNOMED/text", "N", 3)),
            "all_parents": {},
            "age_mean": 0.0,
            "age_std": 1.0,
        }
        return TokenTable(**(fields | overrides))

    return _make


@pytest.fixture
def make_events() -> Callable:
    """Returns a factory for MEDS-shaped events, overriding only what a case needs.

    Columns beyond these exist in the real shards and are irrelevant here: the
    tokeniser reads the code and its two value columns and nothing else.
    """

    def _make(**columns: list) -> pl.LazyFrame:
        height = max((len(values) for values in columns.values()), default=1)
        defaults = {
            "subject_id": [1],
            "code": ["SNOMED/plain"],
            "numeric_value": [None],
            "text_value": [None],
        }
        merged = {
            name: values * height if len(values) == 1 else values
            for name, values in (defaults | columns).items()
        }
        frame = pl.DataFrame(
            merged,
            schema_overrides={
                "subject_id": pl.Int64,
                "code": pl.String,
                "numeric_value": pl.Float32,
                "text_value": pl.String,
            },
        )
        return frame.lazy()

    return _make
