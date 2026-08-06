"""Fixtures shared by every modelling suite.

The released MOTOR dictionary is read by two stages that sit either side of the
pipeline -- the concept map in etl_pipeline, and the tokeniser in motor -- so its
factories live here rather than in either one.
"""

from collections.abc import Callable
from pathlib import Path

import msgpack
import pytest


@pytest.fixture
def rollup_entry() -> Callable:
    """Returns a factory for single dictionary entries."""

    def _make(code_string: str, entry_type: int, **overrides) -> dict:
        return {
            "code_string": code_string,
            "text_string": "",
            "type": entry_type,
            "val_start": 0.0,
            "val_end": 0.0,
            "weight": -0.1,
        } | overrides

    return _make


@pytest.fixture
def make_dictionary(tmp_path: Path) -> Callable:
    """Returns a factory writing a msgpack dictionary."""

    def _make(**overrides) -> Path:
        defaults = {
            "age_stats": {"mean": 0.0, "std": 1.0},
            "all_parents": {"SNOMED/1": ("SNOMED/1",)},
            "ontology_rollup": [],
            "regular": [],
        }
        defaults.update(**overrides)

        path = tmp_path / "dictionary"
        path.write_bytes(msgpack.dumps(defaults))
        return path

    return _make
