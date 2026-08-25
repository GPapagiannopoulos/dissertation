"""Fixtures shared by every modelling suite.

The released MOTOR dictionary is read by two stages that sit either side of the
pipeline -- the concept map in etl, and the tokeniser in backbone -- so its
factories live here rather than in either one, as does the MEDS-shaped event
builder the tokeniser and the sequence builder share.
"""

from collections.abc import Callable
from pathlib import Path

import msgpack
import polars as pl
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
