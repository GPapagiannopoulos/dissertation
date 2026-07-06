"""Test suite for the EDASource class methods and helper functions."""

import polars as pl
import pytest

from thesis.data.sources import cast_frame


@pytest.mark.parametrize(
    "mimic_data, schema_overrides, expected_dtype",
    [
        (
            {"col": pl.Series(["a", "b", "c"], dtype=pl.String)},
            {"col": "String"},
            pl.String,
        )
    ],
)
def test_happy_path_conversions(
    data: dict[str, pl.Series], schema_overrides: dict[str, str], expected_dtype: type
) -> None:
    """Asserts that simple dtype castings work as intended."""
    lf = cast_frame(pl.LazyFrame(data), schema_overrides)
    assert lf.collect_schema()["col"] == expected_dtype
