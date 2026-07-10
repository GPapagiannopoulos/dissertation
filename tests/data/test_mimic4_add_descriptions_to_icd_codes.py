"""Testing suite for the add_descriptions_to_icd_codes helper function."""

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.data.sources import mimic4_add_descriptions_to_icd_codes


@pytest.mark.parametrize(
    "map_data, lf_data, event_type, expected_lf_data",
    [
        # 0. Correctly matches icd code and version
        (
            {
                "icd_version": pl.Series(["a"], dtype=pl.String),
                "icd_code": pl.Series(["123"], dtype=pl.String),
                "long_title": pl.Series(["description"], dtype=pl.String),
            },
            {
                "icd_version": pl.Series(["a"], dtype=pl.String),
                "icd_code": pl.Series(["123"], dtype=pl.String),
            },
            "event_type",
            {
                "event_type/icd_version": pl.Series(["a"], dtype=pl.String),
                "event_type/icd_code": pl.Series(["123"], dtype=pl.String),
                "event_type/description": pl.Series(["description"], dtype=pl.String),
            },
        )
    ],
)
def test_add_descriptions_to_icd_codes_happy_path(
    mapping_csv: Callable,
    source_lazyframe: Callable,
    map_data: dict[str, pl.Series],
    lf_data: dict[str, pl.Series],
    event_type: str,
    expected_lf_data: dict[str, pl.Series],
) -> None:
    """Asserts expected behaviour with intended use."""
    path = mapping_csv(map_data, event_type)
    source = source_lazyframe(event_type, **lf_data)
    joined_lf = mimic4_add_descriptions_to_icd_codes(source, path, event_type)
    expected_lf = pl.LazyFrame(expected_lf_data)
    assert_frame_equal(joined_lf, expected_lf)
