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
        ),
        # 1. Correct composite matching
        (
            {
                "icd_version": pl.Series(["10", "9"], dtype=pl.String),
                "icd_code": pl.Series(["123", "123"], dtype=pl.String),
                "long_title": pl.Series(
                    ["description_one", "description_two"], dtype=pl.String
                ),
            },
            {
                "icd_version": pl.Series(["10", "9"], dtype=pl.String),
                "icd_code": pl.Series(["123", "123"], dtype=pl.String),
            },
            "event_type",
            {
                "event_type/icd_version": pl.Series(["10", "9"], dtype=pl.String),
                "event_type/icd_code": pl.Series(["123", "123"], dtype=pl.String),
                "event_type/description": pl.Series(
                    ["description_one", "description_two"], dtype=pl.String
                ),
            },
        ),
        # 2. Missing event_code returns a null description
        (
            {
                "icd_version": pl.Series(["10", "9"], dtype=pl.String),
                "icd_code": pl.Series(["123", "124"], dtype=pl.String),
                "long_title": pl.Series(
                    ["description_one", "description_two"], dtype=pl.String
                ),
            },
            {
                "icd_version": pl.Series(["10", "9"], dtype=pl.String),
                "icd_code": pl.Series(["123", "123"], dtype=pl.String),
            },
            "event_type",
            {
                "event_type/icd_version": pl.Series(["10", "9"], dtype=pl.String),
                "event_type/icd_code": pl.Series(["123", "123"], dtype=pl.String),
                "event_type/description": pl.Series(
                    ["description_one", None], dtype=pl.String
                ),
            },
        ),
        # 3. Function casts mapping cols to String
        (
            {
                "icd_version": pl.Series([10, 9], dtype=pl.Int32),
                "icd_code": pl.Series(["123", "123"], dtype=pl.String),
                "long_title": pl.Series(
                    ["description_one", "description_two"], dtype=pl.String
                ),
            },
            {
                "icd_version": pl.Series(["10", "9"], dtype=pl.String),
                "icd_code": pl.Series(["123", "123"], dtype=pl.String),
            },
            "event_type",
            {
                "event_type/icd_version": pl.Series(["10", "9"], dtype=pl.String),
                "event_type/icd_code": pl.Series(["123", "123"], dtype=pl.String),
                "event_type/description": pl.Series(
                    ["description_one", "description_two"], dtype=pl.String
                ),
            },
        ),
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


@pytest.mark.parametrize(
    "map_data, lf_data, event_type",
    [
        # 0. Mapping df is missing icd_code field
        (
            {
                "icd_version": pl.Series(["a"], dtype=pl.String),
                "icd_name": pl.Series(["123"], dtype=pl.String),
                "long_title": pl.Series(["description"], dtype=pl.String),
            },
            {
                "icd_version": pl.Series(["a"], dtype=pl.String),
                "icd_code": pl.Series(["123"], dtype=pl.String),
            },
            "event_type",
        ),
        # 1. Mapping df is missing icd_version field
        (
            {
                "icd_revision": pl.Series(["a"], dtype=pl.String),
                "icd_code": pl.Series(["123"], dtype=pl.String),
                "long_title": pl.Series(["description"], dtype=pl.String),
            },
            {
                "icd_version": pl.Series(["a"], dtype=pl.String),
                "icd_code": pl.Series(["123"], dtype=pl.String),
            },
            "event_type",
        ),
        # 2. LazyFrame is missing icd_code
        (
            {
                "icd_version": pl.Series(["a"], dtype=pl.String),
                "icd_code": pl.Series(["123"], dtype=pl.String),
                "long_title": pl.Series(["description"], dtype=pl.String),
            },
            {
                "icd_version": pl.Series(["a"], dtype=pl.String),
                "icd_name": pl.Series(["123"], dtype=pl.String),
            },
            "event_type",
        ),
        # 3. LazyFrame is missing icd_version
        (
            {
                "icd_version": pl.Series(["a"], dtype=pl.String),
                "icd_code": pl.Series(["123"], dtype=pl.String),
                "long_title": pl.Series(["description"], dtype=pl.String),
            },
            {
                "icd_revision": pl.Series(["a"], dtype=pl.String),
                "icd_code": pl.Series(["123"], dtype=pl.String),
            },
            "event_type",
        ),
        # 4. LazyFrame has an icd_version for the wrong event_type
        (
            {
                "icd_version": pl.Series(["a"], dtype=pl.String),
                "icd_code": pl.Series(["123"], dtype=pl.String),
                "long_title": pl.Series(["description"], dtype=pl.String),
            },
            {
                "other_type/icd_version": pl.Series(["a"], dtype=pl.String),
                "icd_name": pl.Series(["123"], dtype=pl.String),
            },
            "event_type",
        ),
    ],
)
def test_add_descriptions_to_icd_raises(
    mapping_csv: Callable,
    source_lazyframe: Callable,
    map_data: dict[str, pl.Series],
    lf_data: dict[str, pl.Series],
    event_type: str,
) -> None:
    """Asserts that missing columns raise an error."""
    path = mapping_csv(map_data, event_type)
    source = source_lazyframe(event_type, **lf_data)
    with pytest.raises(KeyError):
        mimic4_add_descriptions_to_icd_codes(source, path, event_type)
