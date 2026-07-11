"""Testing suite for the replace_mimic4_non_icd_codes helper."""

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.data.sources import replace_mimic4_non_icd_codes


@pytest.mark.parametrize(
    "map_data, lf_data, event_type, expected_lf_data",
    [
        # 0. Single match: label attached, itemid dropped, passthrough col survives
        (
            {
                "itemid": pl.Series(["1"], dtype=pl.String),
                "label": pl.Series(["description"], dtype=pl.String),
            },
            {
                "itemid": pl.Series(["1"], dtype=pl.String),
                "value": pl.Series(["x"], dtype=pl.String),
            },
            "event_type",
            {
                "event_type/value": pl.Series(["x"], dtype=pl.String),
                "event_type/description": pl.Series(["description"], dtype=pl.String),
            },
        ),
        # 1. Single-key correctness: each itemid maps to its own label
        (
            {
                "itemid": pl.Series(["1", "2"], dtype=pl.String),
                "label": pl.Series(["desc_one", "desc_two"], dtype=pl.String),
            },
            {
                "itemid": pl.Series(["1", "2"], dtype=pl.String),
                "value": pl.Series(["x", "y"], dtype=pl.String),
            },
            "event_type",
            {
                "event_type/value": pl.Series(["x", "y"], dtype=pl.String),
                "event_type/description": pl.Series(
                    ["desc_one", "desc_two"], dtype=pl.String
                ),
            },
        ),
        # 2. Unmatched itemid -> null description (left join keeps the row)
        (
            {
                "itemid": pl.Series(["1"], dtype=pl.String),
                "label": pl.Series(["desc_one"], dtype=pl.String),
            },
            {
                "itemid": pl.Series(["1", "2"], dtype=pl.String),
                "value": pl.Series(["x", "y"], dtype=pl.String),
            },
            "event_type",
            {
                "event_type/value": pl.Series(["x", "y"], dtype=pl.String),
                "event_type/description": pl.Series(
                    ["desc_one", None], dtype=pl.String
                ),
            },
        ),
        # 3. Function casts the mapping's itemid to String before joining
        (
            {
                "itemid": pl.Series([1, 2], dtype=pl.Int32),
                "label": pl.Series(["desc_one", "desc_two"], dtype=pl.String),
            },
            {
                "itemid": pl.Series(["1", "2"], dtype=pl.String),
                "value": pl.Series(["x", "y"], dtype=pl.String),
            },
            "event_type",
            {
                "event_type/value": pl.Series(["x", "y"], dtype=pl.String),
                "event_type/description": pl.Series(
                    ["desc_one", "desc_two"], dtype=pl.String
                ),
            },
        ),
    ],
)
def test_replace_non_icd_codes_happy_path(
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
    joined_lf = replace_mimic4_non_icd_codes(source, path, event_type)
    expected_lf = pl.LazyFrame(expected_lf_data)
    assert_frame_equal(joined_lf, expected_lf)


@pytest.mark.parametrize(
    "map_data, lf_data, event_type",
    [
        # 0. Mapping frame is missing the itemid key
        (
            {
                "item_id": pl.Series(["1"], dtype=pl.String),
                "label": pl.Series(["description"], dtype=pl.String),
            },
            {
                "itemid": pl.Series(["1"], dtype=pl.String),
            },
            "event_type",
        ),
        # 1. Mapping frame is missing the label (rename target)
        (
            {
                "itemid": pl.Series(["1"], dtype=pl.String),
                "name": pl.Series(["description"], dtype=pl.String),
            },
            {
                "itemid": pl.Series(["1"], dtype=pl.String),
            },
            "event_type",
        ),
        # 2. LazyFrame is missing the itemid key
        (
            {
                "itemid": pl.Series(["1"], dtype=pl.String),
                "label": pl.Series(["description"], dtype=pl.String),
            },
            {
                "item_id": pl.Series(["1"], dtype=pl.String),
            },
            "event_type",
        ),
        # 3. LazyFrame has an itemid under the wrong event_type prefix
        (
            {
                "itemid": pl.Series(["1"], dtype=pl.String),
                "label": pl.Series(["description"], dtype=pl.String),
            },
            {
                "other_type/itemid": pl.Series(["1"], dtype=pl.String),
            },
            "event_type",
        ),
    ],
)
def test_replace_non_icd_codes_raises(
    mapping_csv: Callable,
    source_lazyframe: Callable,
    map_data: dict[str, pl.Series],
    lf_data: dict[str, pl.Series],
    event_type: str,
) -> None:
    """Asserts that missing key or rename-target columns raise an error."""
    path = mapping_csv(map_data, event_type)
    source = source_lazyframe(event_type, **lf_data)
    with pytest.raises(KeyError):
        replace_mimic4_non_icd_codes(source, path, event_type)
