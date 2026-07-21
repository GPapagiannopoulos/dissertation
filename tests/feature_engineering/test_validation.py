"""Testing suite for diagnostic algorithm validation pipeline."""

import polars as pl
import pytest
from polars.testing import assert_series_equal

from thesis.feature_engineering.validation import aki_ground_truth


@pytest.mark.parametrize(
    "mock_data, expected_series",
    [
        # 0. Correctly identifies v9 codes
        (
            {
                "event_type": pl.Series(["diagnoses_icd"], dtype=pl.String),
                "diagnoses_icd/icd_version": pl.Series(["9"], dtype=pl.String),
                "diagnoses_icd/icd_code": pl.Series(["5841"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
            },
            pl.Series("hadm_id", ["1"], dtype=pl.String),
        )
    ],
)
def test_aki_ground_truth_happy_path(
    mock_data: dict[str, pl.Series], expected_series: pl.Series
) -> None:
    """Asserts normal behaviour for aki_ground_truth_helper."""
    source = pl.LazyFrame(mock_data)
    assert_series_equal(aki_ground_truth(source), expected_series)
