"""Testing suite for diagnostic algorithm validation pipeline."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.feature_engineering.validation import (
    aki_ground_truth,
    evaluable_admissions,
)


@pytest.mark.parametrize(
    "mock_data, expected_ids",
    [
        # 0. Correctly identifies v9 codes
        (
            {
                "event_type": pl.Series(["diagnoses_icd"], dtype=pl.String),
                "diagnoses_icd/icd_version": pl.Series(["9"], dtype=pl.String),
                "diagnoses_icd/icd_code": pl.Series(["5841"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
            },
            ["1"],
        ),
        # 1. Correctly identifies v10 codes
        (
            {
                "event_type": pl.Series(["diagnoses_icd"], dtype=pl.String),
                "diagnoses_icd/icd_version": pl.Series(["10"], dtype=pl.String),
                "diagnoses_icd/icd_code": pl.Series(["N170"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
            },
            ["1"],
        ),
        # 2. Correctly doesn't identify mismatched version and code pairs
        (
            {
                "event_type": pl.Series(["diagnoses_icd"] * 2, dtype=pl.String),
                "diagnoses_icd/icd_version": pl.Series(["10", "9"], dtype=pl.String),
                "diagnoses_icd/icd_code": pl.Series(["5841", "N170"], dtype=pl.String),
                "hadm_id": pl.Series(["1", "2"], dtype=pl.String),
            },
            [],
        ),
        # 3. Returns multiple hadm_ids
        (
            {
                "event_type": pl.Series(["diagnoses_icd"] * 2, dtype=pl.String),
                "diagnoses_icd/icd_version": pl.Series(["9", "10"], dtype=pl.String),
                "diagnoses_icd/icd_code": pl.Series(["5841", "N170"], dtype=pl.String),
                "hadm_id": pl.Series(["1", "2"], dtype=pl.String),
            },
            ["1", "2"],
        ),
        # 4. Doesn't return duplicates
        (
            {
                "event_type": pl.Series(["diagnoses_icd"] * 2, dtype=pl.String),
                "diagnoses_icd/icd_version": pl.Series(["9", "10"], dtype=pl.String),
                "diagnoses_icd/icd_code": pl.Series(["5841", "N170"], dtype=pl.String),
                "hadm_id": pl.Series(["1", "1"], dtype=pl.String),
            },
            ["1"],
        ),
        # 5. Doesn't return 'None' values
        (
            {
                "event_type": pl.Series(["diagnoses_icd"] * 2, dtype=pl.String),
                "diagnoses_icd/icd_version": pl.Series(["9", "10"], dtype=pl.String),
                "diagnoses_icd/icd_code": pl.Series(["5841", "N170"], dtype=pl.String),
                "hadm_id": pl.Series(["1", None], dtype=pl.String),
            },
            ["1"],
        ),
    ],
)
def test_aki_ground_truth_happy_path(
    mock_data: dict[str, pl.Series], expected_ids: list[str]
) -> None:
    """Asserts normal behaviour for aki_ground_truth."""
    source = pl.LazyFrame(mock_data)
    expected = pl.DataFrame({"hadm_id": pl.Series(expected_ids, dtype=pl.String)})
    assert_frame_equal(aki_ground_truth(source).collect(), expected)


@pytest.mark.parametrize(
    "source_data, uo_ids, expected_ids",
    [
        # 0. Creatinine arm alone makes an admission evaluable
        (
            {
                "labevents/label": pl.Series(["Creatinine"], dtype=pl.String),
                "labevents/valuenum": pl.Series([1.0], dtype=pl.Float64),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
            },
            [],
            ["1"],
        ),
        # 1. UO arm alone makes an admission evaluable (non-creatinine labs ignored)
        (
            {
                "labevents/label": pl.Series(["Glucose"], dtype=pl.String),
                "labevents/valuenum": pl.Series([5.0], dtype=pl.Float64),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
            },
            ["2"],
            ["2"],
        ),
        # 2. Union of both arms, de-duplicated on overlap
        (
            {
                "labevents/label": pl.Series(["Creatinine"] * 2, dtype=pl.String),
                "labevents/valuenum": pl.Series([1.0, 1.0], dtype=pl.Float64),
                "hadm_id": pl.Series(["1", "2"], dtype=pl.String),
            },
            ["2", "3"],
            ["1", "2", "3"],
        ),
        # 3. A null-valuenum Creatinine row is not a measurement
        (
            {
                "labevents/label": pl.Series(["Creatinine"], dtype=pl.String),
                "labevents/valuenum": pl.Series([None], dtype=pl.Float64),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
            },
            [],
            [],
        ),
        # 4. Null hadm_ids are dropped from both arms
        (
            {
                "labevents/label": pl.Series(["Creatinine"], dtype=pl.String),
                "labevents/valuenum": pl.Series([1.0], dtype=pl.Float64),
                "hadm_id": pl.Series([None], dtype=pl.String),
            },
            [None],
            [],
        ),
    ],
)
def test_evaluable_admissions_happy_path(
    source_data: dict[str, pl.Series],
    uo_ids: list[str | None],
    expected_ids: list[str],
) -> None:
    """Asserts the evaluable cohort is the union of the creatinine and UO arms."""
    source = pl.LazyFrame(source_data)
    uo_data = pl.LazyFrame({"hadm_id": pl.Series(uo_ids, dtype=pl.String)})
    expected = pl.DataFrame({"hadm_id": pl.Series(expected_ids, dtype=pl.String)})
    assert_frame_equal(evaluable_admissions(source, uo_data).collect(), expected)
