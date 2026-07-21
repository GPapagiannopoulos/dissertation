"""Testing suite for diagnostic algorithm validation pipeline."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.feature_engineering.validation import (
    aki_ground_truth,
    confusion_matrix,
    evaluable_admissions,
    metrics,
)

_METRIC_ORDER = [
    "sensitivity",
    "specificity",
    "precision",
    "npv",
    "f1",
    "accuracy",
    "prevalence",
]


def _id_lf(ids: list[str]) -> pl.LazyFrame:
    """Builds a single-column ``hadm_id`` LazyFrame for the set-op inputs."""
    return pl.LazyFrame({"hadm_id": pl.Series(ids, dtype=pl.String)})


def _cm(tp: int, fp: int, fn: int, tn: int) -> pl.DataFrame:
    """Builds a 2x2 confusion matrix in the ``confusion_matrix`` output shape."""
    return pl.DataFrame(
        {
            "predicted": pl.Series(["positive", "negative"], dtype=pl.String),
            "actual_positive": pl.Series([tp, fn], dtype=pl.UInt32),
            "actual_negative": pl.Series([fp, tn], dtype=pl.UInt32),
        }
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


@pytest.mark.parametrize(
    "predicted_ids, actual_ids, evaluable_ids, tp, fp, fn, tn",
    [
        # 0. A mix of every cell
        (["1", "2"], ["1", "3"], ["1", "2", "3", "4", "5"], 1, 1, 1, 2),
        # 1. An ICD-positive admission outside the cohort is out of scope, not a FN
        (["1"], ["1", "3"], ["1", "2"], 1, 0, 0, 1),
        # 2. A predicted admission outside the cohort is ignored, not a FP
        (["1", "9"], ["1"], ["1", "2"], 1, 0, 0, 1),
        # 3. Nothing predicted, nothing positive -> all true negatives
        ([], [], ["1", "2", "3"], 0, 0, 0, 3),
        # 4. Perfect agreement
        (["1", "2"], ["1", "2"], ["1", "2"], 2, 0, 0, 0),
        # 5. Every prediction is a false positive
        (["1", "2"], [], ["1", "2"], 0, 2, 0, 0),
        # 6. Empty cohort collapses every cell to zero
        (["1"], ["1"], [], 0, 0, 0, 0),
    ],
)
def test_confusion_matrix(
    predicted_ids: list[str],
    actual_ids: list[str],
    evaluable_ids: list[str],
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> None:
    """Asserts the 2x2 matrix cells over the evaluable cohort."""
    expected = pl.DataFrame(
        {
            "predicted": pl.Series(["positive", "negative"], dtype=pl.String),
            "actual_positive": pl.Series([tp, fn], dtype=pl.UInt32),
            "actual_negative": pl.Series([fp, tn], dtype=pl.UInt32),
        }
    )
    result = confusion_matrix(
        _id_lf(predicted_ids), _id_lf(actual_ids), _id_lf(evaluable_ids)
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "tp, fp, fn, tn, expected_values",
    [
        # 0. A normal mix (total = 10)
        (
            2,
            1,
            1,
            6,
            {
                "sensitivity": 2 / 3,
                "specificity": 6 / 7,
                "precision": 2 / 3,
                "npv": 6 / 7,
                "f1": 4 / 6,
                "accuracy": 0.8,
                "prevalence": 0.3,
            },
        ),
        # 1. Perfect agreement
        (
            5,
            0,
            0,
            5,
            {
                "sensitivity": 1.0,
                "specificity": 1.0,
                "precision": 1.0,
                "npv": 1.0,
                "f1": 1.0,
                "accuracy": 1.0,
                "prevalence": 0.5,
            },
        ),
        # 2. No positives at all -> positive-side metrics undefined
        (
            0,
            0,
            0,
            5,
            {
                "sensitivity": None,
                "specificity": 1.0,
                "precision": None,
                "npv": 1.0,
                "f1": None,
                "accuracy": 1.0,
                "prevalence": 0.0,
            },
        ),
        # 3. No negatives at all -> negative-side metrics undefined
        (
            3,
            0,
            0,
            0,
            {
                "sensitivity": 1.0,
                "specificity": None,
                "precision": 1.0,
                "npv": None,
                "f1": 1.0,
                "accuracy": 1.0,
                "prevalence": 1.0,
            },
        ),
        # 4. Empty cohort -> every metric undefined
        (
            0,
            0,
            0,
            0,
            {
                "sensitivity": None,
                "specificity": None,
                "precision": None,
                "npv": None,
                "f1": None,
                "accuracy": None,
                "prevalence": None,
            },
        ),
    ],
)
def test_metrics(
    tp: int,
    fp: int,
    fn: int,
    tn: int,
    expected_values: dict[str, float | None],
) -> None:
    """Asserts metric arithmetic and null-on-zero-denominator behaviour."""
    expected = pl.DataFrame(
        {
            "metric": pl.Series(_METRIC_ORDER, dtype=pl.String),
            "value": pl.Series(
                [expected_values[m] for m in _METRIC_ORDER], dtype=pl.Float64
            ),
        }
    )
    assert_frame_equal(metrics(_cm(tp, fp, fn, tn)), expected)
