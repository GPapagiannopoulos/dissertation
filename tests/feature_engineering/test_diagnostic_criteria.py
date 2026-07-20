"""Test suite for the helper functions under diagnostic_criteria."""

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.feature_engineering.diagnostic_criteria import (
    diagnose_hospital_acquired_aki,
)


@pytest.mark.parametrize(
    "overrides, expected_lf_data",
    [
        # 0. Absolute increase in 48h.
        (
            {
                "labevents/valuenum": pl.Series(
                    ([1.25] * 12 + [1.75, 1.25]), dtype=pl.Float64
                )
            },
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-07 00:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
            },
        ),
        # 1. Check the window is inclusive at the end
        (
            {"labevents/valuenum": pl.Series(([1.25] * 13 + [1.75]), dtype=pl.Float64)},
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-07 12:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
            },
        ),
        # 2. Sustained elevation emits the earliest confirmation, not later ones.
        (
            {
                "labevents/valuenum": pl.Series(
                    ([1.25] * 6 + [1.75] * 8), dtype=pl.Float64
                )
            },
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-04 00:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
            },
        ),
        # 3. Sub-0.3 increments that net >=0.3 within 48h are detected.
        (
            {
                "labevents/valuenum": pl.Series(
                    ([1.0] * 11 + [1.1, 1.2, 1.35]), dtype=pl.Float64
                )
            },
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-07 12:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
            },
        ),
        # 4. Non-creatinine labs are filtered out and cannot trigger a diagnosis.
        (
            {
                "labevents/label": ["Creatinine"] * 5
                + ["Hemoglobin"]
                + ["Creatinine"] * 8,
                "labevents/valuenum": pl.Series(
                    ([1.25] * 5 + [5.0] + [1.25] * 6 + [1.75, 1.25]),
                    dtype=pl.Float64,
                ),
            },
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-07 00:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
            },
        ),
        # 5. Independent admissions each yield their own diagnosis row.
        (
            {
                "patient_id": ["1"] * 7 + ["2"] * 7,
                "hadm_id": ["1"] * 7 + ["2"] * 7,
                "labevents/valuenum": pl.Series(
                    (
                        [1.25, 1.25, 1.25, 1.25, 1.25, 1.7, 1.25]
                        + [1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.0]
                    ),
                    dtype=pl.Float64,
                ),
            },
            {
                "event_type": pl.Series(
                    ["diagnosis_made", "diagnosis_made"], dtype=pl.String
                ),
                "patient_id": pl.Series(["1", "2"], dtype=pl.String),
                "hadm_id": pl.Series(["1", "2"], dtype=pl.String),
                "timestamp": pl.Series(
                    ["2025-01-03 12:00:00", "2025-01-07 00:00:00"], dtype=pl.Datetime
                ),
                "diagnosis": pl.Series(
                    ["Acute Kidney Injury", "Acute Kidney Injury"], dtype=pl.String
                ),
            },
        ),
        # 6. A sub 0.3 rise returns an empty lazyframe
        (
            {},
            {
                "event_type": pl.Series([], dtype=pl.String),
                "patient_id": pl.Series([], dtype=pl.String),
                "hadm_id": pl.Series([], dtype=pl.String),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "diagnosis": pl.Series([], dtype=pl.String),
            },
        ),
        # 7. A rise over more than 48h returns an empty lazyframe
        (
            {"labevents/valuenum": pl.Series([1.25 + 0.025 * i for i in range(14)])},
            {
                "event_type": pl.Series([], dtype=pl.String),
                "patient_id": pl.Series([], dtype=pl.String),
                "hadm_id": pl.Series([], dtype=pl.String),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "diagnosis": pl.Series([], dtype=pl.String),
            },
        ),
        # 8. If meets criteria in first 48h post admission, then not HA
        (
            {
                "labevents/valuenum": pl.Series(
                    [1.25, 2.00] + [1.25] * 12, dtype=pl.Float64
                )
            },
            {
                "event_type": pl.Series([], dtype=pl.String),
                "patient_id": pl.Series([], dtype=pl.String),
                "hadm_id": pl.Series([], dtype=pl.String),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "diagnosis": pl.Series([], dtype=pl.String),
            },
        ),
    ],
)
def test_diagnose_ha_aki_criterion_one(
    labevents_lf: Callable,
    overrides: dict[str, pl.Series],
    expected_lf_data: dict[str, pl.Series],
) -> None:
    """Asserts normal behaviour of criterion one for aki diagnosis."""
    source = labevents_lf(**overrides)
    expected_lf = pl.LazyFrame(expected_lf_data)
    null_uo_data = pl.LazyFrame(
        {
            "patient_id": pl.Series([None], dtype=pl.String),
            "hadm_id": pl.Series([None], dtype=pl.String),
            "stay_id": pl.Series([None], dtype=pl.String),
            "charttime": pl.Series([None], dtype=pl.Datetime),
            "rate": pl.Series([None], dtype=pl.Float64),
            "window_hours": pl.Series([None], dtype=pl.Int16),
            "n_events": pl.Series([None], dtype=pl.Int16),
        }
    )
    diagnosis = diagnose_hospital_acquired_aki(source, null_uo_data)
    assert_frame_equal(diagnosis, expected_lf, check_row_order=False)


@pytest.mark.parametrize(
    "hour_step, overrides, expected_lf_data",
    [
        # 0. Fires on a 1.5x rise off the 7d-min baseline that criterion one
        #    misses (the low sits >48h before the rise).
        (
            12,
            {
                "labevents/valuenum": pl.Series(
                    [0.85] * 3 + [1.10] * 10 + [1.30], dtype=pl.Float64
                ),
            },
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-07 12:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
            },
        ),
        # 1. With <7d of data the partial window still references the running
        #    min, so the rise is detected.
        (
            6,
            {
                "labevents/valuenum": pl.Series(
                    [0.80] * 2 + [1.00] * 11 + [1.25], dtype=pl.Float64
                ),
            },
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-04 06:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
            },
        ),
        # 2. A low older than 7d has expired from the window and is not used as
        #    baseline, so no 1.5x rise is registered (empty).
        (
            16,
            {
                "labevents/valuenum": pl.Series(
                    [0.80] + [1.00] * 12 + [1.25], dtype=pl.Float64
                ),
            },
            {
                "event_type": pl.Series([], dtype=pl.String),
                "patient_id": pl.Series([], dtype=pl.String),
                "hadm_id": pl.Series([], dtype=pl.String),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "diagnosis": pl.Series([], dtype=pl.String),
            },
        ),
        # 3. A value just above 1.5x baseline (ratio 1.5125) fires.
        (
            12,
            {
                "labevents/valuenum": pl.Series(
                    [0.80] * 2 + [1.00] * 11 + [1.21], dtype=pl.Float64
                ),
            },
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-07 12:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
            },
        ),
        # 4. A value just below 1.5x baseline (ratio 1.4875) does not fire.
        (
            12,
            {
                "labevents/valuenum": pl.Series(
                    [0.80] * 2 + [1.00] * 11 + [1.19], dtype=pl.Float64
                ),
            },
            {
                "event_type": pl.Series([], dtype=pl.String),
                "patient_id": pl.Series([], dtype=pl.String),
                "hadm_id": pl.Series([], dtype=pl.String),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "diagnosis": pl.Series([], dtype=pl.String),
            },
        ),
    ],
)
def test_diagnose_ha_aki_criterion_two(
    labevents_lf: Callable,
    hour_step: int,
    overrides: dict[str, pl.Series],
    expected_lf_data: dict[str, pl.Series],
) -> None:
    """Asserts normal behaviour for the second criterion."""
    source = labevents_lf(hour_step=hour_step, **overrides)
    expected_lf = pl.LazyFrame(expected_lf_data)
    null_uo_data = pl.LazyFrame(
        {
            "patient_id": pl.Series([None], dtype=pl.String),
            "hadm_id": pl.Series([None], dtype=pl.String),
            "stay_id": pl.Series([None], dtype=pl.String),
            "charttime": pl.Series([None], dtype=pl.Datetime),
            "rate": pl.Series([None], dtype=pl.Float64),
            "window_hours": pl.Series([None], dtype=pl.Int16),
            "n_events": pl.Series([None], dtype=pl.Int16),
        }
    )
    diagnosis = diagnose_hospital_acquired_aki(source, null_uo_data)
    assert_frame_equal(diagnosis, expected_lf, check_row_order=False)


@pytest.mark.parametrize(
    "overrides, expected_lf_data",
    [
        # 0. Single patient, admission, correct rate and window
        (
            {},
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-01 00:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
            },
        ),
        # 1. Rate is too high
        (
            {"rate": pl.Series([0.5] * 4, dtype=pl.Float64)},
            {
                "event_type": pl.Series([], dtype=pl.String),
                "patient_id": pl.Series([], dtype=pl.String),
                "hadm_id": pl.Series([], dtype=pl.String),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "diagnosis": pl.Series([], dtype=pl.String),
            },
        ),
        # 2. Rate is negative
        (
            {"rate": pl.Series([-0.1] * 4, dtype=pl.Float64)},
            {
                "event_type": pl.Series([], dtype=pl.String),
                "patient_id": pl.Series([], dtype=pl.String),
                "hadm_id": pl.Series([], dtype=pl.String),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "diagnosis": pl.Series([], dtype=pl.String),
            },
        ),
        # 3. Time window is too small
        (
            {"window_hours": pl.Series([4] * 4)},
            {
                "event_type": pl.Series([], dtype=pl.String),
                "patient_id": pl.Series([], dtype=pl.String),
                "hadm_id": pl.Series([], dtype=pl.String),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "diagnosis": pl.Series([], dtype=pl.String),
            },
        ),
        # 4. Select the first entry to fullfill the criteria
    ],
)
def test_diagnose_ha_aki_criterion_three(
    uo_arm_frame: Callable,
    overrides: dict[str, pl.Series],
    expected_lf_data: dict[str, pl.Series],
) -> None:
    """Asserts normal behaviour for the third criterion."""
    source = uo_arm_frame(**overrides)
    expected_lf = pl.LazyFrame(expected_lf_data)
    null_cr_data = pl.LazyFrame(
        {
            "patient_id": pl.Series(["1"], dtype=pl.String),
            "hadm_id": pl.Series(["1"], dtype=pl.String),
            "event_type": pl.Series([None], dtype=pl.String),
            "timestamp": pl.Series([None], dtype=pl.Datetime),
            "labevents/label": pl.Series([None], dtype=pl.String),
            "labevents/valuenum": pl.Series([None], dtype=pl.Float64),
            "labevents/valueuom": pl.Series([None], dtype=pl.String),
            "admissions/admittime": pl.Series(
                ["2024-12-28 00:00:00"],  # ensuring the gate doesn't apply
                dtype=pl.Datetime,
            ),
        }
    )
    diagnosis = diagnose_hospital_acquired_aki(null_cr_data, source)
    assert_frame_equal(diagnosis, expected_lf, check_row_order=False)


@pytest.mark.parametrize(
    "overrides, expected_lf_data",
    [
        # 0. Onset >48h after admission is hospital-acquired and kept.
        (
            {
                "labevents/valuenum": pl.Series(
                    [1.25] * 12 + [1.75, 1.25], dtype=pl.Float64
                ),
                "admissions/admittime": pl.Series(
                    ["2025-01-04 18:00:00"] * 14, dtype=pl.Datetime
                ),
            },
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-07 00:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
            },
        ),
        # 1. Onset exactly 48h after admission is excluded (the gate is strict).
        (
            {
                "labevents/valuenum": pl.Series(
                    [1.25] * 12 + [1.75, 1.25], dtype=pl.Float64
                ),
                "admissions/admittime": pl.Series(
                    ["2025-01-05 00:00:00"] * 14, dtype=pl.Datetime
                ),
            },
            {
                "event_type": pl.Series([], dtype=pl.String),
                "patient_id": pl.Series([], dtype=pl.String),
                "hadm_id": pl.Series([], dtype=pl.String),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "diagnosis": pl.Series([], dtype=pl.String),
            },
        ),
        # 2. Onset within 48h of admission (community / present-on-admission)
        #    excludes the whole admission.
        (
            {
                "labevents/valuenum": pl.Series(
                    [1.25] * 12 + [1.75, 1.25], dtype=pl.Float64
                ),
                "admissions/admittime": pl.Series(
                    ["2025-01-06 00:00:00"] * 14, dtype=pl.Datetime
                ),
            },
            {
                "event_type": pl.Series([], dtype=pl.String),
                "patient_id": pl.Series([], dtype=pl.String),
                "hadm_id": pl.Series([], dtype=pl.String),
                "timestamp": pl.Series([], dtype=pl.Datetime),
                "diagnosis": pl.Series([], dtype=pl.String),
            },
        ),
        # 3. Admittime is applied per hadm_id: one admission clears the gate,
        #    the other is excluded.
        (
            {
                "patient_id": ["1"] * 7 + ["2"] * 7,
                "hadm_id": ["1"] * 7 + ["2"] * 7,
                "labevents/valuenum": pl.Series(
                    [1.25, 1.25, 1.25, 1.25, 1.25, 1.70, 1.25]
                    + [1.25] * 5
                    + [1.75, 1.25],
                    dtype=pl.Float64,
                ),
                "admissions/admittime": pl.Series(
                    ["2025-01-02 00:00:00"] * 7 + ["2025-01-01 00:00:00"] * 7,
                    dtype=pl.Datetime,
                ),
            },
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["2"], dtype=pl.String),
                "hadm_id": pl.Series(["2"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-07 00:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
            },
        ),
    ],
)
def test_diagnose_ha_aki_gate(
    labevents_lf: Callable,
    overrides: dict[str, pl.Series],
    expected_lf_data: dict[str, pl.Series],
) -> None:
    """Asserts the >=48h-post-admission gate for hospital-acquired AKI."""
    source = labevents_lf(**overrides)
    expected_lf = pl.LazyFrame(expected_lf_data)
    null_uo_data = pl.LazyFrame(
        {
            "patient_id": pl.Series([None], dtype=pl.String),
            "hadm_id": pl.Series([None], dtype=pl.String),
            "stay_id": pl.Series([None], dtype=pl.String),
            "charttime": pl.Series([None], dtype=pl.Datetime),
            "rate": pl.Series([None], dtype=pl.Float64),
            "window_hours": pl.Series([None], dtype=pl.Int16),
            "n_events": pl.Series([None], dtype=pl.Int16),
        }
    )
    diagnosis = diagnose_hospital_acquired_aki(source, null_uo_data)
    assert_frame_equal(diagnosis, expected_lf, check_row_order=False)
