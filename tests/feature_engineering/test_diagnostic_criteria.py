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
        # 0. Fires on a 1.5x rise off the admission baseline that criterion one
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
        # 1. The baseline is drawn from the values inside the 24h admission
        #    window, so a low there is what the later rise is measured against.
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
        # 2. The admission baseline never expires. A low drawn on arrival is
        #    still the comparator eight days later, where the old 7d rolling
        #    minimum would have aged it out and registered nothing.
        (
            16,
            {
                "labevents/valuenum": pl.Series(
                    [0.80] + [1.00] * 12 + [1.25], dtype=pl.Float64
                ),
            },
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-09 16:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
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
        # 0. Recovery after arrival does not lower the baseline. The patient
        #    arrives at 1.30, falls to 0.85 and climbs back to 1.30. Measured
        #    against the arrival value nothing happened; measured against a
        #    rolling minimum the climb back would have registered as new injury.
        #    This is the whole point of pinning the baseline at admission.
        (
            {
                "labevents/valuenum": pl.Series(
                    [1.30] * 3
                    + [
                        0.85,
                        0.85,
                        0.90,
                        0.95,
                        1.00,
                        1.05,
                        1.10,
                        1.15,
                        1.20,
                        1.25,
                        1.30,
                    ],
                    dtype=pl.Float64,
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
        # 1. No creatinine inside the 24h admission window leaves the baseline
        #    null, so the ratio arm cannot fire at all. The rise here would clear
        #    1.5x any of the observed values, and still nothing is emitted.
        (
            {
                "admittime": "2024-12-25 00:00:00",
                "labevents/valuenum": pl.Series(
                    [1.00 + 0.05 * i for i in range(14)], dtype=pl.Float64
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
def test_diagnose_ha_aki_admission_baseline(
    labevents_lf: Callable,
    null_uo_data: pl.LazyFrame,
    overrides: dict[str, pl.Series],
    expected_lf_data: dict[str, pl.Series],
) -> None:
    """Asserts the baseline is the admission creatinine, not a rolling minimum."""
    source = labevents_lf(**overrides)
    expected_lf = pl.LazyFrame(expected_lf_data)
    diagnosis = diagnose_hospital_acquired_aki(source, null_uo_data)
    assert_frame_equal(diagnosis, expected_lf, check_row_order=False)


def test_diagnose_ha_aki_baseline_window_is_inclusive(
    labevents_lf: Callable, null_uo_data: pl.LazyFrame
) -> None:
    """Asserts a draw landing exactly on admittime+24h still sets the baseline.

    Admission is at 2024-12-31 00:00 and the first creatinine at 2025-01-01
    00:00, exactly 24h later. If that draw were excluded the baseline would be
    null and nothing would be emitted, so a non-empty result is the assertion.
    """
    source = labevents_lf(
        hour_step=24,
        admittime="2024-12-31 00:00:00",
        **{
            "labevents/valuenum": pl.Series(
                [0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35]
                + [1.40] * 4,
                dtype=pl.Float64,
            )
        },
    )
    expected_lf = pl.LazyFrame(
        {
            "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
            "patient_id": pl.Series(["1"], dtype=pl.String),
            "hadm_id": pl.Series(["1"], dtype=pl.String),
            "timestamp": pl.Series(["2025-01-10 00:00:00"], dtype=pl.Datetime),
            "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
        }
    )
    diagnosis = diagnose_hospital_acquired_aki(source, null_uo_data)
    assert_frame_equal(diagnosis, expected_lf, check_row_order=False)


def test_diagnose_ha_aki_baseline_is_earliest_across_many_admissions(
    labevents_lf: Callable, null_uo_data: pl.LazyFrame
) -> None:
    """Asserts the baseline is the earliest draw in the window, not an arbitrary one.

    Each admission opens at 1.00 and dips to 0.60 twenty-four hours later; both
    sit inside the baseline window. Against the arrival value (threshold 1.50)
    onset is the 1.50 on day eleven. Against the dip (threshold 0.90) it would be
    the 0.90 on day four, so the onset timestamp separates the two readings.

    Two hundred admissions are used because the failure is order-dependent: a
    Polars join returns rows in hash-probe order, and taking the window's first
    *row* rather than its earliest *timestamp* is only wrong for some orderings.
    """
    source = labevents_lf(
        hour_step=24,
        n_admissions=200,
        **{
            "labevents/valuenum": pl.Series(
                [1.00, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.45]
                + [1.50] * 3,
                dtype=pl.Float64,
            )
        },
    )
    admissions = [str(admission) for admission in range(1, 201)]
    expected_lf = pl.LazyFrame(
        {
            "event_type": pl.Series(["diagnosis_made"] * 200, dtype=pl.String),
            "patient_id": pl.Series(admissions, dtype=pl.String),
            "hadm_id": pl.Series(admissions, dtype=pl.String),
            "timestamp": pl.Series(["2025-01-12 00:00:00"] * 200, dtype=pl.Datetime),
            "diagnosis": pl.Series(["Acute Kidney Injury"] * 200, dtype=pl.String),
        }
    )
    diagnosis = diagnose_hospital_acquired_aki(source, null_uo_data)
    assert_frame_equal(diagnosis, expected_lf, check_row_order=False)


def test_diagnose_ha_aki_rolling_window_holds_across_many_admissions(
    labevents_lf: Callable, null_uo_data: pl.LazyFrame
) -> None:
    """Asserts the 48h absolute-rise arm survives the baseline join.

    ``rolling_min_by`` assumes its ``by`` column is sorted and returns wrong
    answers silently when it is not. Every admission steps 1.00 -> 1.40 at day
    five, a 0.40 rise inside 48h, so all two hundred must be diagnosed there.
    """
    source = labevents_lf(
        n_admissions=200,
        **{"labevents/valuenum": pl.Series([1.00] * 10 + [1.40] * 4, dtype=pl.Float64)},
    )
    admissions = [str(admission) for admission in range(1, 201)]
    expected_lf = pl.LazyFrame(
        {
            "event_type": pl.Series(["diagnosis_made"] * 200, dtype=pl.String),
            "patient_id": pl.Series(admissions, dtype=pl.String),
            "hadm_id": pl.Series(admissions, dtype=pl.String),
            "timestamp": pl.Series(["2025-01-06 00:00:00"] * 200, dtype=pl.Datetime),
            "diagnosis": pl.Series(["Acute Kidney Injury"] * 200, dtype=pl.String),
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
        # 4. Selects the first entry to fulfill the criteria
        (
            {"rate": pl.Series([0.5] * 3 + [0.2], dtype=pl.Float64)},
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-01 03:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
            },
        ),
        # 5. Groups by patient_id
        (
            {"patient_id": pl.Series(["1"] * 2 + ["2"] * 2)},
            {
                "event_type": pl.Series(["diagnosis_made"] * 2, dtype=pl.String),
                "patient_id": pl.Series(["1", "2"], dtype=pl.String),
                "hadm_id": pl.Series(["1"] * 2, dtype=pl.String),
                "timestamp": pl.Series(
                    ["2025-01-01 00:00:00", "2025-01-01 02:00:00"], dtype=pl.Datetime
                ),
                "diagnosis": pl.Series(["Acute Kidney Injury"] * 2, dtype=pl.String),
            },
        ),
        # 6 Groups by hadm_id
        (
            {"hadm_id": pl.Series(["1"] * 2 + ["2"] * 2)},
            {
                "event_type": pl.Series(["diagnosis_made"] * 2, dtype=pl.String),
                "patient_id": pl.Series(["1"] * 2, dtype=pl.String),
                "hadm_id": pl.Series(["1", "2"], dtype=pl.String),
                "timestamp": pl.Series(
                    ["2025-01-01 00:00:00", "2025-01-01 02:00:00"], dtype=pl.Datetime
                ),
                "diagnosis": pl.Series(["Acute Kidney Injury"] * 2, dtype=pl.String),
            },
        ),
        # 7. 'None' values are excluded
        (
            {
                "patient_id": pl.Series([None] * 2 + ["2"] * 2, dtype=pl.String),
                "hadm_id": pl.Series([None] * 2 + ["1"] * 2),
            },
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["2"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-01 02:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
            },
        ),
        # 8. Rate is exactly '0' (testing for anuric aki)
        (
            {"rate": pl.Series([0] * 4, dtype=pl.Float64)},
            {
                "event_type": pl.Series(["diagnosis_made"], dtype=pl.String),
                "patient_id": pl.Series(["1"], dtype=pl.String),
                "hadm_id": pl.Series(["1"], dtype=pl.String),
                "timestamp": pl.Series(["2025-01-01 00:00:00"], dtype=pl.Datetime),
                "diagnosis": pl.Series(["Acute Kidney Injury"], dtype=pl.String),
            },
        ),
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
            "patient_id": pl.Series(["1", "2"], dtype=pl.String),
            "hadm_id": pl.Series(["1", "2"], dtype=pl.String),
            # admissions rows supply admittime as their timestamp; an early value
            # keeps the >=48h gate from excluding the urine-output onsets.
            "event_type": pl.Series(["admissions", "admissions"], dtype=pl.String),
            "timestamp": pl.Series(
                ["2024-12-28 00:00:00", "2024-12-28 00:00:00"], dtype=pl.Datetime
            ),
            "labevents/label": pl.Series([None, None], dtype=pl.String),
            "labevents/valuenum": pl.Series([None, None], dtype=pl.Float64),
            "labevents/valueuom": pl.Series([None, None], dtype=pl.String),
        }
    )
    diagnosis = diagnose_hospital_acquired_aki(null_cr_data, source)
    assert_frame_equal(diagnosis, expected_lf, check_row_order=False)


@pytest.mark.parametrize(
    "cr_overrides, uo_overrides, expected_lf_data",
    [
        # 0. Both arms fire on the same admission and uo is earlier
        (
            {"labevents/valuenum": pl.Series([1.25] * 8 + [1.75] + [1.25] * 5)},
            {
                "charttime": pl.Series(
                    [
                        "2025-01-04 00:00:00",
                        "2025-01-04 01:00:00",
                        "2025-01-04 02:00:00",
                        "2025-01-04 03:00:00",
                    ],
                    dtype=pl.Datetime,
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
        # 1. Both arms fire on the same admission and cr_arm is earlier
        (
            {"labevents/valuenum": pl.Series([1.25] * 6 + [1.75] + [1.25] * 7)},
            {
                "charttime": pl.Series(
                    [
                        "2025-01-05 00:00:00",
                        "2025-01-05 01:00:00",
                        "2025-01-05 02:00:00",
                        "2025-01-05 03:00:00",
                    ],
                    dtype=pl.Datetime,
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
        # 2. Outer join preserves arms independently
        (
            {
                "patient_id": ["1"] * 7 + ["2"] * 7,
                "hadm_id": ["1"] * 7 + ["2"] * 7,
                "labevents/valuenum": pl.Series([1.25] * 5 + [1.75, 1.25] + [1.25] * 7),
            },
            {
                "patient_id": ["2"] * 4,
                "hadm_id": ["2"] * 4,
                "charttime": pl.Series(
                    [
                        "2025-01-04 00:00:00",
                        "2025-01-04 01:00:00",
                        "2025-01-04 02:00:00",
                        "2025-01-04 03:00:00",
                    ],
                    dtype=pl.Datetime,
                ),
            },
            {
                "event_type": pl.Series(
                    ["diagnosis_made", "diagnosis_made"], dtype=pl.String
                ),
                "patient_id": pl.Series(["1", "2"], dtype=pl.String),
                "hadm_id": pl.Series(["1", "2"], dtype=pl.String),
                "timestamp": pl.Series(
                    ["2025-01-03 12:00:00", "2025-01-04 00:00:00"], dtype=pl.Datetime
                ),
                "diagnosis": pl.Series(
                    ["Acute Kidney Injury", "Acute Kidney Injury"], dtype=pl.String
                ),
            },
        ),
        # 3. Early meeting of criteria excludes record even if we have a
        # later fulfillment
        (
            {"labevents/valuenum": pl.Series([1.25, 1.25, 1.75] + [1.25] * 11)},
            {
                "charttime": pl.Series(
                    [
                        "2025-01-04 00:00:00",
                        "2025-01-04 01:00:00",
                        "2025-01-04 02:00:00",
                        "2025-01-04 03:00:00",
                    ],
                    dtype=pl.Datetime,
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
def test_diagnose_ha_aki_composition(
    labevents_lf: Callable,
    uo_arm_frame: Callable,
    cr_overrides: dict[str, pl.Series],
    uo_overrides: dict[str, pl.Series],
    expected_lf_data: dict[str, pl.Series],
) -> None:
    """Asserts the two arms merge (full-outer, min onset) then gate together."""
    source = labevents_lf(**cr_overrides)
    uo_data = uo_arm_frame(**uo_overrides)
    expected_lf = pl.LazyFrame(expected_lf_data)
    diagnosis = diagnose_hospital_acquired_aki(source, uo_data)
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
                "admittime": "2025-01-04 18:00:00",
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
                "admittime": "2025-01-05 00:00:00",
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
                "admittime": "2025-01-06 00:00:00",
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
                "admittime": {
                    "1": "2025-01-02 00:00:00",
                    "2": "2025-01-01 00:00:00",
                },
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
