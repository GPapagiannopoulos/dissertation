"""Test suite for the helper functions under diagnostic_criteria."""

from collections.abc import Callable

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis.feature_engineering.diagnostic_criteria import (
    diagnose_hospital_acquired_aki,
)

"""
Desired behaviour includes:
1) identifies a patient with increase in serum creatinine by
26micromol/ 0.3mg within 48 hours
2) increase in serum creatinine to >1.5x baseline in the last 7 days
3) Urine volume <0.5mL/kg/hour - labevents/description == Urine Volume, Total
"""


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
    diagnosis = diagnose_hospital_acquired_aki(source)
    assert_frame_equal(diagnosis, expected_lf)
