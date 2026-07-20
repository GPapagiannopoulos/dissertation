"""Shared fixtures for the feature engineering test suite."""

import datetime

import polars as pl
import pytest


@pytest.fixture
def labevents_lf():
    """Factory for a real-substrate-shaped hospital-acquired AKI source.

    Mirrors the enriched event substrate: creatinine ``labevents`` rows plus one
    ``admissions`` event row per admission. There is no ``admissions/admittime``
    column in the real frame — admittime is the ``timestamp`` of the admissions
    event, so the gate derives it from those rows. ``admittime`` sets that value,
    either a single value applied to every admission or a ``{hadm_id: admittime}``
    mapping.
    """

    def _build(
        drop: tuple[str, ...] = (),
        hour_step: int = 12,
        admittime: str | dict[str, str] = "2025-01-01 00:00:00",
        **overrides: list,
    ) -> pl.LazyFrame:
        labevents = {
            "patient_id": ["1"] * 14,
            "hadm_id": ["1"] * 14,
            "event_type": ["labevents"] * 14,
            "timestamp": [
                (
                    datetime.datetime(2025, 1, 1, 0)
                    + datetime.timedelta(hours=i * hour_step)
                )
                for i in range(14)
            ],
            "labevents/label": ["Creatinine"] * 14,
            "labevents/valuenum": [1.25] * 14,
            "labevents/valueuom": ["mg/dL"] * 14,
        }
        labevents.update(overrides)
        for col in drop:
            labevents.pop(col, None)
        labevents_frame = pl.LazyFrame(labevents)

        admissions = list(
            dict.fromkeys(
                zip(labevents["patient_id"], labevents["hadm_id"], strict=True)
            )
        )
        admissions_frame = pl.LazyFrame(
            {
                "patient_id": [patient for patient, _ in admissions],
                "hadm_id": [hadm for _, hadm in admissions],
                "event_type": ["admissions"] * len(admissions),
                "timestamp": pl.Series(
                    [
                        admittime[hadm] if isinstance(admittime, dict) else admittime
                        for _, hadm in admissions
                    ],
                    dtype=pl.Datetime,
                ),
            }
        )

        return pl.concat([labevents_frame, admissions_frame], how="diagonal")

    return _build


@pytest.fixture
def chartevents_lf():
    """Factory for creating a valid chartevents LazyFrame with overrides."""

    def _build(
        drop: list[str] | None = None, **overrides: dict[str, list]
    ) -> pl.LazyFrame:
        defaults = {
            "subject_id": ["1"] * 6,
            "hadm_id": ["1"] * 6,
            "stay_id": ["1"] * 6,
            "charttime": [
                datetime.datetime(2025, 1, 1, 0) + datetime.timedelta(hours=i)
                for i in range(6)
            ],
            "itemid": ["226531"] + ["224639"] * 5,
            "valuenum": [199.8] + [90.627756] * 5,
            "valueuom": [""] + ["kg"] * 5,
        }
        defaults.update(**overrides)
        if drop:
            for col in drop:
                defaults.pop(col, None)
        return pl.LazyFrame(defaults)

    return _build


@pytest.fixture
def outputevents_lf():
    """Factory for creating a valid output events LazyFrame with overrides."""

    def _build(**overrides: dict[str, list]) -> pl.LazyFrame:
        defaults = {
            "subject_id": ["1"] * 6,
            "hadm_id": ["1"] * 6,
            "stay_id": ["1"] * 6,
            "charttime": [
                datetime.datetime(2025, 1, 1, 0) + datetime.timedelta(hours=i)
                for i in range(6)
            ],
            "itemid": ["1"] * 6,
            "valuenum": [100.0] * 6,
            "valueuom": ["mL"] * 6,
        }
        defaults.update(**overrides)
        return pl.LazyFrame(defaults)

    return _build


@pytest.fixture
def net_urine_frame():
    """Factory for a net-urine-shaped LazyFrame (a ``net_urine`` output)."""

    def _build(**overrides: dict[str, list]) -> pl.LazyFrame:
        defaults = {
            "subject_id": ["1"] * 4,
            "hadm_id": ["1"] * 4,
            "stay_id": ["1"] * 4,
            "charttime": [
                datetime.datetime(2025, 1, 1, 0) + datetime.timedelta(hours=i)
                for i in range(4)
            ],
            "valuenum": [30.0] * 4,
        }
        defaults.update(**overrides)
        return pl.LazyFrame(defaults)

    return _build


@pytest.fixture
def weight_frame():
    """Factory for a normalized-weight LazyFrame (a ``normalize_weights`` output)."""

    def _build(**overrides: dict[str, list]) -> pl.LazyFrame:
        defaults = {
            "subject_id": ["1"],
            "hadm_id": ["1"],
            "stay_id": ["1"],
            "charttime": [datetime.datetime(2025, 1, 1, 0)],
            "valuenum": [60.0],
        }
        defaults.update(**overrides)
        return pl.LazyFrame(defaults)

    return _build


@pytest.fixture
def uo_arm_frame():
    """Factory for uo rate LazyFrames (calculate_urine_output_rate output frames)."""

    def _build(**overrides: dict[str, list]) -> pl.LazyFrame:
        defaults = {
            "patient_id": ["1"] * 4,
            "hadm_id": ["1"] * 4,
            "stay_id": ["1"] * 4,
            "charttime": [
                datetime.datetime(2025, 1, 1, 0) + datetime.timedelta(hours=i)
                for i in range(4)
            ],
            "rate": [0.2] * 4,
            "window_hours": [6] * 4,
            "n_events": [2] * 4,
        }
        defaults.update(**overrides)
        return pl.LazyFrame(defaults)

    return _build
