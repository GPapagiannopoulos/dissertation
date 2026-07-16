"""Contains the identifying the time of diagnosis for specific conditions."""

import polars as pl


def diagnose_hospital_acquired_aki(source: pl.LazyFrame) -> pl.LazyFrame:
    """Identifies admissions where the patient developed AKI.

    This function uses the KDIGO definition criteria to identify
    admissions where the patient developed AKI. It uses the timestamp
    at first laboratory confirmation to create a new record.

    Notes:
        Hospital Acquired AKI (HA-AKI) is defined as having onset
            >=48h after admission. Patients who meet the criteria
            for AKI before that threshold are excluded.
        The definition includes changes to baseline creatinine.
            In the absence of outpatient data for the cohort indicating
            the last healthy kidney function, we follow industry standard
            and set this as the min of the last seven days.
    """
    sorted_labevents = (
        source.filter(pl.col("labevents/label") == "Creatinine")
        .select(
            pl.col("patient_id"),
            pl.col("hadm_id"),
            pl.col("timestamp"),
            pl.col("labevents/valuenum"),
        )
        .sort("hadm_id", "timestamp")
    )

    rolling_48h_min = (
        pl.col("labevents/valuenum")
        .rolling_min_by("timestamp", window_size="48h")
        .over("hadm_id")
    )

    rolling_7d_min = (
        pl.col("labevents/valuenum")
        .rolling_min_by("timestamp", window_size="7d")
        .over("hadm_id")
    )

    result = (
        sorted_labevents.filter(
            pl.any_horizontal(
                pl.col("labevents/valuenum") - rolling_48h_min >= 0.3,
                pl.col("labevents/valuenum") >= rolling_7d_min * 1.5,
            )
        )
        .group_by("hadm_id", maintain_order=True)
        .agg(pl.col("patient_id").first(), pl.col("timestamp").min().alias("timestamp"))
        .with_columns(
            event_type=pl.lit("diagnosis_made"), diagnosis=pl.lit("Acute Kidney Injury")
        )
    )

    return result.select(
        "event_type", "patient_id", "hadm_id", "timestamp", "diagnosis"
    )
