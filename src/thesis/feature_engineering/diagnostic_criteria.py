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
            We define this as the median measurement <=24h post
            admission.
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
    rolling_min = (
        pl.col("labevents/valuenum")
        .rolling_min_by("timestamp", window_size="48h")
        .over("hadm_id")
    )

    result = (
        sorted_labevents.filter(pl.col("labevents/valuenum") - rolling_min >= 0.3)
        .group_by("hadm_id", maintain_order=True)
        .agg(pl.col("patient_id").first(), pl.col("timestamp").min().alias("timestamp"))
        .with_columns(
            event_type=pl.lit("diagnosis_made"), diagnosis=pl.lit("Acute Kidney Injury")
        )
    )

    return result.select(
        "event_type", "patient_id", "hadm_id", "timestamp", "diagnosis"
    )
