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

    marker_baseline_added = sorted_labevents.with_columns(
        _start=pl.col("timestamp").min().over("hadm_id")
    ).with_columns(
        baseline=(
            pl.col("labevents/valuenum")
            .filter(pl.col("timestamp") <= pl.col("_start") + pl.duration(hours=24))
            .median()
            .over("hadm_id")
        )
    )

    rolling_min = (
        pl.col("labevents/valuenum")
        .rolling_min_by("timestamp", window_size="48h")
        .over("hadm_id")
    )

    result = (
        marker_baseline_added.filter(
            pl.any_horizontal(
                pl.col("labevents/valuenum") - rolling_min >= 0.3,
                pl.col("labevents/valuenum") >= pl.col("baseline") * 1.5,
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
