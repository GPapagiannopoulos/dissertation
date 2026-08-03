"""Module housing the custom labeller and its helper functions: Stage 4.

MOTOR natively uses femr to apply a time horizon to the labels and
handle censoring. The module responsible for this works by looping over
subjects, making it inefficient and unfit for our purposes (making predictions
per admission rather than patient). Hence, we develop our own labeller
that vectorizes grid generation per-admission, and let MOTOR handle the
mapping of subjects to labels.
"""

from typing import Final

import polars as pl

_INPATIENT_VISIT_CODES: Final[list[str]] = ["Visit/IP", "Visit/ERIP"]


def build_admission_windows(events: pl.LazyFrame) -> pl.LazyFrame:
    """Extracts the admission window information for the normalized db.

    Args:
        events: a LazyFrame containing all normalized events

    Returns:
        pl.LazyFrame: a frame containing the admission and discharge window ::

            subject_id              (Int64)                 as given
            visit_id                (Int64)                 as given
            visit_code              (String)                e.g. Visit/IP | Visit/ERIP
            admittime               (Datetime('us'))        renamed
            dischtime               (Datetime('us'))        renamed
    """
    return events.filter(pl.col("code").str.starts_with("Visit/")).select(
        pl.col("subject_id"),
        pl.col("visit_id"),
        pl.col("code").alias("visit_code"),
        pl.col("time").alias("admittime"),
        pl.col("end").alias("dischtime"),
    )


def filter_inpatient_admissions(windows: pl.LazyFrame) -> pl.LazyFrame:
    """Filters the admission window only for inpatient admissions."""
    return windows.filter(pl.col("visit_code").is_in(_INPATIENT_VISIT_CODES))


def filter_false_admissions(windows: pl.LazyFrame) -> pl.LazyFrame:
    """Filters out admissions where the LOS is non-positive."""
    return windows.filter(pl.col("admittime") < pl.col("dischtime"))


def join_diagnosis_time(windows: pl.LazyFrame, labels: pl.LazyFrame) -> pl.LazyFrame:
    """Joins the labels with the time of diagnosis to the admission window.

    Not all visits have a timestamp for a diagnosis of interest. Uses a left
    join to preserve this information as a null value for diagtime.

    Args:
        windows (pl.LazyFrame): a LazyFrame containing the admission windows
        labels (pl.LazyFrame): a LazyFrame containing the output of the
            'identify_surviving_admissions.py' script

    Returns:
        pl.LazyFrame: a LazyFrame containing a timestamp in pl.Datetime['us']
            with the moment of diagnosis for a specific visit, null otherwise.
    """
    return windows.join(
        labels.select(
            "subject_id",
            "visit_id",
            pl.col("timestamp").cast(pl.Datetime("us")).alias("diagtime"),
        ),
        on=["subject_id", "visit_id"],
        how="left",
    )


def build_landmark_grid(
    windows_with_onset: pl.LazyFrame, delta_hours: str = "12h"
) -> pl.LazyFrame:
    """Generates the prediction landmark grid for each visit.

    Args:
        windows_with_onset (pl.LazyFrame): a LazyFrame containing the admission windows
            for each visit_id and the time of diagnosis
        delta_hours (str): a string representing the number of hours
            between prediction times.

    Returns:
        pl.LazyFrame: a LazyFrame containing one record for each prediction
            landmark, starting 48h post admission, and ending at discharge or
            diagnosis confirmation exclusive, whichever comes first. Drops visits
            without any admission landmarks in the generated window.
    """
    return (
        windows_with_onset.with_columns(
            pl.datetime_ranges(
                pl.col("admittime") + pl.duration(hours=48),
                pl.min_horizontal(pl.col("dischtime"), pl.col("diagtime")),
                interval=delta_hours,
                closed="left",
            ).alias("prediction_times")
        )
        .filter(pl.col("prediction_times").list.len() > 0)
        .explode("prediction_times")
    )
