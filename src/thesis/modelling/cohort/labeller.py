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
