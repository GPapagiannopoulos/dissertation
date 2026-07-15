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
    pass
