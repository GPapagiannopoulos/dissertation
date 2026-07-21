"""Module for validating the output of the diagnostic criteria functions."""

import polars as pl

from thesis.constants import ICD_V9_AKI_PREFIX, ICD_V10_AKI_PREFIX


def aki_ground_truth(source: pl.LazyFrame) -> pl.Series:
    """Returns the unique admission ids for patients with AKI."""
    valid_ids = (
        source.filter(
            pl.any_horizontal(
                (pl.col("diagnoses_icd/icd_version") == "9")
                & (pl.col("diagnoses_icd/icd_code").str.starts_with(ICD_V9_AKI_PREFIX)),
                (pl.col("diagnoses_icd/icd_version") == "10")
                & (
                    pl.col("diagnoses_icd/icd_code").str.starts_with(ICD_V10_AKI_PREFIX)
                ),
            ),
        )
        .select("hadm_id")
        .drop_nulls()
        .unique()
        .collect(engine="streaming")
        .to_series()
        .sort()
    )

    return valid_ids
