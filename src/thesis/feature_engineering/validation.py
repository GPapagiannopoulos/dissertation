"""Module for validating the output of the diagnostic criteria functions."""

import polars as pl

from thesis.constants import ICD_V9_AKI_PREFIX, ICD_V10_AKI_PREFIX


def aki_ground_truth(source: pl.LazyFrame) -> pl.LazyFrame:
    """Returns the unique admission ids carrying an AKI ICD code.

    Args:
        source (pl.LazyFrame): base MIMIC-IV dataset.

    Returns:
        pl.LazyFrame: a single ``hadm_id`` column of the unique, non-null,
            sorted admission IDs with an AKI diagnosis code.
    """
    return (
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
        .sort("hadm_id")
    )


def evaluable_admissions(source: pl.LazyFrame, uo_data: pl.LazyFrame) -> pl.LazyFrame:
    """Returns the admissions the AKI algorithm could evaluate.

    An admission is evaluable when it carries the data at least one arm of
    diagnose_aki consumes: a Creatinine labevents measurement or a
    urine-output rate. The two id sets are unioned, since either arm alone
    is sufficient to reach a diagnosis. A Creatinine row with a null valuenum
    is not a measurement and does not make an admission evaluable.

    Args:
        source (pl.LazyFrame): base MIMIC-IV dataset.
        uo_data (pl.LazyFrame): the urine-output rate frame (_load_uo_data).

    Returns:
        pl.LazyFrame: a single hadm_id column of the unique, non-null,
            sorted evaluable admission IDs.
    """
    creatinine_ids = source.filter(
        (pl.col("labevents/label") == "Creatinine")
        & pl.col("labevents/valuenum").is_not_null()
    ).select("hadm_id")

    uo_ids = uo_data.select("hadm_id")

    return (
        pl.concat([creatinine_ids, uo_ids], how="vertical")
        .drop_nulls()
        .unique()
        .sort("hadm_id")
    )
