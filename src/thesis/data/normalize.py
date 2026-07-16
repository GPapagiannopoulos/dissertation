"""Canonical shape for the event substrate.

Owns the per-event-type transform that turns PyHealth's raw
``global_event_df`` into the canonical LazyFrame every consumer (EDA lens,
feature engineering, training) reads. Kept separate from the caching I/O in
``thesis.eda.cache`` so the substrate's shape can evolve independently of how
it is persisted.
"""

import polars as pl
from pyhealth.datasets import MIMIC4Dataset

from thesis.config import settings
from thesis.data.sources import (
    cast_frame,
    cleanse_float_values,
    mimic4_add_descriptions_to_icd_codes,
    replace_mimic4_non_icd_codes,
)


def build_event_pipeline(ds: MIMIC4Dataset, event_type: str) -> pl.LazyFrame:
    """Lazily generates a transformation pipeline schema.

    Runs the pipeline per event_type to handle the memory load of the
    full dataset.

    Args:
        ds (MIMIC4Dataset): PyHealth's native MIMIC4Dataset loader object.
        event_type (str): the event_type on which to run the transformations

    Returns:
        LazyFrame: a lf containing the transformed data

    Raises:
        InvalidOperationError: if no hadm_id columns are present.
    """
    prefix = f"{event_type}/"
    table_cols = [
        "patient_id",
        "event_type",
        "timestamp",
        *[
            c
            for c in ds.global_event_df.collect_schema().names()
            if c.startswith(prefix)
        ],
    ]
    lf = ds.global_event_df.filter(pl.col("event_type") == event_type).select(
        table_cols
    )

    float_fields = [
        col
        for col, dtype in settings.mimic4_ehr_dtype_mapping.items()
        if dtype == "Float" and col.startswith(prefix)
    ]
    if float_fields:
        lf = cleanse_float_values(lf, float_fields)

    dtype_map = {
        col: dtype
        for col, dtype in settings.mimic4_ehr_dtype_mapping.items()
        if col.startswith(prefix)
    }
    if dtype_map:
        lf = cast_frame(lf, dtype_map)

    if event_type == "diagnoses_icd":
        lf = mimic4_add_descriptions_to_icd_codes(
            lf, settings.mimic4_ehr_d_icd_diagnoses, event_type
        )
    elif event_type == "procedures_icd":
        lf = mimic4_add_descriptions_to_icd_codes(
            lf, settings.mimic4_ehr_d_icd_procedures, event_type
        )
    elif event_type == "labevents":
        lf = replace_mimic4_non_icd_codes(
            lf, settings.mimic4_ehr_d_labitems, event_type
        )

    hadm_selector = pl.selectors.ends_with("/hadm_id")
    lf = lf.with_columns(pl.coalesce(hadm_selector).alias("hadm_id")).drop(
        hadm_selector,
    )

    return lf
