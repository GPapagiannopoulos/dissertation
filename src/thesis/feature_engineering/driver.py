"""Module driving the transformations for the diagnostic pipeline."""

import polars as pl

from thesis.config import settings

from .urine_output import calculate_urine_output_rate, net_urine, normalize_weights


def load_uo_data() -> pl.LazyFrame:
    """Loads and runs transformations on the urine and weight data."""
    weights_lf = pl.scan_parquet(
        settings.mimic4_ehr_weight_parquet,
        schema={
            "subject_id": pl.String,
            "hadm_id": pl.String,
            "stay_id": pl.String,
            "itemid": pl.String,
            "charttime": pl.Datetime,
            "valuenum": pl.Float64,
        },
    )
    urine_output_lf = pl.scan_parquet(
        settings.mimic4_ehr_urine_output_parquet,
        schema={
            "subject_id": pl.String,
            "hadm_id": pl.String,
            "stay_id": pl.String,
            "itemid": pl.String,
            "charttime": pl.Datetime,
            "valuenum": pl.Float64,
        },
    )

    normalized_weights_lf = normalize_weights(weights_lf)
    net_urine_output_lf = net_urine(urine_output_lf)

    return calculate_urine_output_rate(
        normalized_weights_lf, net_urine_output_lf
    ).rename({"subject_id": "patient_id"})
