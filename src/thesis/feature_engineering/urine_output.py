"""Logic for normalizing and computing urine output rate."""

from typing import Final

import polars as pl

LBS_TO_KGS: Final[float] = 0.45359237


def normalize_weights(source: pl.LazyFrame) -> pl.LazyFrame:
    """Normalizes units, handles nulls, and sorts weight data."""
    return (
        source.with_columns(
            pl.when(pl.col("itemid") == "226531")
            .then(pl.col("valuenum") * LBS_TO_KGS)
            .otherwise(pl.col("valuenum"))
        )
        .drop(["itemid", "valueuom"])
        .sort(["subject_id", "hadm_id"])
    )
