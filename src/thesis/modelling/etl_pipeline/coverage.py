"""Helpers for mapping MIMIC-IV native codes to the standardized SSSOMOP of MOTOR."""

import polars as pl


def code_inventory(events: pl.LazyFrame) -> pl.LazyFrame:
    """Determines the number of events per code in the dataset."""
    return (
        events.group_by(pl.col("code"))
        .agg(pl.len().alias("count"))
        .sort(pl.col("code"))
    )
