"""Logic for normalizing and computing urine output rate."""

import polars as pl


def normalize_weights(source: pl.LazyFrame) -> pl.LazyFrame:
    """Normalizes units, handles nulls, and sorts weight data."""
    pass
