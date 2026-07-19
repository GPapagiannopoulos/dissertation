"""Logic for normalizing and computing urine output rate."""

from typing import Final

import polars as pl

LBS_TO_KGS: Final[float] = 0.45359237


def normalize_weights(source: pl.LazyFrame) -> pl.LazyFrame:
    """Normalizes units and sorts weight data.

    Args:
        source (pl.LazyFrame): a lazyframe with the loaded weight data.

    Returns:
        pl.LazyFrame: a lazyframe where the weights are all in kgs,
            sorted by patient, admission, and timestamp

    Raises:
         KeyError: if core fields are missing from the lazyframe
         ValueError: if core fields are the wrong dtype preventing processing
    """
    schema = source.collect_schema()
    necessary_cols: Final[list[str]] = [
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "valuenum",
    ]
    for col in necessary_cols:
        dtype = schema.get(col, None)
        if not dtype:
            raise KeyError(f"'{col}' is missing.")
        if col == "charttime" and dtype != pl.Datetime:
            raise ValueError(f"'charttime' needs to be a Datetime field,not {dtype}.")
        if col == "valuenum" and not dtype.is_numeric():
            raise ValueError(f"'valuenum' needs to be a numeric field, not {dtype}.")

    return (
        source.with_columns(
            pl.when(pl.col("itemid") == "226531")
            .then(pl.col("valuenum") * LBS_TO_KGS)
            .otherwise(pl.col("valuenum"))
        )
        .drop(["itemid", "valueuom"])
        .sort(["subject_id", "hadm_id", "charttime"])
    )
