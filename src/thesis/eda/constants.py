"""Invariants used in EDA, definitions, and modeling."""

from typing import Final

import polars as pl

DTYPE_TO_POLARS_DTYPE_MAP: Final[dict[str, type[pl.DataType]]] = {
    "String": pl.String,
    "UInt": pl.UInt16,
    "Date": pl.Date,
    "DateTime": pl.Datetime,
    "Boolean": pl.Boolean,
    "Float": pl.Float32,
}

ICD_V9_AKI_PREFIX: Final[str] = "584"
ICD_V10_AKI_PREFIX: Final[str] = "N17"
