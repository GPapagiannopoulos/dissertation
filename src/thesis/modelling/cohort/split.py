"""Module for handling the splitting of the cohort into training/validation/testing."""

import polars as pl


def build_subject_strata(
    subject_ids: list[int], admissions: pl.LazyFrame, labels: pl.LazyFrame
) -> pl.LazyFrame:
    """Assign subjects to strata for subsequent cohort assignment."""
    pass
