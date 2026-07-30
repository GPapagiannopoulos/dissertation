"""Module for extracting patients with a positive AKI diagnosis as per criteria."""

import polars as pl


def identify_surviving_aki_admissions(
    labels: pl.LazyFrame, events: pl.LazyFrame
) -> pl.LazyFrame:
    """Identifies the admissions with valid tokens post normalization.

    The process of normalizing MIMIC-IV data resulted into dropping of
    events that could not be mapped to any valid MOTOR token. For positive
    diagnosis events to be useful they need surviving events.
    """
    return pl.LazyFrame()
