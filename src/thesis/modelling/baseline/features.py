"""Helper functions for the implementation of the XGBoost tree."""

import polars as pl


def build_spine(labels: pl.LazyFrame, split: pl.LazyFrame) -> pl.LazyFrame:
    """Builds the spine for the model.

    Args:
        labels (pl.LazyFrame): The dataset labelled and with prediction
            grids
        split (pl.LazyFrame): The subject_ids split into training/validation/
            testing

    Returns:
        pl.LazyFrame: A LazyFrame representing the spine, specifically
            times when predictions where made for each subject.
    """
    return (
        labels.join(split, on="subject_id", how="left")
        .sort("subject_id", "visit_id", "prediction_time")
        .with_row_index("landmark_id")
    )


def visible_events(events: pl.LazyFrame, drop: list[str]) -> pl.LazyFrame:
    """Selects events to be included in the model.

    Args:
        events: The dataset labelled and with prediction grids
        drop (list[str]): A list of columns to exclude from the model

    Returns:
        pl.LazyFrame: A LazyFrame containing only events that the model is
            allowed to see
    """
    return events.select("subject_id", "code", "time", "numeric_value").filter(
        pl.col("code").is_in(drop).not_() & pl.col("time").is_not_null()
    )


def running_counts(events: pl.LazyFrame) -> pl.LazyFrame:
    """Returns a running count of visible events per subject and event type."""
    return events.sort("subject_id", "code", "time").with_columns(
        pl.col("time").cum_count().over("subject_id", "code").alias("n_obs")
    )


def candidate_pairs(spine: pl.LazyFrame, events: pl.LazyFrame) -> pl.LazyFrame:
    """Joins every landmark with every code its subject ever carries."""
    return spine.select("landmark_id", "subject_id", "prediction_time").join(
        events.select("subject_id", "code").unique(), on="subject_id", how="inner"
    )


def last_observation(pairs: pl.LazyFrame, events: pl.LazyFrame) -> pl.LazyFrame:
    """Joins landmarks with all the codes that happened during it."""
    return (
        pairs.sort("prediction_time")
        .join_asof(
            events.sort("time"),
            left_on="prediction_time",
            right_on="time",
            by=["subject_id", "code"],
            strategy="backward",
        )
        .drop_nulls("time")
        .with_columns(
            (
                (pl.col("prediction_time") - pl.col("time")).dt.total_minutes() / 60
            ).alias("hours_since")
        )
        .select("landmark_id", "code", "last_value", "hours_since", "n_obs")
    )
