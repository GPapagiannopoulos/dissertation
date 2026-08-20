"""Logic for normalizing and computing urine output rate."""

from typing import Final

import polars as pl

LBS_TO_KGS: Final[float] = 0.45359237

# KDIGO stage-1 oliguria is assessed over a trailing six-hour window.
AKI_UO_WINDOW: Final[str] = "6h"


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
        "stay_id",
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
        .sort(["subject_id", "hadm_id", "stay_id", "charttime"])
    )


def net_urine(source: pl.LazyFrame) -> pl.LazyFrame:
    """Calculates the net urine output at a given charttime.

    GU irrigant instilled (itemid 227488) is subtracted from the measured
    output so bladder-irrigation volumes are not counted as urine.

    Args:
        source (pl.LazyFrame): a lazyframe with the loaded urine output data.

    Returns:
        pl.LazyFrame: net urine per (subject, admission, stay, charttime),
            sorted by those keys.

    Raises:
         KeyError: if core fields are missing from the lazyframe
         ValueError: if core fields are the wrong dtype preventing processing
    """
    schema = source.collect_schema()
    necessary_cols: Final[list[str]] = [
        "subject_id",
        "hadm_id",
        "stay_id",
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
        source.drop_nulls()
        .with_columns(
            pl.when(
                (pl.col("itemid") == "227488") & (pl.col("valuenum") > 0)
            )  # accounting for irrigant
            .then(pl.col("valuenum") * -1)
            .otherwise(pl.col("valuenum"))
        )
        .group_by(["subject_id", "hadm_id", "stay_id", "charttime"])
        .agg(pl.col("valuenum").sum())
        .sort(["subject_id", "hadm_id", "stay_id", "charttime"])
    )


def _combine_weight_and_urine(
    weight_lf: pl.LazyFrame, net_urine_lf: pl.LazyFrame
) -> pl.LazyFrame:
    """Aligns a weight against each net urine measurement.

    Each urine measurement is matched to the most recent weight recorded
    at or before its ``charttime`` within the same admission (a backward
    as-of join, i.e. the weight is fed forward). Urine measurements that
    precede the first weight of the admission are backfilled with that
    first weight, so only admissions with no weight at all stay null.

    Args:
        weight_lf (pl.LazyFrame): normalized weight data (kg) as produced by
            ``normalize_weights``.
        net_urine_lf (pl.LazyFrame): net urine per charttime as produced by
            ``net_urine``.

    Returns:
        pl.LazyFrame: the net urine frame with a ``weight`` (kg) column, sorted
            by subject, admission, and charttime. ``weight`` is null only when
            the admission has no recorded weight.
    """
    weight = weight_lf.select(
        "subject_id",
        "hadm_id",
        "charttime",
        pl.col("valuenum").alias("weight"),
    ).sort(["subject_id", "hadm_id", "charttime"])

    urine = net_urine_lf.sort(["subject_id", "hadm_id", "charttime"])

    return urine.join_asof(
        weight,
        on="charttime",
        by=["subject_id", "hadm_id"],
        strategy="backward",
        check_sortedness=False,
    ).with_columns(
        pl.col("weight").fill_null(strategy="backward").over(["subject_id", "hadm_id"])
    )


def calculate_urine_output_rate(
    weight_lf: pl.LazyFrame, net_urine_lf: pl.LazyFrame
) -> pl.LazyFrame:
    """Calculates the rolling rate of urine output per kg of body weight.

    For each measurement the net urine over the trailing KDIGO oliguria
    window is summed per ICU stay and divided by the patient's weight and
    the observed span of that window, giving a rate in mL/kg/h. It is summed
    over stay_id as opposed to hadm_id because outputevents is ICU-scoped.
    This means that an admission with a 'bounce-back' (ICU->ward->ICU again)
    would have a discontinuity in measurements masked.

    The denominator is the *observed* span of the window (current
    ``charttime`` minus the earliest ``charttime`` still inside it), not a
    fixed six hours: a left-open window over intermittent charting never
    spans the full period. ``window_hours`` and ``n_events`` are returned so
    the caller can decide how much coverage to demand before trusting a rate.

    Args:
        weight_lf (pl.LazyFrame): normalized weight data (kg).
        net_urine_lf (pl.LazyFrame): net urine per charttime.

    Returns:
        pl.LazyFrame: sorted by subject, admission, stay, and charttime, with
            columns ``rate`` (mL/kg/h; null when the window is instantaneous
            or the weight is unknown), ``window_hours`` (observed span), and
            ``n_events`` (measurements in the window). A negative ``rate``
            flags a window whose net urine is negative (a bladder-irrigation
            artifact) rather than genuine oliguria.
    """
    combined = _combine_weight_and_urine(weight_lf, net_urine_lf).sort(
        ["subject_id", "hadm_id", "stay_id", "charttime"]
    )

    window_volume = (
        pl.col("valuenum")
        .rolling_sum_by("charttime", AKI_UO_WINDOW, closed="both")
        .over("stay_id")
    )
    window_start = (
        pl.col("charttime")
        .rolling_min_by("charttime", AKI_UO_WINDOW, closed="both")
        .over("stay_id")
    )
    n_events = (
        pl.col("_one")
        .rolling_sum_by("charttime", AKI_UO_WINDOW, closed="both")
        .over("stay_id")
    )

    return (
        combined.with_columns(_one=pl.lit(1, dtype=pl.UInt32))
        .with_columns(
            window_volume.alias("window_volume"),
            window_start.alias("window_start"),
            n_events.alias("n_events"),
        )
        .with_columns(
            (
                (pl.col("charttime") - pl.col("window_start")).dt.total_seconds() / 3600
            ).alias("window_hours")
        )
        .with_columns(
            pl.when(pl.col("window_hours") > 0)
            .then(pl.col("window_volume") / pl.col("weight") / pl.col("window_hours"))
            .otherwise(None)
            .alias("rate")
        )
        .select(
            "subject_id",
            "hadm_id",
            "stay_id",
            "charttime",
            "rate",
            "window_hours",
            "n_events",
        )
    )
