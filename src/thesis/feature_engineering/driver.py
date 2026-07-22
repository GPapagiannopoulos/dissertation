"""Module driving the transformations for the diagnostic pipeline."""

from typing import Final

import polars as pl

from thesis.config import settings
from thesis.eda.cache import ensure_event_cache

from .diagnostic_criteria import diagnose_hospital_acquired_aki
from .urine_output import calculate_urine_output_rate, net_urine, normalize_weights

# The derived ICU parquets store the ids and itemid as Int64, while the pure
# transforms (and the canonical event substrate) key on String. They are cast
# at this edge, before the transforms compare itemid against string literals.
_ID_COLUMNS: Final[list[str]] = ["subject_id", "hadm_id", "stay_id", "itemid"]


def load_uo_data() -> pl.LazyFrame:
    """Loads and transforms the weight and urine parquets into a rate frame.

    Scans the derived weight and urine-output parquets, casts their integer
    ids to String, and renames the urine ``value`` column to ``valuenum`` so
    both frames match the shape the pure transforms expect. It then runs the
    urine output rate pipeline and renames ``subject_id`` to ``patient_id`` to
    align with the canonical event substrate.

    Returns:
        pl.LazyFrame: the per-charttime urine output rate keyed by
            ``patient_id``, ``hadm_id``, and ``stay_id``.
    """
    weights_lf = pl.scan_parquet(settings.mimic4_ehr_weight_parquet).with_columns(
        pl.col(_ID_COLUMNS).cast(pl.String)
    )
    urine_output_lf = (
        pl.scan_parquet(settings.mimic4_ehr_urine_output_parquet)
        .with_columns(pl.col(_ID_COLUMNS).cast(pl.String))
        .rename({"value": "valuenum"})
    )

    normalized_weights_lf = normalize_weights(weights_lf)
    net_urine_output_lf = net_urine(urine_output_lf)

    return (
        calculate_urine_output_rate(normalized_weights_lf, net_urine_output_lf)
        .rename({"subject_id": "patient_id"})
        .with_columns(pl.col("charttime").dt.cast_time_unit("ns"))
    )


def diagnose_all(source: pl.LazyFrame) -> pl.LazyFrame:
    """Applies the diagnostic criteria functions to the base dataset.

    Produces a small LazyFrame that contains the diagnosis_made events
    in accordance to the diagnostic_criteria functions.

    Args:
        source (pl.LazyFrame): a LazyFrame containing the base dataset.

    Returns:
        pl.LazyFrame: a LazyFrame containing the vertically concatenated
            diagnoses events at admission level.
    """
    diagnoses_frames: list[pl.LazyFrame] = []
    complete_diagnosis_frames: pl.LazyFrame = pl.LazyFrame(
        {
            "event_type": pl.Series([], dtype=pl.String),
            "patient_id": pl.Series([], dtype=pl.String),
            "hadm_id": pl.Series([], dtype=pl.String),
            "timestamp": pl.Series([], dtype=pl.Datetime("ns")),
            "diagnosis_made/diagnosis": pl.Series([], dtype=pl.String),
        }
    )
    diagnoses_frames.append(complete_diagnosis_frames)

    uo_rate_data = load_uo_data()
    ha_aki_diagnoses_lf = diagnose_hospital_acquired_aki(source, uo_rate_data).rename(
        {"diagnosis": "diagnosis_made/diagnosis"}
    )
    diagnoses_frames.append(ha_aki_diagnoses_lf)

    return pl.concat(diagnoses_frames, how="vertical", parallel=True)


def driver() -> pl.LazyFrame:
    """Initiates the diagnostic pipeline."""
    return diagnose_all(pl.scan_parquet(ensure_event_cache()))
