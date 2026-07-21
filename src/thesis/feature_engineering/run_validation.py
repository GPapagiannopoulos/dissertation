"""Runnable entry point for the AKI diagnostic validation pipeline.

Run with ``python -m thesis.feature_engineering.run_validation``.
"""

import polars as pl

from thesis.eda.cache import ensure_event_cache
from thesis.feature_engineering.diagnostic_criteria import diagnose_aki
from thesis.feature_engineering.driver import _load_uo_data
from thesis.feature_engineering.validation import (
    aki_ground_truth,
    confusion_matrix,
    evaluable_admissions,
    metrics,
)


def run_validation() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Validates the AKI detector against the ICD ground truth.

    Loads the cached base event substrate and the urine-output rate data, runs
    the ungated KDIGO detector, and compares the admissions it flags against the
    ICD-coded AKI phenotype over the evaluable cohort.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: the 2x2 confusion matrix and the
            derived metrics frame.
    """
    source = pl.scan_parquet(ensure_event_cache())
    uo_data = _load_uo_data()

    predicted = diagnose_aki(source, uo_data).select("hadm_id").unique()
    actual = aki_ground_truth(source)
    evaluable = evaluable_admissions(source, uo_data)

    matrix = confusion_matrix(predicted, actual, evaluable)
    return matrix, metrics(matrix)


def main() -> None:
    """Prints the AKI validation confusion matrix and metrics."""
    matrix, scores = run_validation()
    with pl.Config(tbl_hide_dataframe_shape=True):
        print("AKI diagnostic validation — confusion matrix (evaluable cohort)")
        print(matrix)
        print("\nMetrics")
        print(scores)


if __name__ == "__main__":
    main()
