"""Module for validating the output of the diagnostic criteria functions."""

import polars as pl

from thesis.constants import ICD_V9_AKI_PREFIX, ICD_V10_AKI_PREFIX


def aki_ground_truth(source: pl.LazyFrame) -> pl.LazyFrame:
    """Returns the unique admission ids carrying an AKI ICD code.

    Args:
        source (pl.LazyFrame): base MIMIC-IV dataset.

    Returns:
        pl.LazyFrame: a single ``hadm_id`` column of the unique, non-null,
            sorted admission IDs with an AKI diagnosis code.
    """
    return (
        source.filter(
            pl.any_horizontal(
                (pl.col("diagnoses_icd/icd_version") == "9")
                & (pl.col("diagnoses_icd/icd_code").str.starts_with(ICD_V9_AKI_PREFIX)),
                (pl.col("diagnoses_icd/icd_version") == "10")
                & (
                    pl.col("diagnoses_icd/icd_code").str.starts_with(ICD_V10_AKI_PREFIX)
                ),
            ),
        )
        .select("hadm_id")
        .drop_nulls()
        .unique()
        .sort("hadm_id")
    )


def evaluable_admissions(source: pl.LazyFrame, uo_data: pl.LazyFrame) -> pl.LazyFrame:
    """Returns the admissions the AKI algorithm could evaluate.

    An admission is evaluable when it carries the data at least one arm of
    diagnose_aki consumes: a Creatinine labevents measurement or a
    urine-output rate. The two id sets are unioned, since either arm alone
    is sufficient to reach a diagnosis. A Creatinine row with a null valuenum
    is not a measurement and does not make an admission evaluable.

    Args:
        source (pl.LazyFrame): base MIMIC-IV dataset.
        uo_data (pl.LazyFrame): the urine-output rate frame (_load_uo_data).

    Returns:
        pl.LazyFrame: a single hadm_id column of the unique, non-null,
            sorted evaluable admission IDs.
    """
    creatinine_ids = source.filter(
        (pl.col("labevents/label") == "Creatinine")
        & pl.col("labevents/valuenum").is_not_null()
    ).select("hadm_id")

    uo_ids = uo_data.select("hadm_id")

    return (
        pl.concat([creatinine_ids, uo_ids], how="vertical")
        .drop_nulls()
        .unique()
        .sort("hadm_id")
    )


def confusion_matrix(
    predicted: pl.LazyFrame,
    actual: pl.LazyFrame,
    evaluable: pl.LazyFrame,
) -> pl.DataFrame:
    """Returns the 2x2 AKI confusion matrix over the evaluable cohort.

    Compares the algorithm's predictions against the ICD ground truth,
    restricted to the admissions the algorithm could evaluate. Both ``actual``
    and ``predicted`` are narrowed to ``evaluable`` first, so the four cells
    always sum to the size of the evaluable cohort: an ICD-positive admission
    with no creatinine or urine-output data is out of scope, not a false
    negative.

    Args:
        predicted (pl.LazyFrame): hadm_ids flagged by ``diagnose_aki``.
        actual (pl.LazyFrame): hadm_ids with an AKI ICD code
            (``aki_ground_truth``).
        evaluable (pl.LazyFrame): the evaluable cohort
            (``evaluable_admissions``).

    Returns:
        pl.DataFrame: a 2x2 matrix with a ``predicted`` label column
            ("positive"/"negative") and ``actual_positive``/``actual_negative``
            count columns, laid out as::

                predicted   actual_positive   actual_negative
                positive    TP                FP
                negative    FN                TN
    """
    actual_evaluable = actual.join(evaluable, on="hadm_id", how="semi")
    predicted_evaluable = predicted.join(evaluable, on="hadm_id", how="semi")

    cells = {
        "tp": predicted_evaluable.join(actual_evaluable, on="hadm_id", how="semi"),
        "fp": predicted_evaluable.join(actual_evaluable, on="hadm_id", how="anti"),
        "fn": actual_evaluable.join(predicted_evaluable, on="hadm_id", how="anti"),
        "tn": evaluable.join(predicted_evaluable, on="hadm_id", how="anti").join(
            actual_evaluable, on="hadm_id", how="anti"
        ),
    }
    counts = {
        name: frame.select(pl.len()).collect().item() for name, frame in cells.items()
    }

    return pl.DataFrame(
        {
            "predicted": pl.Series(["positive", "negative"], dtype=pl.String),
            "actual_positive": pl.Series([counts["tp"], counts["fn"]], dtype=pl.UInt32),
            "actual_negative": pl.Series([counts["fp"], counts["tn"]], dtype=pl.UInt32),
        }
    )


def metrics(confusion: pl.DataFrame) -> pl.DataFrame:
    """Derives diagnostic metrics from an AKI confusion matrix.

    A metric whose denominator is zero (e.g. sensitivity with no actual
    positives) is undefined and reported as null rather than a placeholder
    number.

    Args:
        confusion (pl.DataFrame): a 2x2 matrix as returned by confusion_matrix.

    Returns:
        pl.DataFrame: a tidy frame with a metric column and a value
            column, covering sensitivity, specificity, precision, NPV, F1,
            accuracy, and prevalence.
    """

    def _safe_div(numerator: int, denominator: int) -> float | None:
        """Returns numerator / denominator, or None when the denominator is 0."""
        return numerator / denominator if denominator else None

    positive = confusion.filter(pl.col("predicted") == "positive")
    negative = confusion.filter(pl.col("predicted") == "negative")
    tp = positive["actual_positive"].item()
    fp = positive["actual_negative"].item()
    fn = negative["actual_positive"].item()
    tn = negative["actual_negative"].item()

    total = tp + fp + fn + tn
    scores = {
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "precision": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "accuracy": _safe_div(tp + tn, total),
        "prevalence": _safe_div(tp + fn, total),
    }

    return pl.DataFrame(
        {
            "metric": pl.Series(list(scores.keys()), dtype=pl.String),
            "value": pl.Series(list(scores.values()), dtype=pl.Float64),
        }
    )
