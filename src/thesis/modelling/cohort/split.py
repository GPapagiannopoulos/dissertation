"""Module for handling the splitting of the cohort into training/validation/testing."""

import polars as pl

_POSITIVE_LABEL = "positive"
_NEGATIVE_LABEL = "negative"
_STRATUM_SEPARATOR = "_"


def _admission_bucket() -> pl.Expr:
    """Bands n_admissions into the buckets the stratification crosses with the label.

    The bands are 0, 1, 2-3 and 4+. The share of subjects who ever develop HA-AKI
    climbs roughly six-fold across these bands. A subject with many admissions
    contributes proportionally many landmarks. Balancing the label alone would
    leave admission-level prevalence free to drift between folds.

    Returns:
        pl.Expr: a String column derived from n_admissions
    """
    return (
        pl.when(pl.col("n_admissions") == 0)
        .then(pl.lit("0"))
        .when(pl.col("n_admissions") == 1)
        .then(pl.lit("1"))
        .when(pl.col("n_admissions") <= 3)
        .then(pl.lit("2-3"))
        .otherwise(pl.lit("4+"))
    )


def build_subject_strata(
    subject_ids: list[int], admissions: pl.LazyFrame, labels: pl.LazyFrame
) -> pl.LazyFrame:
    """Assign subjects to strata for subsequent cohort assignment.

    The strata are the cross of whether a subject was ever diagnosed with HA-AKI
    and how many admissions they hold, which is what lets a subject-level split
    still land admission-level prevalence on target.

    subject_ids is the left side of the join and every subject in it survives,
    including those holding no admissions at all. Those can never be labelled, so
    they are absent from the evaluation cohort, but they are still assigned a
    stratum so that pretraining has an unambiguous train set. Joining the other
    way round would delete them silently.

    An admission absent from admissions counts towards neither total even if the
    labels name it. admissions is the normalised event data, so absence means the
    concept map left that admission with no events and it cannot be modelled.

    Args:
        subject_ids: every subject in the database, as read from stage 3
        admissions: normalised events, needing subject_id and visit_id columns
        labels: the positive diagnoses, needing a visit_id column

    Returns:
        pl.LazyFrame: one row per subject, sorted by subject_id::

            subject_id            (Int64)   as given
            n_admissions          (UInt32)  distinct visit_id, 0 if none
            n_positive_admissions (UInt32)  of which diagnosed, 0 if none
            ever_positive         (Boolean)
            admission_bucket      (String)  0, 1, 2-3 or 4+
            stratum               (String)  e.g. positive_4+, negative_0

    Raises:
        ValueError: if subject_ids holds a duplicate, which would fan out the join
    """
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError(
            f"subject_ids holds {len(subject_ids) - len(set(subject_ids))} duplicated "
            f"ids, which would fan out the join and double those subjects' admissions."
        )

    universe = pl.LazyFrame(
        {"subject_id": subject_ids}, schema={"subject_id": pl.Int64}
    )
    positives = (
        labels.select("visit_id").unique().with_columns(is_positive=pl.lit(True))
    )

    per_subject = (
        admissions.select("subject_id", "visit_id")
        .drop_nulls()
        .unique()
        .join(positives, on="visit_id", how="left")
        .group_by("subject_id")
        .agg(
            n_admissions=pl.len(),
            n_positive_admissions=pl.col("is_positive").fill_null(False).sum(),
        )
    )

    return (
        universe.join(per_subject, on="subject_id", how="left")
        .with_columns(
            n_admissions=pl.col("n_admissions").fill_null(0),
            n_positive_admissions=pl.col("n_positive_admissions").fill_null(0),
        )
        .with_columns(
            ever_positive=pl.col("n_positive_admissions") > 0,
            admission_bucket=_admission_bucket(),
        )
        .with_columns(
            stratum=pl.concat_str(
                pl.when(pl.col("ever_positive"))
                .then(pl.lit(_POSITIVE_LABEL))
                .otherwise(pl.lit(_NEGATIVE_LABEL)),
                pl.col("admission_bucket"),
                separator=_STRATUM_SEPARATOR,
            )
        )
        .sort("subject_id")
    )
