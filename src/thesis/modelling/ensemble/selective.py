"""Deferring the contested predictions inside an alert budget.

Selective prediction asks a narrower question than the uncertainty decomposition
does: given that a ward can act on a fixed number of alerts, is member disagreement
a better reason to withhold one than the score itself already is?

The framing here is dictated by a measurement, not by convention. Ambiguity --
the members' variance in probability space -- correlates with the ensemble's own
prediction at Spearman 0.954, because variance in probability space is bounded by
`p(1-p)` and so carries a steep function of risk inside it. Deferring the most
contested 10% of the fold therefore deletes the alert list and 41.6% of every
positive in it. Three consequences shape this module:

* the deferral signal must be **scale-free**, so `Var(logit p)` rather than `Var(p)`;
* deferral happens **inside the alert list**, never over the whole fold;
* the acted-on **count** is held fixed rather than the rate, or the operating point
  drifts as coverage falls and the comparison measures the drift instead.

`selective_curve` reports both readouts, because they answer different questions and
cost one computation. Without backfill it is an ordinary risk/coverage curve, and the
control to beat is the score's own ordering. With backfill the acted-on set stays the
size the ward budgeted for, and the same control becomes near-null by construction --
so a signal that wins there is carrying information the ranking does not have.
"""

from typing import NamedTuple

import numpy as np

from thesis.modelling.ensemble.uncertainty import decompose

# keeps `logit` finite on members that saturate; 1e-6 is two orders below the
# smallest ensemble score the validation fold carries
EPSILON = 1e-6


class SelectiveCurve(NamedTuple):
    """One deferral signal's behaviour across coverages.

    Attributes:
        coverage (np.ndarray): The fraction of the alert list still acted on.
        n_acted (np.ndarray): How many landmarks are acted on at each coverage.
        precision (np.ndarray): The AKI rate among them.
        recall (np.ndarray): The share of the fold's positives they hold.
        deferred_prevalence (np.ndarray): The AKI rate among the deferred alerts. A
            working signal defers false alarms, so this falls below the alert list's
            own precision.
    """

    coverage: np.ndarray
    n_acted: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    deferred_prevalence: np.ndarray


def logit(probabilities: np.ndarray) -> np.ndarray:
    """The log-odds of a probability, clipped away from the asymptotes.

    Args:
        probabilities (np.ndarray): Values in [0, 1].

    Returns:
        np.ndarray: `log(p / (1 - p))`, finite everywhere.
    """
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def within_band_rank(values: np.ndarray, by: np.ndarray, *, bins: int = 200):
    """Percentile rank of `values` among the rows sharing its `by` band.

    The scale-free alternative to a transform: instead of dividing out the
    prediction's influence, compare each landmark only against others predicted to be
    at the same risk. Bands are quantile cuts, so each holds the same number of rows.

    The band count matters more than it looks. At ten bands the top band spans a
    predicted-risk range of 0.059 and the ranking inside it is still mostly the
    prediction; at two hundred the leak is 0.0002.

    Args:
        values (np.ndarray): The column being ranked.
        by (np.ndarray): The column defining the bands.
        bins (int): How many quantile bands to cut `by` into.

    Returns:
        np.ndarray: A rank in [0, 1] per row, comparable across bands.

    Raises:
        ValueError: If the two columns differ in length, or `bins` is below one.
    """
    if values.shape != by.shape:
        raise ValueError(
            f"Ranking {values.shape} values inside {by.shape} bands is not a "
            f"row-for-row operation."
        )
    if bins < 1:
        raise ValueError(f"Need at least one band, got {bins}.")

    edges = np.quantile(by, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    band = np.searchsorted(edges, by, side="right")

    ranked = np.empty(values.shape, dtype=float)
    for index in np.unique(band):
        rows = np.flatnonzero(band == index)
        if rows.size == 1:
            ranked[rows] = 0.5
            continue
        order = np.argsort(values[rows], kind="stable")
        position = np.empty(rows.size, dtype=float)
        position[order] = np.arange(rows.size, dtype=float)
        ranked[rows] = position / (rows.size - 1)
    return ranked


def disagreement_signals(members: np.ndarray, *, bins: int = 200) -> dict:
    """Every candidate deferral signal, from one aligned member matrix.

    Args:
        members (np.ndarray): Members' probabilities, shaped (n_members, n_labels).
        bins (int): Bands for the within-band ranking.

    Returns:
        dict: Signal name to a per-landmark column, higher meaning defer sooner.

    Raises:
        ValueError: If `decompose` rejects the matrix.
    """
    parts = decompose(members)
    return {
        # the scale-free one: variance in log-odds space, which p(1-p) does not bound
        "logit_variance": np.var(logit(members), axis=0),
        # the same disagreement compared only against equally-risky landmarks
        "banded_variance": within_band_rank(parts.variance, parts.mean, bins=bins),
        # the two probability-space columns, kept as the confounded comparison
        "variance": parts.variance,
        "mutual_information": parts.epistemic,
    }


def control_signals(scores: np.ndarray, targets: np.ndarray, *, seed: int = 0) -> dict:
    """The signals a deferral rule has to beat to be worth an ensemble.

    `low_score` is the one that matters. Inside an alert list, deferring the lowest
    scores IS deferring the landmarks closest to the operating threshold, so the two
    controls the design named collapse into one -- and it is a strong control, because
    the bottom of an alert list genuinely does hold most of its false alarms.

    `oracle` cheats, deferring true negatives first. It is the ceiling any real signal
    is a fraction of, and measuring it first is what stops an unpromising correction
    from being built twice.

    Args:
        scores (np.ndarray): The ensemble's predictions.
        targets (np.ndarray): The labels.
        seed (int): Seeds the random control.

    Returns:
        dict: Signal name to a per-landmark column, higher meaning defer sooner.
    """
    rng = np.random.default_rng(seed)
    return {
        "low_score": -scores,
        "random": rng.random(scores.shape),
        "oracle": 1.0 - np.asarray(targets, dtype=float),
    }


def rank_by_score(scores: np.ndarray) -> np.ndarray:
    """Row indices ordered from the highest score down.

    Args:
        scores (np.ndarray): The ensemble's predictions.

    Returns:
        np.ndarray: An index array; ties keep their original order.
    """
    return np.argsort(-np.asarray(scores), kind="stable")


def selective_curve(
    scores: np.ndarray,
    targets: np.ndarray,
    signal: np.ndarray,
    *,
    budget: float,
    coverages: np.ndarray,
    backfill: bool,
    ranking: np.ndarray | None = None,
) -> SelectiveCurve:
    """Precision on the acted-on set as alerts are deferred.

    Args:
        scores (np.ndarray): The ensemble's predictions.
        targets (np.ndarray): The labels.
        signal (np.ndarray): Deferral signal; the highest values are deferred first.
        budget (float): The alert budget as a fraction of the fold.
        coverages (np.ndarray): Fractions of the alert list to keep acting on.
        backfill (bool): Replace each deferred alert with the next-ranked landmark, so
            the acted-on count stays at the budget.
        ranking (np.ndarray | None): A precomputed `argsort(-scores)`. It depends only
            on the scores, so a bootstrap comparing several signals on one draw would
            otherwise re-sort the whole fold once per signal -- the dominant cost of
            this analysis by two orders of magnitude.

    Returns:
        SelectiveCurve: One row per coverage.

    Raises:
        ValueError: If the arrays disagree in length, if the budget is not in (0, 1],
            or if any coverage falls outside [0, 1].
    """
    shapes = {scores.shape, targets.shape, signal.shape}
    if len(shapes) != 1:
        raise ValueError(
            f"Selective prediction needs one row per landmark in every array, got "
            f"shapes {sorted(str(shape) for shape in shapes)}."
        )
    if not 0.0 < budget <= 1.0:
        raise ValueError(
            f"The alert budget must be a fraction in (0, 1], got {budget}."
        )
    coverages = np.asarray(coverages, dtype=float)
    if coverages.min() < 0.0 or coverages.max() > 1.0:
        raise ValueError(
            f"Coverage is a fraction of the alert list; got the range "
            f"[{coverages.min()}, {coverages.max()}]."
        )

    targets = np.asarray(targets, dtype=float)
    if ranking is None:
        ranking = rank_by_score(scores)
    elif ranking.shape != scores.shape:
        raise ValueError(
            f"A precomputed ranking must name every row once; got {ranking.shape} "
            f"for {scores.shape} scores."
        )
    budgeted = int(round(budget * scores.size))
    if budgeted < 1:
        raise ValueError(
            f"A budget of {budget} over {scores.size} landmarks alerts on nothing."
        )
    alerts = ranking[:budgeted]

    # the deferral order INSIDE the alert list, and the queue backfill draws from
    deferral = alerts[np.argsort(-signal[alerts], kind="stable")]
    reserve = ranking[budgeted:]
    positives = targets.sum()

    rows = []
    for coverage in coverages:
        deferred_count = int(round((1.0 - coverage) * budgeted))
        deferred = deferral[:deferred_count]
        acted = deferral[deferred_count:]
        if backfill:
            acted = np.concatenate([acted, reserve[:deferred_count]])

        held = targets[acted]
        rows.append(
            (
                coverage,
                held.size,
                held.mean() if held.size else 0.0,
                held.sum() / positives if positives else 0.0,
                targets[deferred].mean() if deferred.size else float("nan"),
            )
        )

    columns = np.array(rows, dtype=float).T
    return SelectiveCurve(
        coverage=columns[0],
        n_acted=columns[1].astype(int),
        precision=columns[2],
        recall=columns[3],
        deferred_prevalence=columns[4],
    )
