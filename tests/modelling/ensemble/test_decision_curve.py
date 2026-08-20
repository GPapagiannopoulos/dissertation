"""Testing suite for decision curve analysis."""

import numpy as np
import pytest

from thesis.modelling.ensemble.decision_curve import (
    alert_rate,
    net_benefit,
    subset_curves,
    treat_all_net_benefit,
)

SCORES = np.array([0.9, 0.8, 0.3, 0.1])
TARGETS = np.array([1.0, 0.0, 1.0, 0.0])


def brute_force(scores: np.ndarray, targets: np.ndarray, threshold: float) -> float:
    """The definition, written out with no index arithmetic to get wrong."""
    alerted = scores >= threshold
    true_positive = float((alerted & (targets == 1)).sum())
    false_positive = float((alerted & (targets == 0)).sum())
    odds = threshold / (1.0 - threshold)
    return (true_positive - false_positive * odds) / scores.size


def test_the_hand_computed_case() -> None:
    """At a 0.2 threshold three landmarks alert: two real, one false.

    One false alarm costs a quarter of a true positive (0.2 / 0.8), so the benefit is
    2/4 - (1/4)(0.25) = 0.4375.
    """
    assert net_benefit(SCORES, TARGETS, np.array([0.2]))[0] == pytest.approx(0.4375)


def test_it_matches_a_brute_force_reading_everywhere() -> None:
    """The vectorised search must agree with the definition at every threshold.

    This is the test that catches an off-by-one in the alerted count, which would
    shift every curve by one landmark and still look entirely reasonable.
    """
    rng = np.random.default_rng(0)
    scores = rng.random(300)
    targets = (rng.random(300) < 0.2).astype(float)
    thresholds = np.linspace(0.0, 0.95, 40)

    got = net_benefit(scores, targets, thresholds)

    assert got == pytest.approx([brute_force(scores, targets, t) for t in thresholds])


def test_the_alert_is_inclusive_of_the_threshold() -> None:
    """A landmark scoring exactly the threshold is alerted, not skipped.

    Two landmarks sit exactly at 0.3, one of them positive. Included, they give
    (1 - 1 x 3/7) / 3 = 0.1905; excluded, nothing alerts and the benefit is zero.
    The threshold is deliberately not 0.5, where the odds are 1 and a true positive
    cancels a false one, hiding the difference.
    """
    scores = np.array([0.3, 0.3, 0.1])
    targets = np.array([1.0, 0.0, 0.0])

    got = net_benefit(scores, targets, np.array([0.3]))[0]

    assert got == pytest.approx((1.0 - 0.3 / 0.7) / 3)
    assert got > 0.0


def test_at_a_zero_threshold_everything_alerts_and_benefit_is_the_prevalence() -> None:
    """False alarms are free at a threshold of zero, so only the positives count."""
    got = net_benefit(SCORES, TARGETS, np.array([0.0]))[0]

    assert got == pytest.approx(TARGETS.mean())


def test_a_threshold_above_every_score_alerts_nobody() -> None:
    """No alerts means no benefit and no harm, exactly like the treat-none line."""
    assert net_benefit(SCORES, TARGETS, np.array([0.95]))[0] == 0.0


def test_a_perfect_ranker_reaches_the_prevalence_at_every_threshold() -> None:
    """Scores equal to the labels alert on all positives and nothing else.

    That is the ceiling: net benefit can never exceed the prevalence, because it
    cannot count more true positives than exist.
    """
    scores = np.array([1.0, 0.0, 1.0, 0.0])
    thresholds = np.array([0.1, 0.4, 0.8])

    assert net_benefit(scores, TARGETS, thresholds) == pytest.approx(
        [TARGETS.mean()] * 3
    )


def test_benefit_never_exceeds_the_prevalence() -> None:
    """An invariant that holds for any model, and a cheap guard against sign errors."""
    rng = np.random.default_rng(1)
    scores = rng.random(500)
    targets = (rng.random(500) < 0.3).astype(float)

    got = net_benefit(scores, targets, np.linspace(0.0, 0.9, 30))

    assert got.max() <= targets.mean() + 1e-12


def test_treat_all_crosses_zero_at_the_prevalence() -> None:
    """Above the prevalence, alerting on everyone does net harm. That is the anchor."""
    targets = np.array([1.0, 0.0, 0.0, 0.0])  # prevalence 0.25

    curve = treat_all_net_benefit(targets, np.array([0.1, 0.25, 0.5]))

    assert curve[0] > 0.0
    assert curve[1] == pytest.approx(0.0)
    assert curve[2] < 0.0


def test_treat_all_equals_a_model_that_alerts_on_everything() -> None:
    """The reference line is not a special case; it is the constant predictor."""
    thresholds = np.array([0.05, 0.2, 0.4])

    assert treat_all_net_benefit(TARGETS, thresholds) == pytest.approx(
        net_benefit(np.ones_like(TARGETS), TARGETS, thresholds)
    )


def test_the_alert_rate_counts_the_landmarks_that_clear_the_threshold() -> None:
    """Reported beside the curve, because an undeployable alert volume must show."""
    assert alert_rate(SCORES, [0.0, 0.15, 0.35, 0.85, 0.95]) == pytest.approx(
        [1.0, 0.75, 0.5, 0.25, 0.0]
    )


def test_subsets_are_enumerated_and_averaged() -> None:
    """Three members give three pairs, and the mean sits inside the spread."""
    rng = np.random.default_rng(2)
    members = rng.random((3, 200))
    targets = (rng.random(200) < 0.3).astype(float)
    thresholds = np.linspace(0.05, 0.6, 12)

    got = subset_curves(members, targets, thresholds, size=2)

    assert got.n_subsets == 3
    assert np.all(got.low <= got.mean + 1e-12)
    assert np.all(got.mean <= got.high + 1e-12)


def test_every_subset_is_a_different_set_of_members() -> None:
    """Each single-member subset must be its own member, not the first one repeated.

    With three clearly different members, the size-one curves are exactly the three
    member curves: their mean is the elementwise mean and the spread is the
    elementwise range. An implementation that always took the leading members would
    collapse all three onto one and the spread would vanish.
    """
    rng = np.random.default_rng(7)
    targets = (rng.random(300) < 0.3).astype(float)
    strong = np.clip(targets * 0.6 + rng.random(300) * 0.4, 0.0, 1.0)
    weak = np.clip(targets * 0.1 + rng.random(300) * 0.9, 0.0, 1.0)
    members = np.stack([strong, rng.random(300), weak])
    thresholds = np.linspace(0.05, 0.5, 10)

    got = subset_curves(members, targets, thresholds, size=1)
    singles = np.stack([net_benefit(row, targets, thresholds) for row in members])

    assert got.n_subsets == 3
    assert got.mean == pytest.approx(singles.mean(axis=0))
    assert got.low == pytest.approx(singles.min(axis=0))
    assert got.high == pytest.approx(singles.max(axis=0))
    assert np.any(got.high > got.low)


def test_a_full_size_subset_is_the_whole_ensemble() -> None:
    """One subset exists, so mean, low and high all collapse onto it."""
    rng = np.random.default_rng(3)
    members = rng.random((4, 150))
    targets = (rng.random(150) < 0.25).astype(float)
    thresholds = np.linspace(0.05, 0.5, 8)

    got = subset_curves(members, targets, thresholds, size=4)
    whole = net_benefit(members.mean(axis=0), targets, thresholds)

    assert got.n_subsets == 1
    assert got.mean == pytest.approx(whole)
    assert got.low == pytest.approx(whole)


def test_sampling_caps_the_subset_count_reproducibly() -> None:
    """Ten members give 252 five-subsets; the cap keeps that tractable and seeded."""
    rng = np.random.default_rng(4)
    members = rng.random((10, 120))
    targets = (rng.random(120) < 0.3).astype(float)
    thresholds = np.linspace(0.05, 0.5, 6)

    first = subset_curves(members, targets, thresholds, size=5, max_subsets=20, seed=1)
    second = subset_curves(members, targets, thresholds, size=5, max_subsets=20, seed=1)

    assert first.n_subsets == 20
    assert first.mean == pytest.approx(second.mean)


@pytest.mark.parametrize(
    ("scores", "targets", "thresholds", "match"),
    [
        (np.array([0.5, 0.5]), np.array([1.0]), np.array([0.2]), "do not match"),
        (np.array([]), np.array([]), np.array([0.2]), "no predictions"),
        (np.array([0.5, 1.4]), np.array([1.0, 0.0]), np.array([0.2]), "probabilities"),
        (np.array([0.5, -0.2]), np.array([1.0, 0.0]), np.array([0.2]), "probabilities"),
        (np.array([0.5, 0.5]), np.array([1.0, 0.0]), np.array([1.0]), r"\[0, 1\)"),
        (np.array([0.5, 0.5]), np.array([1.0, 0.0]), np.array([-0.1]), r"\[0, 1\)"),
    ],
)
def test_a_malformed_request_is_refused(
    scores: np.ndarray, targets: np.ndarray, thresholds: np.ndarray, match: str
) -> None:
    """A threshold of one divides by zero; scores outside [0, 1] make it meaningless."""
    with pytest.raises(ValueError, match=match):
        net_benefit(scores, targets, thresholds)


@pytest.mark.parametrize("size", [0, 5])
def test_an_impossible_subset_size_is_refused(size: int) -> None:
    """Zero members is not an ensemble, and four cannot yield five."""
    members = np.random.default_rng(5).random((4, 50))
    targets = np.zeros(50)
    targets[:10] = 1.0

    with pytest.raises(ValueError, match="Cannot draw ensembles"):
        subset_curves(members, targets, np.array([0.2]), size=size)
