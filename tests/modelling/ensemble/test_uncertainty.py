"""Testing suite for the uncertainty decomposition.

Grouped per module, as `test_diversity.py` is: three helpers over one member matrix.
The decomposition is what the epistemic-uncertainty findings rest on, and it fails
quietly -- a swapped term or a lost sign still returns small plausible numbers in
nats.
"""

import numpy as np
import pytest

from thesis.modelling.ensemble.uncertainty import (
    binary_entropy,
    decompose,
    explained_by_bins,
)


def members(*columns: list[float]) -> np.ndarray:
    """A (n_members, n_labels) matrix, one argument per landmark."""
    return np.array(columns, dtype=float).T


def test_entropy_peaks_at_a_half_and_vanishes_at_the_ends() -> None:
    """The scale: `ln 2` at a coin flip, ~0 at certainty."""
    assert binary_entropy(np.array([0.5]))[0] == pytest.approx(np.log(2))
    assert binary_entropy(np.array([0.0]))[0] < 1e-5
    assert binary_entropy(np.array([1.0]))[0] < 1e-5


def test_entropy_is_symmetric_about_a_half() -> None:
    """Being 80% sure of yes is exactly as uncertain as 80% sure of no."""
    assert binary_entropy(np.array([0.2, 0.7])) == pytest.approx(
        binary_entropy(np.array([0.8, 0.3]))
    )


def test_entropy_is_finite_at_the_boundaries() -> None:
    """`log(0)` is negative infinity, and one such landmark would poison every mean."""
    assert np.all(np.isfinite(binary_entropy(np.array([0.0, 1.0, 0.5]))))


def test_agreeing_members_carry_no_epistemic_uncertainty() -> None:
    """Unanimity means naming the right member would tell you nothing.

    All three say 0.30, so the whole of the total is the members' own admitted noise.
    """
    parts = decompose(members([0.3, 0.3, 0.3]))

    assert parts.epistemic[0] == pytest.approx(0.0)
    assert parts.variance[0] == pytest.approx(0.0)
    assert parts.total[0] == pytest.approx(parts.aleatoric[0])


def test_disagreeing_members_split_the_same_total() -> None:
    """Three landmarks with identical predictions and very different splits.

    Every column below averages to 0.30, so the total is `H(0.3)` for all three. What
    changes is how much of it is disagreement: none, some, most. This is the whole
    point of the decomposition -- the prediction alone cannot tell them apart.
    """
    parts = decompose(members([0.3, 0.3, 0.3], [0.02, 0.3, 0.58], [0.01, 0.01, 0.88]))

    assert parts.mean == pytest.approx([0.3, 0.3, 0.3])
    assert parts.total == pytest.approx([0.6109, 0.6109, 0.6109], abs=1e-4)
    assert parts.epistemic == pytest.approx([0.0, 0.1478, 0.4512], abs=1e-4)
    assert parts.aleatoric == pytest.approx([0.6109, 0.4631, 0.1596], abs=1e-4)


def test_the_variance_column_is_the_per_landmark_ambiguity() -> None:
    """It ties exactly to the aggregate `diversity.py` already reports.

    Krogh and Vedelsby's decomposition is an identity on squared error, so the mean of
    this column must equal `mean_member_brier - ensemble_brier` to floating point --
    not approximately. If the two ever disagree, one of the two is wrong.
    """
    rng = np.random.default_rng(0)
    scores = rng.random((5, 400)) * 0.4
    targets = (rng.random(400) < 0.3).astype(float)

    parts = decompose(scores)
    ensemble_brier = ((parts.mean - targets) ** 2).mean()
    member_brier = ((scores - targets) ** 2).mean(axis=1).mean()

    assert parts.variance.mean() == pytest.approx(
        member_brier - ensemble_brier, rel=1e-12
    )


def test_epistemic_is_never_negative() -> None:
    """Concavity forbids it, so a negative value means the members are misaligned."""
    rng = np.random.default_rng(1)

    parts = decompose(rng.random((8, 500)))

    assert parts.epistemic.min() >= 0.0


def test_the_prediction_is_the_members_mean() -> None:
    """The column the rest of the project reports, recomputed here, must agree."""
    scores = np.array([[0.1, 0.9], [0.3, 0.5], [0.8, 0.4]])

    assert decompose(scores).mean == pytest.approx([0.4, 0.6])


@pytest.mark.parametrize(
    ("scores", "match"),
    [
        (np.array([0.1, 0.2, 0.3]), "n_members, n_labels"),
        (np.array([[0.1, 0.2]]), "at least two members"),
        (np.array([[0.1, 1.4], [0.2, 0.3]]), "probabilities"),
        (np.array([[0.1, -0.2], [0.2, 0.3]]), "probabilities"),
    ],
)
def test_a_malformed_matrix_is_refused(scores: np.ndarray, match: str) -> None:
    """One member cannot disagree, and logits would silently give nonsense entropy."""
    with pytest.raises(ValueError, match=match):
        decompose(scores)


def test_a_column_explains_itself_completely() -> None:
    """The upper end of the scale."""
    values = np.arange(50.0)

    assert explained_by_bins(values, values, bins=50) == pytest.approx(1.0)


def test_an_unrelated_column_explains_nothing() -> None:
    """Alternating values against a monotone predictor: the bin means are all equal."""
    values = np.tile([0.0, 1.0], 50)

    assert explained_by_bins(values, np.arange(100.0), bins=10) == pytest.approx(
        0.0, abs=1e-12
    )


def test_a_non_monotone_relationship_is_still_explained() -> None:
    """The reason for binning rather than correlating.

    `values` is a V in `by`, so the linear correlation is zero by symmetry while the
    relationship is perfect. A correlation coefficient would call this noise.
    """
    by = np.arange(-25.0, 26.0)
    values = np.abs(by)

    assert abs(np.corrcoef(by, values)[0, 1]) < 1e-12
    assert explained_by_bins(values, by, bins=25) > 0.99


def test_a_constant_column_has_nothing_to_explain() -> None:
    """Zero variance would otherwise divide by zero and return NaN."""
    assert explained_by_bins(np.ones(20), np.arange(20.0), bins=5) == 0.0


def test_fewer_bins_explain_less() -> None:
    """The resolution of the step function is what makes the test generous or strict."""
    by = np.arange(200.0)
    values = by**2

    assert explained_by_bins(values, by, bins=2) < explained_by_bins(
        values, by, bins=50
    )


@pytest.mark.parametrize(
    ("values", "by", "bins", "match"),
    [
        (np.ones(5), np.ones(4), 10, "same length"),
        (np.ones(5), np.ones(5), 0, "at least one bin"),
    ],
)
def test_a_malformed_request_is_refused(
    values: np.ndarray, by: np.ndarray, bins: int, match: str
) -> None:
    """Mismatched columns are not the same landmarks; zero bins cannot be split."""
    with pytest.raises(ValueError, match=match):
        explained_by_bins(values, by, bins=bins)
