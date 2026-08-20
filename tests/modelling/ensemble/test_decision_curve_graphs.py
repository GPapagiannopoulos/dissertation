"""Testing suite for the decision curve figures.

A figure cannot be checked by eye in CI, so the assertions are structural: that every
curve handed in is drawn, that it is drawn with the values given rather than some
transformation of them, and that a curve of the wrong length is refused rather than
silently truncated or broadcast.

The Agg backend is selected before pyplot is imported, so nothing here needs a
display.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from thesis.modelling.ensemble.graphs.decision_curves import (  # noqa: E402
    PER,
    arms_axes,
    difference_axes,
    differences_figure,
    order_arms,
    panels_figure,
    sizes_axes,
)

THRESHOLDS = np.linspace(0.01, 0.30, 12)


@pytest.fixture
def axes():
    """A bare axes, closed after the test so figures do not accumulate."""
    figure, drawn = plt.subplots()
    yield drawn
    plt.close(figure)


def curve(offset: float = 0.0) -> np.ndarray:
    """A plausible net benefit curve, falling as the threshold rises."""
    return 0.01 - 0.02 * THRESHOLDS + offset


def test_every_arm_and_reference_is_drawn(axes) -> None:
    """Two arms over two references is four lines, plus the zero rule."""
    arms_axes(
        axes,
        THRESHOLDS,
        {"lora": curve(), "xgboost": curve(0.001)},
        {"treat_all": curve(-0.02), "treat_none": np.zeros_like(THRESHOLDS)},
    )

    labelled = [line.get_label() for line in axes.get_lines()]

    assert {"lora", "xgboost", "treat_all", "treat_none"} <= set(labelled)


def test_the_plotted_values_are_the_values_given(axes) -> None:
    """Scaled to per 1,000 and nothing else -- no smoothing, no reordering."""
    values = curve()

    arms_axes(axes, THRESHOLDS, {"lora": values}, {})
    drawn = next(line for line in axes.get_lines() if line.get_label() == "lora")

    assert drawn.get_ydata() == pytest.approx(values * PER)
    assert drawn.get_xdata() == pytest.approx(THRESHOLDS * 100)


def test_references_are_drawn_beneath_the_arms(axes) -> None:
    """A reference read as a competitor would misrepresent the comparison."""
    arms_axes(axes, THRESHOLDS, {"lora": curve()}, {"treat_all": curve(-0.02)})

    lines = {line.get_label(): line for line in axes.get_lines()}

    assert lines["treat_all"].get_zorder() < lines["lora"].get_zorder()
    assert lines["treat_all"].get_linewidth() < lines["lora"].get_linewidth()


def test_the_clinical_band_is_shaded_when_given(axes) -> None:
    """The defensible window is fixed in advance, so it belongs on the figure."""
    before = len(axes.patches)

    arms_axes(axes, THRESHOLDS, {"lora": curve()}, {}, band=(0.05, 0.15))

    assert len(axes.patches) == before + 1


def test_no_band_is_shaded_by_default(axes) -> None:
    """Shading a window nobody asked for would imply a claim about thresholds."""
    arms_axes(axes, THRESHOLDS, {"lora": curve()}, {})

    assert not axes.patches


def test_each_ensemble_size_gets_a_line_and_a_spread(axes) -> None:
    """The band is the across-subset range, which must be drawn, not implied."""
    sizes = {
        3: (curve(), curve(-0.001), curve(0.001)),
        10: (curve(0.0005), curve(0.0005), curve(0.0005)),
    }

    sizes_axes(axes, THRESHOLDS, sizes)

    labels = [line.get_label() for line in axes.get_lines()]
    assert "3 runs" in labels
    assert "10 runs" in labels
    assert len(axes.collections) == 2


def test_a_single_run_is_labelled_in_the_singular(axes) -> None:
    """A size of one must read "1 run"; the plural is what an examiner circles."""
    sizes_axes(axes, THRESHOLDS, {1: (curve(), curve(), curve())})

    assert "1 run" in [line.get_label() for line in axes.get_lines()]


def test_the_size_reference_is_drawn_when_given(axes) -> None:
    """The comparator the size sweep is answering, on the same axes."""
    sizes_axes(
        axes,
        THRESHOLDS,
        {3: (curve(), curve(), curve())},
        reference=curve(-0.0005),
    )

    assert "monolithic, 3 runs" in [line.get_label() for line in axes.get_lines()]


def test_a_difference_is_drawn_against_a_zero_rule(axes) -> None:
    """Whether the band clears zero is the entire content of the panel."""
    value = np.full_like(THRESHOLDS, 0.0005)

    difference_axes(
        axes, THRESHOLDS, value, value - 0.0002, value + 0.0002, label="a - b"
    )

    lines = [line.get_label() for line in axes.get_lines()]
    assert "a - b" in lines
    assert len(axes.collections) == 1
    assert any(
        line.get_ydata() == pytest.approx([0.0, 0.0]) for line in axes.get_lines()
    )


def test_arms_are_reordered_onto_the_preferred_order() -> None:
    """Colour assignment follows position, so the order fixes the palette."""
    arms = {"xgboost": curve(), "lora": curve(), "monolithic": curve()}

    got = order_arms(arms, ["monolithic", "lora"])

    assert list(got) == ["monolithic", "lora", "xgboost"]


def test_reordering_keeps_every_arm() -> None:
    """A preferred order naming an absent arm must not drop the present ones."""
    arms = {"lora": curve(), "xgboost": curve()}

    got = order_arms(arms, ["absent", "xgboost"])

    assert set(got) == set(arms)
    assert list(got)[0] == "xgboost"


@pytest.mark.parametrize(
    "draw",
    [
        lambda ax, bad: arms_axes(ax, THRESHOLDS, {"lora": bad}, {}),
        lambda ax, bad: arms_axes(ax, THRESHOLDS, {}, {"treat_all": bad}),
        lambda ax, bad: sizes_axes(ax, THRESHOLDS, {3: (bad, bad, bad)}),
        lambda ax, bad: difference_axes(ax, THRESHOLDS, bad, bad, bad),
    ],
)
def test_a_curve_of_the_wrong_length_is_refused(axes, draw) -> None:
    """Matplotlib would raise something opaque, or broadcast a scalar silently."""
    with pytest.raises(ValueError, match="not the same curve"):
        draw(axes, np.zeros(THRESHOLDS.size - 1))


def make_report() -> dict:
    """A report shaped exactly as `scripts/decision_curves.py` writes one."""
    return {
        "thresholds": THRESHOLDS.tolist(),
        "arms": {
            "monolithic_seeds": {"net_benefit": curve().tolist()},
            "lora_last2": {"net_benefit": curve(0.0005).tolist()},
            "xgboost": {"net_benefit": curve(0.001).tolist()},
        },
        "references": {
            "treat_all": curve(-0.02).tolist(),
            "treat_none": np.zeros_like(THRESHOLDS).tolist(),
        },
        "subset_curves": {
            "3": {
                "mean": curve().tolist(),
                "low": curve(-0.001).tolist(),
                "high": curve(0.001).tolist(),
            },
            "10": {
                "mean": curve(0.0005).tolist(),
                "low": curve(0.0005).tolist(),
                "high": curve(0.0005).tolist(),
            },
        },
        "differences": {
            "lora_last2-monolithic_seeds": {
                "value": np.full_like(THRESHOLDS, 0.0005).tolist(),
                "lo": np.full_like(THRESHOLDS, 0.0002).tolist(),
                "hi": np.full_like(THRESHOLDS, 0.0008).tolist(),
            },
            "lora_last2-xgboost": {
                "value": np.full_like(THRESHOLDS, -0.0005).tolist(),
                "lo": np.full_like(THRESHOLDS, -0.0009).tolist(),
                "hi": np.full_like(THRESHOLDS, -0.0001).tolist(),
            },
        },
    }


def test_the_panels_figure_holds_both_panels() -> None:
    """Panel A compares models, panel B sweeps ensemble size."""
    figure = panels_figure(make_report())

    try:
        assert len(figure.axes) == 2
        titles = [axes.get_title(loc="left") for axes in figure.axes]
        assert titles[0].startswith("A.")
        assert titles[1].startswith("B.")
    finally:
        plt.close(figure)


def test_the_panels_figure_relabels_the_arms_for_a_reader() -> None:
    """Report keys are snake_case identifiers; a figure legend must not be."""
    figure = panels_figure(make_report())

    try:
        labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert "XGBoost" in labels
        assert "monolithic_seeds" not in labels
        assert "MOTOR, 3 fine-tunes" in labels
    finally:
        plt.close(figure)


def test_the_band_is_shaded_only_when_asked_for() -> None:
    """The clinical window is a stated assumption, so it must be explicit."""
    shaded = panels_figure(make_report(), band=(0.05, 0.15))
    plain = panels_figure(make_report(), band=None)

    try:
        assert len(shaded.axes[0].patches) == 1
        assert not plain.axes[0].patches
    finally:
        plt.close(shaded)
        plt.close(plain)


def test_the_differences_figure_gives_each_pair_its_own_panel() -> None:
    """Two differences over two columns is one row, with no blank axes left showing."""
    figure = differences_figure(make_report(), columns=2)

    try:
        visible = [axes for axes in figure.axes if axes.get_visible()]
        assert len(visible) == 2
    finally:
        plt.close(figure)


def test_unused_panels_are_hidden_rather_than_left_empty() -> None:
    """An empty axes in a thesis figure reads as a missing result."""
    figure = differences_figure(make_report(), columns=3)

    try:
        assert len([axes for axes in figure.axes if axes.get_visible()]) == 2
        assert len(figure.axes) == 3
    finally:
        plt.close(figure)


def test_a_difference_panel_is_titled_with_both_arms() -> None:
    """The reader has to know which way round the subtraction went."""
    figure = differences_figure(make_report())

    try:
        titles = " ".join(axes.get_title(loc="left") for axes in figure.axes)
        assert "LoRA, paired checkpoints" in titles
        assert "XGBoost" in titles
    finally:
        plt.close(figure)
