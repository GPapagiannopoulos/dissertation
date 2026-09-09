"""The reliability diagram and its residual panel.

Same convention as `decision_curves.py`: each function draws onto an axes handed to
it, so composition stays with the caller and nothing needs a display. Nothing here
computes a number -- the curves arrive already built by
`evaluation/metrics.py::calibration_curve`.

Three choices to keep consistent across the report:

* Both axes are logarithmic. The bins are equal-count, so at this prevalence nine of
  ten sit below a predicted risk of 0.05 while the tenth runs to 1.0; on linear axes
  every bin but the last collapses onto the origin.
* Perfect calibration is the diagonal, drawn thin and grey. It is a reference, not a
  competitor.
* Colours and display names are imported from `decision_curves.py` rather than
  redefined, so an arm keeps one colour and one label across every figure.

When the report carries a flexible curve, both are drawn: the smooth line with its
band is the moderate-calibration statement, and the binned markers are the
decomposition of the reported ECE.
"""

from collections.abc import Mapping

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from thesis.modelling.figures.decision_curves import LABELS, PALETTE, order_arms

# the arm order that fixes the palette, shared with the decision curve figures
ARM_ORDER = (
    "xgboost",
    "lora_last2",
    "lora_all_last",
    "lora_worst_trio",
    "monolithic_seeds",
    "monolithic_snapshot",
    "monolithic_single",
)


def _flexible(curve: Mapping) -> tuple[np.ndarray, ...] | None:
    """One arm's smooth curve and its band, or None if it was not fitted."""
    if "flexible_grid" not in curve:
        return None
    grid = np.asarray(curve["flexible_grid"], dtype=float)
    fitted = np.asarray(curve["flexible_rate"], dtype=float)
    if "flexible_lo" not in curve:
        return grid, fitted, None, None
    return (
        grid,
        fitted,
        np.asarray(curve["flexible_lo"], dtype=float),
        np.asarray(curve["flexible_hi"], dtype=float),
    )


def _columns(curve: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """The two plotted columns of one arm's curve, as float arrays."""
    return (
        np.asarray(curve["mean_score"], dtype=float),
        np.asarray(curve["observed_rate"], dtype=float),
    )


def reliability_axes(
    axes: Axes,
    curves: Mapping[str, Mapping],
    *,
    budget_threshold: float | None = None,
) -> Axes:
    """Panel A: observed frequency against predicted probability, per arm.

    Args:
        axes (Axes): Where to draw.
        curves (Mapping[str, Mapping]): One arm per key, each holding `mean_score`,
            `observed_rate` and optionally `observed_lo` / `observed_hi`.
        budget_threshold (float | None): Draw a vertical rule at the score that the
            reported alert budget cuts at. Calibration to the left of it is never
            acted on, so a model may be badly calibrated there and still deploy well.

    Returns:
        Axes: The axes, for chaining.

    Raises:
        ValueError: If an arm's two columns are different lengths.
    """
    drawn = [value for curve in curves.values() for value in _columns(curve)]
    finite = np.concatenate([column[np.isfinite(column)] for column in drawn])
    low = float(finite[finite > 0].min()) if (finite > 0).any() else 1e-4
    high = float(finite.max())

    axes.plot(
        [low, high],
        [low, high],
        color="0.55",
        linewidth=0.9,
        linestyle="--",
        label="perfect calibration",
        zorder=1,
    )
    if budget_threshold is not None:
        axes.axvline(budget_threshold, color="0.75", linewidth=0.9, zorder=0)

    for index, (name, curve) in enumerate(curves.items()):
        predicted, observed = _columns(curve)
        if predicted.shape != observed.shape:
            raise ValueError(
                f"{name} has {predicted.shape} predicted against {observed.shape} "
                f"observed; they are not the same curve."
            )
        colour = PALETTE[index % len(PALETTE)]
        if "observed_lo" in curve:
            axes.fill_between(
                predicted,
                np.asarray(curve["observed_lo"], dtype=float),
                np.asarray(curve["observed_hi"], dtype=float),
                color=colour,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
        smooth = _flexible(curve)
        if smooth is not None:
            grid, fitted, low, high = smooth
            if low is not None:
                axes.fill_between(
                    grid, low, high, color=colour, alpha=0.13, linewidth=0, zorder=2
                )
            # the smooth fit is the MODERATE-calibration statement; the binned markers
            # are the ECE decomposition. Drawn together so a reader can see the
            # grouping is not doing the work
            axes.plot(grid, fitted, color=colour, linewidth=1.6, zorder=4)
        axes.plot(
            predicted,
            observed,
            color=colour,
            linewidth=0.0 if smooth is not None else 1.4,
            marker="o",
            markersize=3.5,
            label=LABELS.get(name, name),
            zorder=3,
        )

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("mean predicted probability")
    axes.set_ylabel("observed AKI rate")
    return axes


def residual_axes(
    axes: Axes,
    curves: Mapping[str, Mapping],
    *,
    budget_threshold: float | None = None,
) -> Axes:
    """Panel B: observed minus predicted, which is what ECE takes the size of.

    The diagonal of panel A hides the sign and the size of a miss at low risk, where
    every arm is squeezed into one corner. Here a bin above zero is UNDER-predicted --
    more AKI happened than the model said -- and the distance from zero, weighted by
    the bin's share of rows, is exactly the arm's ECE.

    Args:
        axes (Axes): Where to draw.
        curves (Mapping[str, Mapping]): As `reliability_axes`.
        budget_threshold (float | None): Rule at the reported alert budget's score.

    Returns:
        Axes: The axes, for chaining.
    """
    axes.axhline(0.0, color="0.2", linewidth=0.8, zorder=1)
    if budget_threshold is not None:
        axes.axvline(budget_threshold, color="0.75", linewidth=0.9, zorder=0)

    for index, (name, curve) in enumerate(curves.items()):
        predicted, observed = _columns(curve)
        colour = PALETTE[index % len(PALETTE)]
        if "observed_lo" in curve:
            axes.fill_between(
                predicted,
                np.asarray(curve["observed_lo"], dtype=float) - predicted,
                np.asarray(curve["observed_hi"], dtype=float) - predicted,
                color=colour,
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
        smooth = _flexible(curve)
        if smooth is not None:
            grid, fitted, low, high = smooth
            if low is not None:
                axes.fill_between(
                    grid,
                    low - grid,
                    high - grid,
                    color=colour,
                    alpha=0.13,
                    linewidth=0,
                    zorder=2,
                )
            axes.plot(grid, fitted - grid, color=colour, linewidth=1.6, zorder=4)
        axes.plot(
            predicted,
            observed - predicted,
            color=colour,
            linewidth=0.0 if smooth is not None else 1.4,
            marker="o",
            markersize=3.5,
            label=LABELS.get(name, name),
            zorder=3,
        )

    axes.set_xscale("log")
    axes.set_xlabel("mean predicted probability")
    axes.set_ylabel("observed - predicted")
    return axes


def calibration_figure(
    report: Mapping,
    *,
    arms: tuple[str, ...] | None = None,
    figsize: tuple[float, float] = (10.0, 4.2),
) -> Figure:
    """The two-panel calibration figure, built from a `calibration_curves.py` report.

    Args:
        report (Mapping): The parsed JSON the driver writes.
        arms (tuple[str, ...] | None): Which arms to draw, defaulting to all of them.
            Six overlapping curves is already dense; a report figure usually names
            three.
        figsize (tuple[float, float]): Figure size in inches.

    Returns:
        Figure: The composed figure, ready to save.

    Raises:
        ValueError: If a named arm is absent from the report.
    """
    import matplotlib.pyplot as pyplot

    available = report["arms"]
    missing = [name for name in arms or () if name not in available]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} not in the report; it holds {', '.join(available)}."
        )
    chosen = {name: available[name] for name in arms} if arms else dict(available)
    ordered = order_arms(chosen, ARM_ORDER)
    budget = report.get("budget_threshold")

    figure, (left, right) = pyplot.subplots(1, 2, figsize=figsize)
    reliability_axes(left, ordered, budget_threshold=budget)
    residual_axes(right, ordered, budget_threshold=budget)

    left.set_title("A. reliability", loc="left", fontsize=10)
    right.set_title("B. residual", loc="left", fontsize=10)
    left.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    return figure
