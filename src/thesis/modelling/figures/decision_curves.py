"""The two decision curve panels.

Each function draws onto an axes handed to it rather than creating its own, so the
composition of a figure stays with the caller and every piece is testable without a
display. Nothing here computes a number: the curves arrive already built by
`decision_curve.py`, which is where their correctness is asserted.

Two conventions worth keeping consistent across the report:

* Net benefit is plotted in true positives per 1,000 landmarks, not per landmark.
* The reference lines are drawn thin and grey. Alerting on everyone and alerting
  on nobody are not competitors. Drawing them with the same weight as a model
  invites them to be read as results.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

PER = 1000.0

# ordered so the palette stays stable as arms are added or dropped, and chosen to
# survive greyscale printing and the common forms of colour blindness
PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#994F00",
)


def _check(thresholds: np.ndarray, values: np.ndarray, name: str) -> None:
    """Guards that a curve lines up with the threshold grid.

    Raises:
        ValueError: If the curve is not one value per threshold.
    """
    if np.asarray(values).shape != thresholds.shape:
        raise ValueError(
            f"{name} has {np.asarray(values).shape} values against "
            f"{thresholds.shape} thresholds; they are not the same curve."
        )


def arms_axes(
    axes: Axes,
    thresholds: np.ndarray,
    arms: Mapping[str, np.ndarray],
    references: Mapping[str, np.ndarray],
    *,
    band: tuple[float, float] | None = None,
    clip: bool = True,
) -> Axes:
    """Panel A: one line per arm, over the two reference strategies.

    Args:
        axes (Axes): Where to draw.
        thresholds (np.ndarray): The threshold grid, shaped (n,).
        arms (Mapping[str, np.ndarray]): Net benefit per arm, each shaped (n,).
        references (Mapping[str, np.ndarray]): Treat-all and treat-none, same shape.
        band (tuple[float, float] | None): Shade this threshold range as the
            clinically defensible window.
        clip (bool): Set the y range from the ARMS alone, letting the treat-all line
            leave the frame. Alerting on everyone falls to roughly minus ten times the
            prevalence by a 30% threshold, so an unclipped axis compresses every model
            into an indistinguishable smear near zero. Clipping is the convention in
            published decision curves for exactly this reason; the reference still
            shows where it crosses zero, which is the part that carries meaning.

    Returns:
        Axes: The axes, for chaining.

    Raises:
        ValueError: If any curve does not match the threshold grid.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    if band is not None:
        axes.axvspan(band[0] * 100, band[1] * 100, color="0.92", zorder=0)

    for name, values in references.items():
        _check(thresholds, values, name)
        axes.plot(
            thresholds * 100,
            np.asarray(values) * PER,
            color="0.55",
            linewidth=0.9,
            linestyle="--",
            label=name,
            zorder=1,
        )

    for index, (name, values) in enumerate(arms.items()):
        _check(thresholds, values, name)
        axes.plot(
            thresholds * 100,
            np.asarray(values) * PER,
            color=PALETTE[index % len(PALETTE)],
            linewidth=1.6,
            label=name,
            zorder=2,
        )

    axes.set_xlabel("threshold probability (%)")
    axes.set_ylabel("net benefit (true positives per 1,000 landmarks)")
    axes.axhline(0.0, color="0.2", linewidth=0.6, zorder=1)

    if clip and arms:
        drawn = np.concatenate([np.asarray(v).ravel() for v in arms.values()]) * PER
        top = float(drawn.max())
        axes.set_ylim(-0.10 * top, 1.08 * top)
    return axes


def sizes_axes(
    axes: Axes,
    thresholds: np.ndarray,
    sizes: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    reference: np.ndarray | None = None,
) -> Axes:
    """Panel B: mean net benefit by ensemble size, with the across-subset spread.

    The shaded band is the range over which the runs were picked, not a confidence
    interval over patients.

    Args:
        axes (Axes): Where to draw.
        thresholds (np.ndarray): The threshold grid, shaped (n,).
        sizes (Mapping[int, tuple]): Size -> (mean, low, high), each shaped (n,).
        reference (np.ndarray | None): A comparator curve, drawn dashed.

    Returns:
        Axes: The axes, for chaining.

    Raises:
        ValueError: If any curve does not match the threshold grid.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    for index, (size, (mean, low, high)) in enumerate(sorted(sizes.items())):
        _check(thresholds, mean, f"size {size} mean")
        _check(thresholds, low, f"size {size} low")
        colour = PALETTE[index % len(PALETTE)]
        axes.fill_between(
            thresholds * 100,
            np.asarray(low) * PER,
            np.asarray(high) * PER,
            color=colour,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )
        axes.plot(
            thresholds * 100,
            np.asarray(mean) * PER,
            color=colour,
            linewidth=1.6,
            label=f"{size} run{'s' if size != 1 else ''}",
            zorder=2,
        )

    if reference is not None:
        _check(thresholds, reference, "reference")
        axes.plot(
            thresholds * 100,
            np.asarray(reference) * PER,
            color="0.2",
            linewidth=1.2,
            linestyle="--",
            label="monolithic, 3 runs",
            zorder=3,
        )

    axes.set_xlabel("threshold probability (%)")
    axes.set_ylabel("net benefit (true positives per 1,000 landmarks)")
    return axes


def difference_axes(
    axes: Axes,
    thresholds: np.ndarray,
    value: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    label: str = "",
    colour: str = PALETTE[0],
) -> Axes:
    """One paired difference with its bootstrap band, against a zero line.

    Curves for similar models overlap almost exactly, so the difference is used to
    clearly indicate which ones are ahead.

    Args:
        axes (Axes): Where to draw.
        thresholds (np.ndarray): The threshold grid, shaped (n,).
        value (np.ndarray): The observed difference, shaped (n,).
        low (np.ndarray): The lower bound, shaped (n,).
        high (np.ndarray): The upper bound, shaped (n,).
        label (str): Legend entry.
        colour (str): Line colour.

    Returns:
        Axes: The axes, for chaining.

    Raises:
        ValueError: If any curve does not match the threshold grid.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    for name, values in (("value", value), ("low", low), ("high", high)):
        _check(thresholds, values, name)

    axes.axhline(0.0, color="0.2", linewidth=0.8, zorder=1)
    axes.fill_between(
        thresholds * 100,
        np.asarray(low) * PER,
        np.asarray(high) * PER,
        color=colour,
        alpha=0.2,
        linewidth=0,
        zorder=2,
    )
    axes.plot(
        thresholds * 100,
        np.asarray(value) * PER,
        color=colour,
        linewidth=1.6,
        label=label,
        zorder=3,
    )
    axes.set_xlabel("threshold probability (%)")
    axes.set_ylabel("difference in net benefit (per 1,000)")
    return axes


def order_arms(
    arms: Mapping[str, np.ndarray], preferred: Sequence[str]
) -> dict[str, np.ndarray]:
    """Puts arms into a fixed order so colours stay stable across figures.

    Args:
        arms (Mapping[str, np.ndarray]): The arms to draw.
        preferred (Sequence[str]): Names in the order they should appear. Names not
            listed follow, in their original order.

    Returns:
        dict[str, np.ndarray]: The same arms, reordered.
    """
    listed = [name for name in preferred if name in arms]
    return {name: arms[name] for name in listed + [k for k in arms if k not in listed]}


# how each arm is named on a figure; the report's keys are snake_case identifiers and
# an examiner should not have to decode them
LABELS = {
    "monolithic_single": "MOTOR, one fine-tune",
    "monolithic_snapshot": "MOTOR, one run's checkpoints",
    "monolithic_seeds": "MOTOR, 3 fine-tunes",
    "lora_all_last": "LoRA, final checkpoints",
    "lora_last2": "LoRA, paired checkpoints",
    "lora_worst_trio": "LoRA, weakest 3 runs",
    "xgboost": "XGBoost",
    "treat_all": "alert on everyone",
    "treat_none": "alert on nobody",
}

# fixed so a colour means the same arm in every figure of the report
ARM_ORDER = (
    "xgboost",
    "lora_last2",
    "lora_all_last",
    "lora_worst_trio",
    "monolithic_seeds",
    "monolithic_snapshot",
    "monolithic_single",
)


def _named(values: Mapping[str, object]) -> dict:
    """Relabels report keys for display, leaving unknown keys as they are."""
    return {LABELS.get(key, key): value for key, value in values.items()}


def panels_figure(
    report: Mapping,
    *,
    band: tuple[float, float] | None = (0.05, 0.15),
    figsize: tuple[float, float] = (10.0, 4.2),
) -> Figure:
    """The two-panel comparison figure, built from a `decision_curves.py` report.

    Args:
        report (Mapping): The parsed JSON the driver writes.
        band (tuple[float, float] | None): The clinically defensible window to shade.
        figsize (tuple[float, float]): Figure size in inches.

    Returns:
        Figure: A figure holding panel A and panel B.

    Raises:
        KeyError: If the report is missing the arms or the threshold grid.
    """
    import matplotlib.pyplot as plt

    thresholds = np.asarray(report["thresholds"], dtype=float)
    figure, (left, right) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    arms = order_arms(
        {name: np.asarray(a["net_benefit"]) for name, a in report["arms"].items()},
        ARM_ORDER,
    )
    references = {
        name: np.asarray(values) for name, values in report["references"].items()
    }
    arms_axes(left, thresholds, _named(arms), _named(references), band=band)
    left.legend(frameon=False, fontsize=8, loc="upper right")
    left.set_title("A. models", loc="left", fontsize=10)

    sizes = {
        int(size): (
            np.asarray(entry["mean"]),
            np.asarray(entry["low"]),
            np.asarray(entry["high"]),
        )
        for size, entry in report["subset_curves"].items()
    }
    reference = np.asarray(report["arms"]["monolithic_seeds"]["net_benefit"])
    sizes_axes(right, thresholds, sizes, reference=reference)
    right.legend(frameon=False, fontsize=8, loc="upper right")
    right.set_title("B. training runs in the ensemble", loc="left", fontsize=10)

    for axes in (left, right):
        axes.set_xlim(thresholds[0] * 100, thresholds[-1] * 100)
        axes.spines[["top", "right"]].set_visible(False)
    return figure


def differences_figure(
    report: Mapping,
    *,
    columns: int = 2,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """One panel per paired difference, each against its own zero rule.

    Curves for similar models sit on top of one another, so the difference panels are
    where a reader can actually see whether one arm is ahead.

    Args:
        report (Mapping): The parsed JSON the driver writes.
        columns (int): How many panels per row.
        figsize (tuple[float, float] | None): Figure size; derived from the grid when
            not given.

    Returns:
        Figure: A grid of difference panels.

    Raises:
        KeyError: If the report carries no `differences` block.
    """
    import matplotlib.pyplot as plt

    thresholds = np.asarray(report["thresholds"], dtype=float)
    differences = report["differences"]
    rows = int(np.ceil(len(differences) / columns))
    figure, grid = plt.subplots(
        rows,
        columns,
        figsize=figsize or (5.0 * columns, 3.2 * rows),
        constrained_layout=True,
        squeeze=False,
    )

    for index, (name, entry) in enumerate(differences.items()):
        axes = grid[index // columns][index % columns]
        left, right = name.split("-", 1)
        difference_axes(
            axes,
            thresholds,
            np.asarray(entry["value"]),
            np.asarray(entry["lo"]),
            np.asarray(entry["hi"]),
            colour=PALETTE[index % len(PALETTE)],
        )
        axes.set_title(
            f"{LABELS.get(left, left)} \u2212 {LABELS.get(right, right)}",
            loc="left",
            fontsize=9,
        )
        axes.set_xlim(thresholds[0] * 100, thresholds[-1] * 100)
        axes.spines[["top", "right"]].set_visible(False)

    for empty in range(len(differences), rows * columns):
        grid[empty // columns][empty % columns].set_visible(False)
    return figure
