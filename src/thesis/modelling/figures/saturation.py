"""AUPRC and net benefit saturation by training run count."""

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.axes import Axes

from thesis.modelling.figures.decision_curves import PALETTE

# net benefit is reported per 1,000 landmarks, as in `decision_curves.py`
PER = 1000.0


def _series(
    rows: Sequence[Mapping], mean_key: str, low_key: str, high_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sizes, mean, low and high as arrays, sorted by size.

    Args:
        rows (Sequence[Mapping]): One entry per ensemble size.
        mean_key (str): Key holding the across-subset mean.
        low_key (str): Key holding the across-subset minimum.
        high_key (str): Key holding the across-subset maximum.

    Returns:
        tuple: `(sizes, mean, low, high)`.

    Raises:
        ValueError: If a row is missing any of the four keys.
    """
    needed = ("n_runs", mean_key, low_key, high_key)
    for row in rows:
        missing = [key for key in needed if key not in row]
        if missing:
            raise ValueError(f"saturation row {dict(row)} lacks {', '.join(missing)}.")
    order = sorted(rows, key=lambda row: row["n_runs"])
    return tuple(
        np.asarray([row[key] for row in order], dtype=float)
        for key in ("n_runs", mean_key, low_key, high_key)
    )


def saturation_axes(
    axes: Axes,
    rows: Sequence[Mapping],
    *,
    mean_key: str,
    low_key: str,
    high_key: str,
    scale: float = 1.0,
    colour: str | None = None,
    label: str | None = None,
) -> Axes:
    """Renders the saturation curve on the input axes.

    Args:
        axes (Axes): Where to draw.
        rows (Sequence[Mapping]): One entry per ensemble size.
        mean_key (str): Key holding the across-subset mean.
        low_key (str): Key holding the across-subset minimum.
        high_key (str): Key holding the across-subset maximum.
        scale (float): Multiplies every value, for per-1,000 reporting.
        colour (str | None): Line colour, defaulting to the shared palette's first.
        label (str | None): Legend label for the mean line.

    Returns:
        Axes: The axes.
    """
    sizes, mean, low, high = _series(rows, mean_key, low_key, high_key)
    colour = colour or PALETTE[0]
    axes.fill_between(
        sizes,
        low * scale,
        high * scale,
        color=colour,
        alpha=0.18,
        linewidth=0,
        zorder=2,
    )
    axes.plot(
        sizes,
        mean * scale,
        color=colour,
        linewidth=1.6,
        marker="o",
        markersize=3.5,
        label=label,
        zorder=3,
    )
    axes.set_xlabel("training runs in the ensemble")
    axes.set_xticks(sizes)
    return axes


def panel_note(axes: Axes, text: str) -> Axes:
    """Writes the members-per-run and subset-count caveat onto the axes itself.

    Args:
        axes (Axes): Where to draw.
        text (str): The note.

    Returns:
        Axes: The axes, for chaining.
    """
    axes.text(
        0.98,
        0.05,
        text,
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="0.35",
    )
    return axes
