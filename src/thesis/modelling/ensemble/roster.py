"""Which banked predictions make up each arm of the study.

Rosters stay explicit lists rather than a glob over `runs/*`. A glob silently absorbs
a smoke test, a diagnostic re-run or a half-scored ladder yielding plausible but false
results.

"""

import json
from pathlib import Path

import numpy as np

from thesis.modelling.ensemble.diversity import Member, checkpoint_bundles

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / "motor_output" / "runs"
COMPARISON = ROOT / "motor_output" / "comparison"
NEWGRID = COMPARISON / "newgrid"

# the twelve new-grid LoRA configuration runs. Each run contributes its minimum
# validation loss run and its last run. If these runs coincide, only one of the
# two is included
LORA_RUNS = [
    "ng-all4-r2-a32",
    "ng-all4-r4-a32",
    "ng-all4-r8",
    "pilot-all4-r16-a32",
    "ng-all4-r32-a32",
    "ng-all4-r16",
    "ng-all4-r32",
    "ng-all5-r8",
    "ng-o-r8",
    "ng-qkv-r8",
    "ng-ff-r8",
    "ng-qv-seed1",
]

# Monolithic runs collapse around the final training step. This catastrophic forgetting
# is detrimental to the performance of the ensemble. Therefore 'last' is not included.
#
# The window is the pre-collapse phase shared by all three seeds: validation loss turns
# sharply upward at step 22,000 in every one, and step_002000 is still pre-convergence
# at AUPRC ~0.128.
MONOLITHIC_RUNS = ["ng-aki-seed0", "ng-aki-seed1", "ng-aki-seed2"]
MONOLITHIC_WINDOW = (6000, 18000)

# the seed the headline XGBoost pairing is measured against, so a single-model
# monolithic arm is the same model as `motor` in newgrid/comparison.json
MONOLITHIC_HEADLINE = "ng-aki-seed1"

# the 600-round booster, which continues the 300-round fit
# the 300-round model is `newgrid/xgboost_predictions.npz` and reads 0.0014 lower
XGBOOST = NEWGRID / "xgboost600_predictions.npz"


def by_loss_stem(run: str) -> str:
    """The checkpoint the frozen rule selects on one run's ladder.

    Args:
        run (str): A run folder name under `motor_output/runs`.

    Returns:
        str: The checkpoint stem, e.g. `step_018000`.

    Raises:
        FileNotFoundError: If the run has not been scored.
    """
    ranking = RUNS / run / "selection" / "checkpoint_ranking.json"
    if not ranking.is_file():
        raise FileNotFoundError(
            f"{run} has no {ranking.name}; score it first with "
            f"scripts/evaluate/score_checkpoints.py."
        )
    rows = json.loads(ranking.read_text())
    return Path(min(rows, key=lambda row: row["loss"])["checkpoint"]).stem


def lora_bundles(run: str) -> list[Path]:
    """One LoRA run's by-loss and `last` bundles, deduplicated.

    A run whose loss never turned selects its own `last`, and the two stems collapse to
    one file. Returning it once keeps every run weighted equally in the average.

    Args:
        run (str): A run folder name under `motor_output/runs`.

    Returns:
        list[Path]: One or two banked prediction bundles.
    """
    available = dict(checkpoint_bundles(RUNS / run))
    stems = dict.fromkeys([by_loss_stem(run), "last"])
    return [available[stem] for stem in stems]


def monolithic_bundles(run: str, window: tuple[int, int]) -> list[Path]:
    """One full fine-tune's pre-collapse checkpoints.

    Args:
        run (str): A run folder name under `motor_output/runs`.
        window (tuple[int, int]): Inclusive first and last training step to keep.

    Returns:
        list[Path]: The banked bundles inside the window, in step order.

    Raises:
        FileNotFoundError: If the window selects nothing.
    """
    low, high = window
    kept = [
        path
        for stem, path in checkpoint_bundles(RUNS / run)
        if stem.startswith("step_") and low <= int(stem.removeprefix("step_")) <= high
    ]
    if not kept:
        raise FileNotFoundError(f"{run} has no banked checkpoint inside {window}.")
    return kept


def lora_ensemble() -> list[Path]:
    """The project's headline LoRA ensemble: every run's by-loss checkpoint and last."""
    return [path for run in LORA_RUNS for path in lora_bundles(run)]


def drop_contested(members: list[Member]) -> list[Member]:
    """Removes landmarks that `(subject, time)` cannot identify uniquely.

    Raw MIMIC-IV records concurrent admissions for one subject, so stage 4 can grid two
    admissions onto the same instant. Neither the labeller nor stage 5.2 mints a
    `landmark_id`, so the two are indistinguishable to anything joining on the shared
    key. The baseline scores them differently, because their spines carry
    different `admittime`s. One key in the validation fold is affected. Dropping it
    keeps every analysis on exactly the rows `align_predictions` pairs on.

    Args:
        members (list[Member]): Loaded bundles, all covering the same cohort.

    Returns:
        list[Member]: The same members with the contested rows removed.
    """
    trimmed, dropped = [], 0
    for member in members:
        # trimmed on its own key column rather than the first member's, because the
        # bundles arrive in unrelated row orders and have not been aligned yet
        # every bundle covers the same cohort, so they lose the same landmarks
        keys = np.rec.fromarrays(
            [member.subjects.astype(np.int64), member.times.astype(np.int64)],
            names="s,t",
        )
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        keep = counts[inverse] == 1
        dropped = max(dropped, int((~keep).sum()))
        trimmed.append(Member(*(column[keep] for column in member)))
    if dropped:
        print(
            f"dropped {dropped:,} rows on landmark keys shared by concurrent "
            f"admissions, leaving {trimmed[0].scores.size:,}\n"
        )
    return trimmed
