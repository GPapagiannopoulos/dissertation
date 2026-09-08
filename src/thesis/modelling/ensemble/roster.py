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
MONOLITHIC_RUNS = ["ng-aki-seed0", "ng-aki-seed1", "ng-aki-seed2"]
MONOLITHIC_WINDOW = (6000, 18000)

# the seed the headline XGBoost pairing is measured against, so a single-model
# monolithic arm is the same model as `motor` in newgrid/comparison.json
MONOLITHIC_HEADLINE = "ng-aki-seed1"

# the 600-round booster, which continues the 300-round fit
# the 300-round model is `newgrid/xgboost_predictions.npz` and reads 0.0014 lower
XGBOOST = NEWGRID / "xgboost600_predictions.npz"

# where each fold's banked predictions live. They are kept apart because
# `score_checkpoints.py` defaults to writing into `selection`, and scoring the test
# fold there would overwrite every validation bundle the reported numbers rest on.
SELECTION = {"validation": "selection", "testing": "selection_test"}


def by_loss_stem(run: str) -> str:
    """The checkpoint the frozen rule selects on one run's ladder.

    Args:
        run (str): A run folder name under `motor_output/runs`.

    Returns:
        str: The checkpoint stem, e.g. `step_018000`.

    Raises:
        FileNotFoundError: If the run has not been scored on validation.
    """
    ranking = RUNS / run / SELECTION["validation"] / "checkpoint_ranking.json"
    if not ranking.is_file():
        raise FileNotFoundError(
            f"{run} has no {ranking.name}; score it first with "
            f"scripts/evaluate/scoring/score_checkpoints.py."
        )
    rows = json.loads(ranking.read_text())
    return Path(min(rows, key=lambda row: row["loss"])["checkpoint"]).stem


def lora_bundles(run: str, fold: str = "validation") -> list[Path]:
    """One LoRA run's by-loss and `last` bundles, deduplicated.

    A run whose loss never turned selects its own `last`, and the two stems collapse to
    one file. Returning it once keeps every run weighted equally in the average.

    Args:
        run (str): A run folder name under `motor_output/runs`.
        fold (str): Which fold's banked bundles to return. The STEM is still
            chosen on validation; only the predictions come from this fold.

    Returns:
        list[Path]: One or two banked prediction bundles.
    """
    available = dict(checkpoint_bundles(RUNS / run, SELECTION[fold]))
    stems = dict.fromkeys([by_loss_stem(run), "last"])
    return [available[stem] for stem in stems]


def monolithic_bundles(
    run: str, window: tuple[int, int], fold: str = "validation"
) -> list[Path]:
    """One full fine-tune's pre-collapse checkpoints.

    Args:
        run (str): A run folder name under `motor_output/runs`.
        window (tuple[int, int]): Inclusive first and last training step to keep.
        fold (str): Which fold's banked bundles to return.

    Returns:
        list[Path]: The banked bundles inside the window, in step order.

    Raises:
        FileNotFoundError: If the window selects nothing.
    """
    low, high = window
    kept = [
        path
        for stem, path in checkpoint_bundles(RUNS / run, SELECTION[fold])
        if stem.startswith("step_") and low <= int(stem.removeprefix("step_")) <= high
    ]
    if not kept:
        raise FileNotFoundError(f"{run} has no banked checkpoint inside {window}.")
    return kept


def lora_ensemble(fold: str = "validation") -> list[Path]:
    """The headline LoRA ensemble: every run's by-loss checkpoint and its last.

    Args:
        fold (str): Which fold's banked bundles to return.

    Returns:
        list[Path]: 23 bundles -- twelve runs at two checkpoints each, less the one
            run whose loss never turned and whose by-loss pick IS its own `last`.
    """
    return [path for run in LORA_RUNS for path in lora_bundles(run, fold)]


def baseline_bundle(fold: str = "validation") -> Path:
    """The 600-round booster's banked predictions for a fold."""
    if fold == "validation":
        return XGBOOST
    return NEWGRID / f"xgboost600_{fold}_predictions.npz"


def arm_definitions(
    window: tuple[int, int] = MONOLITHIC_WINDOW, fold: str = "validation"
) -> dict[str, list[Path]]:
    """Every arm of the study, as the banked bundles that make it up.

    Args:
        window (tuple[int, int]): The monolithic arm's pre-collapse step window.
        fold (str): Which fold's banked bundles each arm is built from. Checkpoint
            SELECTION always happens on validation, whatever this says.

    Returns:
        dict[str, list[Path]]: One arm per key, each a list of member bundles.
    """
    subdir = SELECTION[fold]
    headline = dict(checkpoint_bundles(RUNS / MONOLITHIC_HEADLINE, subdir))
    return {
        "monolithic_single": [headline[by_loss_stem(MONOLITHIC_HEADLINE)]],
        "monolithic_snapshot": monolithic_bundles(MONOLITHIC_HEADLINE, window, fold),
        "monolithic_seeds": [
            path
            for run in MONOLITHIC_RUNS
            for path in monolithic_bundles(run, window, fold)
        ],
        "lora_all_last": [
            dict(checkpoint_bundles(RUNS / run, subdir))["last"] for run in LORA_RUNS
        ],
        "lora_last2": [path for run in LORA_RUNS for path in lora_bundles(run, fold)],
        "xgboost": [baseline_bundle(fold)],
    }


def drop_contested(members: list[Member]) -> list[Member]:
    """Removes landmarks that `(subject, time)` cannot identify uniquely.

    Args:
        members (list[Member]): Loaded bundles, all covering the same cohort.

    Returns:
        list[Member]: The same members with the contested rows removed.
    """
    trimmed, dropped = [], 0
    for member in members:
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
