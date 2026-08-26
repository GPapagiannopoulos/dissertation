"""Scores the fine-tuned MOTOR encoder and the XGBoost baseline on one label set.

Both models go through `training.binary_metrics`, so no metric definition can drift
between them, and both are scored on the **whole** validation fold rather than the
subsample the training loop uses for its periodic checks -- 445,814 landmarks against
the ~11,254 a 150-batch evaluation reaches. Comparing a number measured on 2.5% of a
fold against one measured on all of it would confound architecture with sampling.

Confidence intervals resample **subjects**, not landmarks. The 12-hourly grid puts
around nine highly correlated predictions inside one admission, so the fold's 445,814
rows carry the information of roughly 22,789 independent patients; a row-level
bootstrap would report an interval several times too narrow.

The headline number is the **paired** difference, not the two per-model intervals.
Those two overlap freely even when one model wins on nearly every resample, because
each carries the variance of the cohort; scoring both models on the same draw cancels
it. Pairing needs the two sides on the same rows, and neither pipeline knows the
other's row numbering, so `align_predictions` joins them on `(subject_id,
prediction_time)` -- the one key both derive independently from stage 4's landmarks --
and refuses to proceed if the two cohorts differ or disagree about a label.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import xgboost as xgb

from thesis.modelling.backbone.checkpoint import released_encoder, strip_compile_prefix
from thesis.modelling.backbone.tokenizer import (
    build_ancestor_expansion,
    load_token_table,
)
from thesis.modelling.baseline.model import predict_fold
from thesis.modelling.evaluation.intervals import bootstrap_interval, paired_interval
from thesis.modelling.evaluation.metrics import binary_metrics
from thesis.modelling.finetune.data import fold_subjects, iter_epoch
from thesis.modelling.finetune.head import MotorClassifier
from thesis.modelling.finetune.lora import load_adapter, lora_classifier, lora_config
from thesis.modelling.finetune.training import predict_stream

HEADLINE = ("auprc", "auroc", "brier", "ece", "precision_at_1pct", "recall_at_1pct")
"""The metrics reported side by side, discrimination first then calibration."""


MAX_CONTESTED_SHARE = 0.001
"""How much of a fold may be duplicated keys before the join is called broken.

Overlapping admissions are genuinely rare -- three landmarks in 4,082,007 on the
2026-08-21 grid, needing both an overlap and admit times congruent modulo the 12-hour
step. A percent of the fold means something else is wrong.
"""


def align_predictions(
    left: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    names: tuple[str, str] = ("left", "right"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Puts two models' predictions on the same rows, in the same order.

    The two pipelines produce their predictions in unrelated orders -- the baseline
    in `landmark_id` order within each shard, MOTOR in token-budget batch order --
    and neither carries the other's row identifier. `(subject_id, prediction_time)`
    is the one key both sides derive independently from stage 4's landmarks, so it is
    what they are joined on.

    The join is validated rather than trusted, because every failure here is silent:
    a duplicated key multiplies rows, a missing key drops a cohort, and a
    disagreement about a label means the two feature pipelines are not describing the
    same landmark at all. Each of those produces a plausible number.

    A key repeated on either side is dropped from BOTH, because a duplicate cannot be
    paired: two landmarks for one subject at one instant come from overlapping
    admissions in stage 4 -- MIMIC records an observation stay and an emergency
    admission with the same `admittime`, and stage 5.2 keeps no `visit_id` to tell
    them apart. Dropping is safe here only because the colliding landmarks share one
    patient history, so MOTOR emits the same score for both; they differ only in the
    horizon their own admission ends at. The count is printed, and more than
    `MAX_CONTESTED_SHARE` of the fold raises instead, since at that scale the cause is
    a broken join rather than a handful of overlapping admissions.

    Args:
        left (tuple): `(scores, targets, subjects, times)` for the first model.
        right (tuple): The same four arrays for the second.
        names (tuple[str, str]): What to call the two sides in error messages.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The left scores, the
            right scores, the shared targets and the subjects, all row-aligned.

    Raises:
        ValueError: If duplicated keys exceed `MAX_CONTESTED_SHARE` of either side, if
            the two do not cover exactly the same landmarks, or if they disagree on a
            label.
    """
    frames = []
    for (scores, targets, subjects, times), name in zip(
        (left, right), names, strict=True
    ):
        frame = pl.DataFrame(
            {
                "subject_id": subjects,
                "time": times,
                f"score_{name}": scores,
                f"target_{name}": targets,
            }
        )
        frames.append(frame)

    # the union across both sides, so the two are trimmed identically even when only
    # one of them carries the collision
    contested = pl.concat(
        [
            frame.select("subject_id", "time").filter(
                frame.select("subject_id", "time").is_duplicated()
            )
            for frame in frames
        ]
    ).unique()

    if contested.height:
        share = contested.height / min(frame.height for frame in frames)
        if share > MAX_CONTESTED_SHARE:
            raise ValueError(
                f"{contested.height:,} (subject, prediction_time) key(s) are "
                f"duplicated, {share:.2%} of the fold. That is too many to be "
                f"overlapping admissions; the two sides are probably not scoring the "
                f"same cohort."
            )
        print(
            f"dropping {contested.height:,} duplicated (subject, prediction_time) "
            f"key(s) from both sides -- overlapping admissions in stage 4",
            flush=True,
        )
        frames = [
            frame.join(contested, on=("subject_id", "time"), how="anti")
            for frame in frames
        ]

    joined = frames[0].join(frames[1], on=("subject_id", "time"), how="inner")
    if joined.height != frames[0].height or joined.height != frames[1].height:
        raise ValueError(
            f"The two models were scored on different landmarks: {names[0]} has "
            f"{frames[0].height:,}, {names[1]} has {frames[1].height:,}, and only "
            f"{joined.height:,} are common to both. A paired comparison needs one "
            f"cohort -- check that stage 5.2 placed every landmark stage 7a built."
        )

    disagreed = int(
        (joined[f"target_{names[0]}"] != joined[f"target_{names[1]}"]).sum()
    )
    if disagreed:
        raise ValueError(
            f"{disagreed:,} landmark(s) carry one label in {names[0]} and the other "
            f"in {names[1]}. The two feature pipelines disagree about the outcome "
            f"itself, so no comparison between them means anything."
        )

    return (
        joined[f"score_{names[0]}"].to_numpy(),
        joined[f"score_{names[1]}"].to_numpy(),
        joined[f"target_{names[0]}"].to_numpy(),
        joined["subject_id"].to_numpy(),
    )


def score_baseline(
    booster: Path,
    features: Path,
    *,
    fold: str = "validation",
    resamples: int = 200,
    seed: int = 0,
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Scores the fitted booster over a whole fold.

    Args:
        booster (Path): The folder `run_train_baseline` wrote.
        features (Path): The folder `run_build_features` wrote.
        fold (str): Which fold to score.
        resamples (int): Bootstrap draws for the AUPRC interval.
        seed (int): Seeds the bootstrap.

    Returns:
        tuple: The metrics (carrying `auprc_lo`/`auprc_hi`), then the
            `(scores, targets, subjects, times)` `align_predictions` takes, so a
            caller can pair or re-bootstrap these without predicting again.
    """
    model = xgb.Booster()
    model.load_model(booster / "booster.json")
    index = pl.read_parquet(booster / "code_index.parquet")

    scores, targets, subjects, times = predict_fold(model, features, index, fold=fold)
    metrics = binary_metrics(scores, targets)
    low, high = bootstrap_interval(
        scores, targets, subjects, resamples=resamples, seed=seed
    )
    metrics["auprc_lo"], metrics["auprc_hi"] = low, high
    metrics["n_subjects"] = float(np.unique(subjects).size)
    return metrics, (scores, targets, subjects, times)


def score_motor(
    checkpoint: Path,
    sequences: Path,
    split: Path,
    oracle: Path,
    dictionary: Path,
    *,
    fold: str = "validation",
    vocab_size: int = 65536,
    token_budget: int = 8192,
    device: torch.device | None = None,
    resamples: int = 200,
    seed: int = 0,
    lora: dict | None = None,
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Scores a saved MOTOR classifier over a whole fold.

    `max_batches=None` is the point of this function: the training loop's periodic
    evaluation stops at 150 batches, and the headline AUPRC 0.202 was measured there.
    This runs the fold to exhaustion so the two models are scored on the same labels.

    Args:
        checkpoint (Path): A `best.pt` or `last.pt` from stage 6.
        sequences (Path): Stage 5.2's output folder.
        split (Path): The subject split parquet.
        oracle (Path): The fp32 oracle dump the released weights come out of.
        dictionary (Path): MOTOR's msgpack dictionary.
        fold (str): Which fold to score.
        vocab_size (int): The token count the checkpoint's config declares.
        token_budget (int): The most padded positions one batch may hold.
        device (torch.device | None): Where to run. Defaults to cuda when present.
        resamples (int): Bootstrap draws for the AUPRC interval.
        seed (int): Seeds the bootstrap. The batch order is seeded separately and
            fixed, since a different batching would score the same labels anyway.
        lora (dict | None): A `lora.config_record` for an adapter checkpoint, which
            holds only the adapters and the head. None scores a full fine-tune.

    Returns:
        tuple: The metrics (`binary_metrics`, the mean loss and `auprc_lo`/
            `auprc_hi`), then the `(scores, targets, subjects, times)` that
            `align_predictions` takes.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = load_token_table(dictionary, vocab_size=vocab_size)
    expansion = build_ancestor_expansion(table).collect().lazy()
    subjects = fold_subjects(split, fold).collect().lazy()

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # scored eagerly whatever the run used, so a checkpoint written under
    # torch.compile has to have its `_orig_mod.` segments removed first
    weights = strip_compile_prefix(state["model"])

    # The bias is overwritten by the state dict, so the prevalence passed here only
    # has to be a legal probability.
    if lora is None:
        model = MotorClassifier(released_encoder(oracle), positive_rate=0.05)
        model.load_state_dict(weights)
    else:
        # the backbone comes from the oracle, not the file; only the adapters and
        # the head were saved
        model = lora_classifier(oracle, 0.05, lora_config(**lora))
        load_adapter(model, weights)
    model.to(device)

    batches = iter_epoch(
        sequences, expansion, subjects=subjects, token_budget=token_budget, seed=0
    )
    predictions = predict_stream(
        model,
        batches,
        device=device,
        amp_dtype=torch.bfloat16,
        max_batches=None,
    )

    metrics = binary_metrics(predictions.scores, predictions.targets)
    metrics["loss"] = predictions.loss
    low, high = bootstrap_interval(
        predictions.scores,
        predictions.targets,
        predictions.subjects,
        resamples=resamples,
        seed=seed,
    )
    metrics["auprc_lo"], metrics["auprc_hi"] = low, high
    metrics["n_subjects"] = float(np.unique(predictions.subjects).size)
    return metrics, (
        predictions.scores,
        predictions.targets,
        predictions.subjects,
        predictions.times,
    )


def format_table(
    rows: dict[str, dict[str, float]], delta: dict[str, float] | None = None
) -> str:
    """Renders the comparison as a fixed-width table.

    Args:
        rows (dict[str, dict[str, float]]): Model name to its metrics.
        delta (dict[str, float] | None): The paired difference, carrying `value`,
            `lo`, `hi` and the two model names as `left` and `right`. Omitted when
            the two models were not scored on a common cohort.

    Returns:
        str: The table, one model per column, with the paired difference underneath
            it -- that last line is the comparison; the columns are only its inputs.
    """
    names = list(rows)
    # wide enough for a "[0.1234, 0.5678]" interval, not just for the widest name
    width = max(max(len(name) for name in names) + 2, 18)

    lines = ["metric".ljust(20) + "".join(name.rjust(width) for name in names)]
    for key in HEADLINE:
        cells = "".join(
            f"{rows[name].get(key, float('nan')):>{width}.4f}" for name in names
        )
        lines.append(key.ljust(20) + cells)

    intervals = "".join(
        f"[{rows[name].get('auprc_lo', float('nan')):.4f},"
        f" {rows[name].get('auprc_hi', float('nan')):.4f}]".rjust(width)
        for name in names
    )
    lines.append("auprc 95% CI".ljust(20) + intervals)

    for key in ("base_rate", "n", "n_positive", "n_subjects"):
        cells = "".join(
            f"{rows[name].get(key, float('nan')):>{width},.0f}" for name in names
        )
        lines.append(key.ljust(20) + cells)

    if delta is not None:
        lines.append("")
        lines.append(
            f"auprc {delta['left']} - {delta['right']}: {delta['value']:+.4f} "
            f"[{delta['lo']:+.4f}, {delta['hi']:+.4f}] "
            f"({'excludes' if delta['lo'] > 0 or delta['hi'] < 0 else 'includes'} "
            f"zero)"
        )
    return "\n".join(lines)


def run_comparison(
    booster: Path,
    features: Path,
    checkpoint: Path,
    sequences: Path,
    split: Path,
    oracle: Path,
    dictionary: Path,
    dest: Path,
    *,
    fold: str = "validation",
    resamples: int = 200,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Scores both models on one fold and writes the comparison.

    Args:
        booster (Path): The folder `run_train_baseline` wrote.
        features (Path): The folder `run_build_features` wrote.
        checkpoint (Path): The MOTOR checkpoint to score.
        sequences (Path): Stage 5.2's output folder.
        split (Path): The subject split parquet.
        oracle (Path): The fp32 oracle dump.
        dictionary (Path): MOTOR's msgpack dictionary.
        dest (Path): Where to write `comparison.json`.
        fold (str): Which fold to score.
        resamples (int): Bootstrap draws for the baseline's AUPRC interval.
        seed (int): Seeds the bootstrap.

    Returns:
        dict[str, dict[str, float]]: Model name to its metrics.

    Raises:
        ValueError: If the two models did not score the same number of labels, which
            means they are not being compared on the same cohort.
    """
    print(f"scoring the XGBoost baseline on the {fold} fold", flush=True)
    tree, tree_rows = score_baseline(
        booster, features, fold=fold, resamples=resamples, seed=seed
    )

    print(f"scoring the MOTOR classifier on the {fold} fold", flush=True)
    motor, motor_rows = score_motor(
        checkpoint,
        sequences,
        split,
        oracle,
        dictionary,
        fold=fold,
        resamples=resamples,
        seed=seed,
    )

    # cheap and specific: a count mismatch is the likely failure and says so plainly,
    # where the join would report it as a shortfall in the intersection
    if int(tree["n"]) != int(motor["n"]):
        raise ValueError(
            f"The baseline scored {int(tree['n']):,} labels and MOTOR scored "
            f"{int(motor['n']):,}. They are not on the same cohort, so the "
            f"comparison is meaningless -- check that stage 5.2 placed every "
            f"landmark."
        )

    print("pairing the two models on (subject, prediction_time)", flush=True)
    motor_scores, tree_scores, targets, subjects = align_predictions(
        motor_rows, tree_rows, names=("motor", "xgboost")
    )
    value, low, high = paired_interval(
        motor_scores,
        tree_scores,
        targets,
        subjects,
        resamples=resamples,
        seed=seed,
    )
    delta = {
        "left": "motor",
        "right": "xgboost",
        "metric": "auprc",
        "value": value,
        "lo": low,
        "hi": high,
    }

    rows = {"xgboost": tree, "motor": motor}
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "comparison.json").write_text(
        json.dumps(
            {
                "fold": fold,
                "seed": seed,
                "resamples": resamples,
                "models": rows,
                "delta": delta,
            },
            indent=2,
        )
    )
    print("\n" + format_table(rows, delta))
    return rows
