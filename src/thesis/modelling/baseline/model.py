"""Fits the XGBoost baseline the fine-tuned MOTOR encoder is measured against.

The point of this model is to be a *fair* comparator, not a weak one. It reads the
same stage 2.6 events the transformer does, at the same 33% coverage, and it is given
the untokenised float where the transformer sees a numeric bin -- a small advantage,
deliberately left in its favour, because a handicapped baseline proves nothing.

What it cannot do is the whole point of the comparison: every landmark is an
independent row, so the trees never see the order of a patient's events, only the
last-observation summary this pipeline hands them.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from thesis.modelling.baseline.matrix import (
    build_code_index,
    build_dmatrix,
    feature_names,
    shard_matrix,
    shard_pairs,
)

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": ["aucpr", "auc", "logloss"],
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "min_child_weight": 10,
    "reg_lambda": 1.0,
}
"""Ordinary gradient-boosting defaults, with two deliberate omissions.

There is no `scale_pos_weight`. At a 3.56% base rate it would lift ranking a little
and wreck calibration, and the evaluation framework scores Brier and ECE -- the same
reasoning that put the base rate in `MotorClassifier`'s bias rather than in the loss.

`colsample_bytree` is 0.5 rather than the usual 0.8 because the matrix is ~26,800
columns of which any one landmark populates ~500; sampling harder decorrelates the
trees on a matrix that is 99.4% empty.
"""


def train_baseline(
    features: Path,
    index: pl.DataFrame,
    *,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
    base_score: float | None = None,
    seed: int = 0,
    nthread: int | None = None,
    max_bin: int = 256,
    cache: Path | None = None,
) -> tuple[xgb.Booster, dict[str, dict[str, list[float]]]]:
    """Fits the booster, early-stopping on validation AUPRC.

    The validation matrix is built with `ref=dtrain`, so both folds share one set of
    quantile boundaries. Without that the same creatinine falls in different buckets
    in the two folds and every score is quietly wrong -- no exception, just worse
    numbers that look plausible.

    Args:
        features (Path): The folder `run_build_features` wrote.
        index (pl.DataFrame): The frame `build_code_index` returned.
        params (dict | None): Booster parameters. Defaults to `DEFAULT_PARAMS`.
        num_boost_round (int): The ceiling; early stopping is what usually ends it.
        early_stopping_rounds (int): Rounds without an AUPRC gain before stopping.
        base_score (float | None): The prior probability, i.e. the training base
            rate. Mirrors the head's bias initialisation.
        seed (int): Seeds row and column subsampling.
        nthread (int | None): Worker threads. None lets xgboost choose.
        max_bin (int): Buckets per feature, shared by both folds.
        cache (Path | None): A folder for the on-disk page cache. None holds both
            matrices in memory, which the full feature table does not fit into.

    Returns:
        tuple[xgb.Booster, dict]: The booster and the per-round evaluation history.
    """
    names = feature_names(index)
    dtrain = build_dmatrix(
        features, index, fold="training", max_bin=max_bin, names=names, cache=cache
    )
    dvalid = build_dmatrix(
        features,
        index,
        fold="validation",
        ref=dtrain,
        max_bin=max_bin,
        names=names,
        cache=cache,
    )

    settings = dict(params or DEFAULT_PARAMS)
    settings["seed"] = seed
    # xgboost checks this against the matrices and aborts the first update if they
    # disagree -- `Inconsistent max_bin`, raised from the booster rather than from
    # either matrix, after the whole build has already been paid for.
    settings["max_bin"] = max_bin
    if nthread is not None:
        settings["nthread"] = nthread
    if base_score is not None:
        settings["base_score"] = base_score

    print(
        f"train {dtrain.num_row():,} x {dtrain.num_col():,} | "
        f"valid {dvalid.num_row():,} rows",
        flush=True,
    )

    history: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        settings,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=history,
        verbose_eval=25,
    )
    return booster, history


def training_base_rate(features: Path) -> float:
    """The training fold's prevalence, for `base_score`.

    Args:
        features (Path): The folder `run_build_features` wrote.

    Returns:
        float: The share of training landmarks that are positive.
    """
    spine = pl.scan_parquet(str(features / "spine" / "*.parquet"))
    return float(
        spine.filter(pl.col("fold") == "training")
        .select(pl.col("boolean_value").mean())
        .collect()
        .item()
    )


def run_train_baseline(
    features: Path,
    dest: Path,
    *,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
    seed: int = 0,
    nthread: int | None = None,
    max_bin: int = 256,
) -> Path:
    """Builds the matrices, fits the booster and writes the run's artifacts.

    The page cache goes to `dest / "cache"` and is removed when the run ends,
    whether or not it succeeded. It lives under dest because dest is a folder this
    function has just created and therefore owns -- pointing a cache at a path the
    caller supplied would mean deleting a folder we did not make.

    Args:
        features (Path): The folder `run_build_features` wrote.
        dest (Path): The folder to create, which must not already exist.
        num_boost_round (int): The ceiling on boosting rounds.
        early_stopping_rounds (int): Rounds without an AUPRC gain before stopping.
        seed (int): Seeds row and column subsampling.
        nthread (int | None): Worker threads.
        max_bin (int): Buckets per feature. Lowering it shrinks the per-node
            histogram, which at ~24,800 columns is the largest resident allocation
            left once the binned pages are on disk.

    Returns:
        Path: The folder written, holding `booster.json`, `code_index.parquet`,
            `history.json` and `summary.json`.

    Raises:
        FileExistsError: If dest already exists.
    """
    if dest.exists():
        raise FileExistsError(
            f"Destination {dest} already exists; refusing to overwrite. "
            f"Remove it or choose a new path."
        )
    dest.mkdir(parents=True)
    cache = dest / "cache"

    index = build_code_index(features)
    print(f"code index holds {index.height:,} codes", flush=True)

    prevalence = training_base_rate(features)
    print(f"training prevalence {prevalence:.4%}", flush=True)

    try:
        booster, history = train_baseline(
            features,
            index,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            base_score=prevalence,
            seed=seed,
            nthread=nthread,
            max_bin=max_bin,
            cache=cache,
        )
    finally:
        shutil.rmtree(cache, ignore_errors=True)

    booster.save_model(dest / "booster.json")
    index.write_parquet(dest / "code_index.parquet")
    (dest / "history.json").write_text(json.dumps(history))
    (dest / "summary.json").write_text(
        json.dumps(
            {
                "best_iteration": int(booster.best_iteration),
                "best_score": float(booster.best_score),
                "n_codes": int(index.height),
                "base_rate": prevalence,
                "max_bin": max_bin,
                "seed": seed,
            },
            indent=2,
        )
    )
    print(
        f"best iteration {booster.best_iteration} at valid-aucpr "
        f"{booster.best_score:.4f}",
        flush=True,
    )
    return dest


def predict_fold(
    booster: xgb.Booster,
    features: Path,
    index: pl.DataFrame,
    *,
    fold: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scores one fold, returning the pieces a paired comparison needs.

    Rows come back in `landmark_id` order within each shard, and the subject ids ride
    along because a bootstrap over this task must resample **subjects**: the 12-hourly
    grid puts ~9 highly correlated landmarks in one admission, so resampling rows
    would report an interval several times too narrow.

    The prediction times ride along for a different reason: `(subject_id,
    prediction_time)` is the only key this model and MOTOR share. `landmark_id` is
    this pipeline's own numbering and stage 5.2 never saw it, so pairing the two
    models on anything else would mean trusting two independent sort orders to agree.

    Args:
        booster (xgb.Booster): The fitted model.
        features (Path): The folder `run_build_features` wrote.
        index (pl.DataFrame): The code index the booster was fit with.
        fold (str): Which fold to score.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Scores, targets,
            subject ids, and prediction times as microseconds since the epoch.
    """
    names = feature_names(index)
    # `best_iteration` is an attribute set by early stopping, and it does not
    # reliably survive a save/load round-trip. Falling back to every round is the
    # safe direction: it scores the model as saved rather than raising here.
    best = getattr(booster, "best_iteration", None)
    window = (0, best + 1) if best is not None else (0, booster.num_boosted_rounds())
    print(f"scoring {fold} with rounds {window[0]}..{window[1]}", flush=True)

    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    times: list[np.ndarray] = []

    # Shard at a time, on a plain DMatrix. A QuantileDMatrix would be wrong here
    # twice over: its rows arrive in shard order rather than landmark_id order, so
    # a globally sorted roster would mislabel every score, and inference through a
    # freshly binned matrix is not inference on the values the splits were learnt on.
    for feature_path, spine_path in shard_pairs(features):
        batch = shard_matrix(feature_path, spine_path, index, fold=fold)
        if batch is None:
            continue
        block, labels = batch
        matrix = xgb.DMatrix(block, feature_names=names)
        scores.append(booster.predict(matrix, iteration_range=window))
        targets.append(labels)

        # sorted by landmark_id, exactly as `shard_matrix` numbered the rows it just
        # scored -- this is the same roster read twice, not two orders being trusted
        # to agree
        roster = (
            pl.scan_parquet(spine_path)
            .filter(pl.col("fold") == fold)
            .select("landmark_id", "subject_id", "prediction_time")
            .sort("landmark_id")
            .collect()
        )
        subjects.append(roster["subject_id"].to_numpy())
        times.append(roster["prediction_time"].dt.epoch("us").to_numpy())

    if not scores:
        raise ValueError(f"No shard held a landmark in fold {fold!r}.")
    return (
        np.concatenate(scores),
        np.concatenate(targets),
        np.concatenate(subjects),
        np.concatenate(times),
    )
