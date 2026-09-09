"""This module contains the linear probe.

The probe is used as a baseline for the contributions of fine-tuning.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import torch

from thesis.modelling.evaluation.metrics import binary_metrics
from thesis.modelling.finetune.batching import LABEL_METADATA
from thesis.modelling.finetune.data import batch_to

# precision of cached features
FEATURE_DTYPE = np.float16


class ProbeArrays(NamedTuple):
    """One fold's cached features and everything needed to score them.

    Attributes:
        features (np.ndarray): The encoder's output at each labelled position,
            shaped (n_labels, hidden_size).
        targets (np.ndarray): The binary labels.
        subjects (np.ndarray): The subject each label belongs to, which is the
            unit a confidence interval on this task has to resample.
        times (np.ndarray): Each label's prediction time in microseconds since the
            epoch. With `subjects` it is the key that pairs a probe prediction
            against the fine-tune's or XGBoost's.
    """

    features: np.ndarray
    targets: np.ndarray
    subjects: np.ndarray
    times: np.ndarray


def fold_label_count(sequences: Path, split: Path, fold: str) -> int:
    """Label count for a fold.

    Args:
        sequences (Path): Path to the sequences directory.
        split (Path): Path to the subject split parquet file.
        fold (str): Which fold to count.

    Returns:
        int: The number of labelled positions the fold contributes.

    Raises:
        ValueError: If the fold holds no labels.
    """
    count = int(
        pl.scan_parquet(sequences / "labels" / "*.parquet")
        .join(
            pl.scan_parquet(split).filter(pl.col("fold") == fold).select("subject_id"),
            on="subject_id",
            how="inner",
        )
        .select(pl.len())
        .collect()
        .item()
    )
    if not count:
        raise ValueError(f"Fold {fold!r} holds no label; check the spelling.")
    return count


@torch.no_grad()
def extract_features(
    encoder: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor | int]],
    destination: np.ndarray,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> ProbeArrays:
    """Runs the frozen encoder and keeps its output at every labelled position.

    Args:
        encoder (torch.nn.Module): The backbone in eval mode.
        batches (Iterable): Batches as `collate` returns them.
        destination (np.ndarray): A pre-sized (n_labels, hidden) array to fill.
        device (torch.device): Device on which to run.
        amp_dtype (torch.dtype): The autocast dtype, matching how the fine-tune
            ran so the two are measured on comparable activations.

    Returns:
        ProbeArrays: A view of the filled rows, with the labels and provenance.

    Raises:
        ValueError: If the stream yields more labels than `destination` holds,
            which means the count and the stream disagree about the fold.
    """
    encoder.eval()

    targets: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    times: list[np.ndarray] = []
    filled = 0

    for batch in batches:
        moved = batch_to(batch, device)
        supervision = {key: moved.pop(key) for key in LABEL_METADATA}
        label_indices = moved.pop("label_indices")
        # the clock is a head input, and this runs the BARE encoder, which would
        # reject it at the splat
        moved.pop("label_clocks", None)

        with torch.autocast(device.type, dtype=amp_dtype):
            features = encoder(**moved)

        picked = features.flatten(0, -2).index_select(0, label_indices).float()

        taken = picked.shape[0]
        if filled + taken > destination.shape[0]:
            raise ValueError(
                f"The stream has produced {filled + taken} labels but the cache was "
                f"sized for {destination.shape[0]}. The fold count and the batch "
                f"stream disagree."
            )

        destination[filled : filled + taken] = (
            picked.cpu().numpy().astype(FEATURE_DTYPE)
        )
        filled += taken

        targets.append(supervision["labels"].cpu().numpy())
        subjects.append(supervision["label_subjects"].cpu().numpy())
        times.append(supervision["label_times"].cpu().numpy())

    if not targets:
        raise ValueError("The feature stream yielded no batch.")

    return ProbeArrays(
        features=destination[:filled],
        targets=np.concatenate(targets),
        subjects=np.concatenate(subjects),
        times=np.concatenate(times),
    )


def _minibatches(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Streams the cached table to the device, optionally shuffled."""
    order = np.arange(features.shape[0])
    if rng is not None:
        rng.shuffle(order)

    for start in range(0, order.size, batch_size):
        rows = np.sort(order[start : start + batch_size])
        yield (
            torch.from_numpy(features[rows].astype(np.float32)).to(device),
            torch.from_numpy(targets[rows].astype(np.float32)).to(device),
        )


@torch.no_grad()
def _evaluate_head(
    head: torch.nn.Module,
    arrays: ProbeArrays,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """Scores a head over a whole fold, returning probabilities and mean loss."""
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    scores: list[np.ndarray] = []
    total = 0.0

    for inputs, labels in _minibatches(
        arrays.features, arrays.targets, batch_size=batch_size, device=device
    ):
        logits = head(inputs).squeeze(-1)
        total += float(loss_fn(logits, labels))
        scores.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(scores), total / arrays.targets.size


def fit_probe(
    train: ProbeArrays,
    validation: ProbeArrays,
    *,
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 8192,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[torch.nn.Linear, dict[str, float]]:
    """Fits one linear head on cached features, selecting on validation loss.

    Args:
        train (ProbeArrays): The training fold's cached features.
        validation (ProbeArrays): The validation fold's.
        device (torch.device): Where to run.
        epochs (int): Passes over the training fold.
        batch_size (int): Rows per gradient step.
        learning_rate (float): AdamW's rate. A linear model on standardised-ish
            activations tolerates a far higher rate than the backbone does.
        weight_decay (float): L2 pull toward zero. Unlike the fine-tune's, this
            one bites: at 1e-3 the per-step shrink is 100x the encoder's.
        seed (int): Seeds the shuffle and the initialisation.
        verbose (bool): Whether to print per-epoch progress.

    Returns:
        tuple[torch.nn.Linear, dict[str, float]]: The best head by validation
            loss, and that epoch's metrics.

    Raises:
        ValueError: If either fold is empty, or the two disagree on width.
    """
    if not train.targets.size or not validation.targets.size:
        raise ValueError("Both folds need at least one label to fit a probe.")
    if train.features.shape[1] != validation.features.shape[1]:
        raise ValueError(
            f"Training features are {train.features.shape[1]} wide but validation "
            f"features are {validation.features.shape[1]}."
        )

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    head = torch.nn.Linear(train.features.shape[1], 1).to(device)
    rate = float(train.targets.mean())
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.constant_(head.bias, float(np.log(rate / (1.0 - rate))))

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_state = {k: v.clone() for k, v in head.state_dict().items()}
    best: dict[str, float] = {"loss": float("inf")}

    for epoch in range(epochs):
        head.train()
        for inputs, labels in _minibatches(
            train.features,
            train.targets,
            batch_size=batch_size,
            device=device,
            rng=rng,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss_fn(head(inputs).squeeze(-1), labels).backward()
            optimizer.step()

        head.eval()
        scores, loss = _evaluate_head(
            head, validation, batch_size=batch_size, device=device
        )
        metrics = binary_metrics(scores, validation.targets)
        metrics["loss"] = loss
        metrics["epoch"] = float(epoch + 1)

        improved = loss < best["loss"]
        if improved:
            best = metrics
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

        if verbose:
            print(
                f"  epoch {epoch + 1:>3}/{epochs} | loss {loss:.5f} | "
                f"auprc {metrics['auprc']:.4f} | auroc {metrics['auroc']:.4f}"
                f"{'  <- best' if improved else ''}",
                flush=True,
            )

    head.load_state_dict(best_state)
    return head, best


def run_fit_probe(
    train: ProbeArrays,
    validation: ProbeArrays,
    dest: Path,
    *,
    device: torch.device,
    weight_decays: tuple[float, ...] = (0.0, 1e-4, 1e-2),
    epochs: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Sweeps regularisation strength and keeps the best head by validation loss.

    Args:
        train (ProbeArrays): The training fold's cached features.
        validation (ProbeArrays): The validation fold's.
        dest (Path): Folder to write `probe.pt` and `probe_metrics.json` into.
        device (torch.device): Where to run.
        weight_decays (tuple[float, ...]): The settings to try.
        epochs (int): Passes per setting.
        seed (int): Seeds every fit identically, so the sweep compares settings
            rather than initialisations.

    Returns:
        dict[str, float]: The winning setting's validation metrics.

    Raises:
        ValueError: If no setting is given, which would leave nothing to save.
    """
    if not weight_decays:
        raise ValueError("A sweep needs at least one weight decay to try.")
    dest.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float]] = []
    best_head: torch.nn.Linear | None = None
    best: dict[str, float] = {"loss": float("inf")}

    for decay in weight_decays:
        print(f"weight_decay {decay:g}", flush=True)
        head, metrics = fit_probe(
            train,
            validation,
            device=device,
            epochs=epochs,
            weight_decay=decay,
            seed=seed,
        )
        metrics["weight_decay"] = decay
        results.append(metrics)
        if metrics["loss"] < best["loss"]:
            best, best_head = metrics, head

    if best_head is None:  # unreachable: the guard above forces one iteration
        raise ValueError("No head was fitted; the weight-decay sweep ran empty.")

    torch.save({"head": best_head.state_dict(), "metrics": best}, dest / "probe.pt")
    (dest / "probe_metrics.json").write_text(
        json.dumps({"best": best, "sweep": results}, indent=2)
    )
    return best
