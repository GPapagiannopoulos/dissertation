"""This module is responsible for fine-tuning the PyTorch MOTOR port."""

import json
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import torch

from thesis.modelling.backbone.checkpoint import strip_compile_prefix
from thesis.modelling.evaluation.metrics import binary_metrics
from thesis.modelling.finetune.batching import LABEL_METADATA
from thesis.modelling.finetune.data import batch_to


def learning_rate_at(
    step: int, *, total: int, warmup: int, peak: float, floor: float = 0.0
) -> float:
    """The learning rate for one step: linear warmup, then cosine decay.

    The head starts at zero weight, so its first gradients are large relative to the
    backbone's. Warm-up is used to prevent early fluctuations.

    Args:
        step (int): The step number to be taken, counted from zero.
        total (int): Total number of steps for the training run.
        warmup (int): Total number of warmup steps.
        peak (float): The learning rate at the end of warmup.
        floor (float): The rate the cosine decays to.

    Returns:
        float: The rate to set before this step.

    Raises:
        ValueError: If total is not positive, or warmup exceeds it.
    """
    if total < 1:
        raise ValueError(f"A run takes at least one step, got {total}.")
    if not 0 <= warmup <= total:
        raise ValueError(f"A {warmup} step warmup does not fit in {total} steps.")

    if step < warmup:
        # +1 so the first step is not exactly zero
        return peak * (step + 1) / warmup
    if total == warmup:
        return peak
    progress = (step - warmup) / (total - warmup)
    return floor + (peak - floor) * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))


class Predictions(NamedTuple):
    """One pass' predictions.

    Attributes:
        scores (np.ndarray): Predicted probabilities for each landmark.
        targets (np.ndarray): The binary labels of each landmark.
        subjects (np.ndarray): The subject each label belongs to used in
            resampling.
        times (np.ndarray): Each label's prediction time, in microseconds since the
            epoch. With `subjects` it identifies a landmark, so another model's
            predictions can be aligned against these row for row.
        loss (float): Mean BCE over the labels.
    """

    scores: np.ndarray
    targets: np.ndarray
    subjects: np.ndarray
    times: np.ndarray
    loss: float


@torch.no_grad()
def predict_stream(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor | int]],
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
) -> Predictions:
    """Runs the model over a stream of batches, storing prediction.

    Args:
        model (torch.nn.Module): The classifier in eval mode.
        batches (Iterable): Batches as `collate` returns them.
        device (torch.device): Device on which to run.
        amp_dtype (torch.dtype): The autocast dtype, matching training.
        max_batches (int | None): Stop after this many, or None to exhaust the
            stream.

    Returns:
        Predictions: Custom container for the predicted scores, their labels,
            subjects, prediction times, and loss.

    Raises:
        ValueError: If the stream yields no batch.
    """
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    times: list[np.ndarray] = []
    total_loss = 0.0
    total_labels = 0

    for index, batch in enumerate(batches):
        if max_batches is not None and index >= max_batches:
            break
        moved = batch_to(batch, device)
        supervision = {key: moved.pop(key) for key in LABEL_METADATA}
        labels = supervision["labels"]

        with torch.autocast(device.type, dtype=amp_dtype):
            logits = model(**moved)

        total_loss += float(loss_fn(logits.float(), labels.float()))
        total_labels += labels.numel()
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(labels.cpu().numpy())
        subjects.append(supervision["label_subjects"].cpu().numpy())
        times.append(supervision["label_times"].cpu().numpy())

    if not scores:
        raise ValueError("The evaluation stream yielded no batch.")

    return Predictions(
        scores=np.concatenate(scores),
        targets=np.concatenate(targets),
        subjects=np.concatenate(subjects),
        times=np.concatenate(times),
        loss=total_loss / total_labels,
    )


def evaluate(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor | int]],
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Scores the model over a stream of batches.

    Args:
        model (torch.nn.Module): The classifier in eval mode.
        batches (Iterable): Batches as `collate` returns them.
        device (torch.device): Device on which to run.
        amp_dtype (torch.dtype): The autocast dtype, matching training.
        max_batches (int | None): Stop after this many, or None to
            exhaust the stream.

    Returns:
        dict[str, float]: `binary_metrics` plus the mean loss.

    Raises:
        ValueError: If the stream yields no batch.
    """
    predictions = predict_stream(
        model, batches, device=device, amp_dtype=amp_dtype, max_batches=max_batches
    )
    metrics = binary_metrics(predictions.scores, predictions.targets)
    metrics["loss"] = predictions.loss
    return metrics


def _log(path: Path, record: dict) -> None:
    """Appends one JSON line, flushed, so a killed run keeps its history."""
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def format_duration(seconds: float) -> str:
    """Seconds as `H:MM:SS`, for a run measured in hours.

    Args:
        seconds (float): A duration. Negatives clamp to zero, which is what an
            estimate built from a not-yet-measured rate can produce.

    Returns:
        str: The duration, hours unpadded so the field grows rather than truncates.
    """
    whole = max(0, int(seconds))
    return f"{whole // 3600}:{whole // 60 % 60:02d}:{whole % 60:02d}"


def format_train_line(record: dict, *, total_steps: int) -> str:
    """One progress line for the terminal, from a `train` log record.

    Args:
        record (dict): A `train` record as `run_training` writes it.
        total_steps (int): The run's step budget, for the ETA and the counter.

    Returns:
        str: A single line, no trailing newline.
    """
    step = record["step"]
    per_step = record["elapsed_s"] / max(1, step)
    return (
        f"step {step:>6}/{total_steps} | "
        f"loss {record['loss']:.4f} | "
        f"lr {record['lr']:.2e} | "
        f"gnorm {record['grad_norm']:.2f} | "
        f"{per_step:.2f} s/step | "
        f"{record['peak_gib']:.1f} GiB | "
        f"{format_duration(record['elapsed_s'])} elapsed | "
        f"eta {format_duration(per_step * max(0, total_steps - step))}"
    )


def format_validation_line(record: dict, *, best: bool, stalled: int = 0) -> str:
    """Two lines of validation results for the terminal.

    Args:
        record (dict): A `validation` record as `run_training` writes it.
        best (bool): Whether this evaluation improved on every earlier one.
        stalled (int): How many evaluations have passed without an improvement.

    Returns:
        str: Two lines joined by a newline, no trailing newline.
    """
    rate = record["base_rate"]
    lift = record["auprc"] / rate if rate else float("nan")
    mark = "  <- best" if best else (f"  ({stalled} stalled)" if stalled else "")
    return (
        f"  val {record['step']:>6} | "
        f"auprc {record['auprc']:.4f} ({lift:.1f}x base) | "
        f"auroc {record['auroc']:.4f} | "
        f"loss {record['loss']:.4f}{mark}\n"
        f"             | "
        f"brier {record['brier']:.4f} | ece {record['ece']:.4f} | "
        f"p@1% {record['precision_at_1pct']:.3f} "
        f"r@1% {record['recall_at_1pct']:.3f} "
        f"p@5% {record['precision_at_5pct']:.3f} | "
        f"n {record['n']:.0f} ({record['n_positive']:.0f}+) | "
        f"{record['eval_s']:.1f}s"
    )


def run_training(
    model: torch.nn.Module,
    train_batches: Iterator[dict[str, torch.Tensor | int]],
    validation: Iterable[dict[str, torch.Tensor | int]],
    dest: Path,
    *,
    device: torch.device,
    total_steps: int,
    encoder_lr: float = 1e-5,
    head_lr: float = 1e-4,
    warmup: int = 300,
    accumulate: int = 4,
    clip: float = 1.0,
    eval_every: int = 500,
    eval_batches: int = 150,
    checkpoint_every: int = 4,
    patience: int | None = None,
    min_delta: float = 0.0,
    max_hours: float | None = None,
    amp_dtype: torch.dtype = torch.bfloat16,
    save_state: Callable[[torch.nn.Module], dict[str, torch.Tensor]] | None = None,
    checkpoint_extra: dict | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Fine-tunes the classifier storing checkpoints at set step counts.

    The loop is bounded either by step count, wall clock, or patience. The
    schedule is laid by step count. Other mechanisms of stopping do not account
    for this, possibly ending the run at a high learning rate.

    Patience counts evaluations.

    `min_delta` is the floor a loss drop must clear to reset the patience counter.

    Args:
        model (torch.nn.Module): The classifier.
        train_batches (Iterator): The training stream. Exhausting it ends the run,
            so pass a stream that spans as many epochs as the budget allows.
        validation (Iterable): A re-iterable source of validation batches, called
            once per evaluation.
        dest (Path): The folder to write checkpoints and the log into.
        device (torch.device): Device on which to run.
        total_steps (int): The total number of training steps.
        encoder_lr (float): Peak rate for the pretrained backbone.
        head_lr (float): Peak rate for the fresh head.
        warmup (int): Total number of warmup steps.
        accumulate (int): Batches per optimizer step.
        clip (float): Global gradient-norm clip.
        eval_every (int): Optimizer steps between evaluations.
        eval_batches (int): Batches per evaluation.
        checkpoint_every (int): Number of steps between storing a checkpoint.
        patience (int | None): Stop after this many consecutive evaluations without
            a validation-loss improvement, or None to run the full budget.
        min_delta (float): How far validation loss must drop to reset patience.
        max_hours (float | None): Wall-clock budget, or None for no limit.
        amp_dtype (torch.dtype): The autocast dtype.
        save_state (Callable | None): What a checkpoint's `model` entry holds,
            defaulting to the whole state dict.
        checkpoint_extra (dict | None): Merged into every checkpoint, so a file
            describes what has to be rebuilt to load it.
        verbose (bool): Whether to mirror the log to stdout. True by default.

    Returns:
        dict[str, float]: The metrics and step for the lowest by-loss evaluation.

    Raises:
        FileExistsError: If dest already exists, as elsewhere in the pipeline.
        ValueError: If accumulate is not positive, if patience is not positive, if
            min_delta is negative, or if checkpoint_every is negative.
    """
    if accumulate < 1:
        raise ValueError(f"A step accumulates at least one batch, got {accumulate}.")
    if checkpoint_every < 0:
        raise ValueError(
            f"A checkpoint interval counts evaluations, got {checkpoint_every}."
        )
    if patience is not None and patience < 1:
        raise ValueError(f"Patience waits at least one evaluation, got {patience}.")
    if min_delta < 0.0:
        raise ValueError(
            f"An improvement threshold is not negative, got {min_delta}; a negative "
            f"one would count every decline as progress."
        )
    if dest.exists():
        raise FileExistsError(
            f"Destination {dest} already exists; refusing to overwrite. "
            f"Remove it or choose a new path."
        )
    dest.mkdir(parents=True)
    log_path = dest / "log.jsonl"
    save = save_state or (lambda module: strip_compile_prefix(module.state_dict()))
    extra = checkpoint_extra or {}

    head_names = {f"head.{name}" for name, _ in model.head.named_parameters()}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if n not in head_names
                ],
                "lr": encoder_lr,
                "name": "encoder",
            },
            {
                "params": [p for n, p in model.named_parameters() if n in head_names],
                "lr": head_lr,
                "name": "head",
            },
        ],
        weight_decay=0.01,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    best = {"loss": float("inf"), "step": -1}
    started = time.perf_counter()
    step = 0
    window_loss = 0.0
    window_labels = 0
    seen_batches = 0
    running_loss = 0.0
    running_labels = 0
    stalled = 0
    stop_reason = "step_budget"

    if verbose:
        budget = f"{total_steps} steps"
        if max_hours is not None:
            budget += f" or {max_hours}h"
        if patience is not None:
            budget += f", patience {patience} evals ({patience * eval_every} steps)"
        print(f"training: {budget}, evaluating every {eval_every}", flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for batch in train_batches:
        if step >= total_steps:
            break
        if max_hours is not None and (time.perf_counter() - started) > max_hours * 3600:
            stop_reason = "wall_clock"
            _log(log_path, {"event": "wall_clock_reached", "step": step})
            if verbose:
                print(f"stopping at step {step}: wall clock reached", flush=True)
            break

        moved = batch_to(batch, device)
        labels = {key: moved.pop(key) for key in LABEL_METADATA}["labels"]

        with torch.autocast(device.type, dtype=amp_dtype):
            logits = model(**moved)

        # summed, then divided by the window's labels at the step
        loss = loss_fn(logits.float(), labels.float())
        (loss / max(1, labels.numel() * accumulate)).backward()

        window_loss += float(loss.detach())
        window_labels += labels.numel()
        seen_batches += 1

        if seen_batches % accumulate:
            continue

        rate = learning_rate_at(step, total=total_steps, warmup=warmup, peak=1.0)
        for group, peak in zip(
            optimizer.param_groups, (encoder_lr, head_lr), strict=False
        ):
            group["lr"] = rate * peak

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        running_loss += window_loss
        running_labels += window_labels

        if step % 50 == 0:
            record = {
                "event": "train",
                "step": step,
                "loss": running_loss / max(1, running_labels),
                "lr": rate * encoder_lr,
                "grad_norm": float(grad_norm),
                "elapsed_s": round(time.perf_counter() - started, 1),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if device.type == "cuda"
                else 0.0,
            }
            _log(log_path, record)
            if verbose:
                print(format_train_line(record, total_steps=total_steps), flush=True)
            running_loss = 0.0
            running_labels = 0

        window_loss = 0.0
        window_labels = 0

        if step % eval_every == 0:
            eval_started = time.perf_counter()
            metrics = evaluate(
                model,
                validation,
                device=device,
                amp_dtype=amp_dtype,
                max_batches=eval_batches,
            )
            model.train()

            # The loop no longer selects checkpoints as subsample
            # scoring is unreliable
            improved = metrics["loss"] < best["loss"] - min_delta
            stalled = 0 if improved else stalled + 1

            record = {
                "event": "validation",
                "step": step,
                "elapsed_s": round(time.perf_counter() - started, 1),
                "eval_s": round(time.perf_counter() - eval_started, 1),
                "stalled": stalled,
                **metrics,
            }
            _log(log_path, record)
            if verbose:
                print(
                    format_validation_line(record, best=improved, stalled=stalled),
                    flush=True,
                )

            if improved:
                best = {**metrics, "step": step}

            if checkpoint_every and (step // eval_every) % checkpoint_every == 0:
                torch.save(
                    {**extra, "model": save(model), "step": step, "metrics": metrics},
                    dest / f"step_{step:06d}.pt",
                )

            if patience is not None and stalled >= patience:
                stop_reason = "patience"
                _log(
                    log_path,
                    {"event": "early_stop", "step": step, "stalled": stalled},
                )
                if verbose:
                    print(
                        f"stopping at step {step}: {stalled} evaluations without a "
                        f"loss drop of {min_delta:.4f}; best was step {best['step']}",
                        flush=True,
                    )
                break

    torch.save({**extra, "model": save(model), "step": step}, dest / "last.pt")
    _log(
        log_path,
        {"event": "finished", "step": step, "reason": stop_reason, "best": best},
    )
    if verbose:
        print(
            f"finished at step {step} ({stop_reason}) after "
            f"{format_duration(time.perf_counter() - started)}",
            flush=True,
        )
        print(f"best step {best['step']}: {json.dumps(best)}", flush=True)
    return best


def validation_stream(factory) -> Iterable[dict[str, torch.Tensor | int]]:
    """Wraps a zero-argument batch-iterator factory so it can be re-iterated.

    `evaluate` is called many times over one run, and a generator is spent after the
    first. This calls the factory again for each pass instead of holding the batches,
    which would defeat the point of streaming them.

    Args:
        factory: A callable returning a fresh iterator of batches.

    Returns:
        Iterable: An object that builds a new iterator each time it is iterated.
    """

    class _Restartable:
        def __iter__(self):
            return iter(factory())

    return _Restartable()


def positive_rate(labels: Path, split: Path, fold: str = "training") -> float:
    """The prevalence in one fold, for the head's bias initialisation.

    Args:
        labels (Path): Stage 5.2's `labels/` folder.
        split (Path): The subject split parquet.
        fold (str): Which fold to measure.

    Returns:
        float: The fraction of landmarks that are positive.
    """
    return float(
        pl.scan_parquet(labels / "*.parquet")
        .join(
            pl.scan_parquet(split).filter(pl.col("fold") == fold).select("subject_id"),
            on="subject_id",
            how="inner",
        )
        .select(pl.col("boolean_value").mean())
        .collect()
        .item()
    )
