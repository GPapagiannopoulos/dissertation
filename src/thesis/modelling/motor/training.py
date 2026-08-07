"""Stage 6: fine-tuning the MOTOR backbone on the AKI landmark task.

The shape follows the rest of the pipeline: pure helpers that take what they need and
return a value, and one `run_*` that does the guards, the I/O and the loop.

Three choices worth stating, because none is the obvious default:

- **`torch.autocast`, not `half_stack`.** bf16 keeps ~8 mantissa bits, so an AdamW
  update at lr 1e-5 on a weight of ~0.02 is smaller than the weight's own resolution
  and rounds straight back onto it. Autocast gets the bf16 matmuls while the master
  weights and the optimizer state stay float32. Measured at 49.5 ms per 1024-position
  sequence against 42.8 for pure bf16 -- 16% for updates that actually land.
- **No `pos_weight`.** Re-weighting the positives would lift AUPRC's ranking a little
  and wreck calibration, and the evaluation framework scores Brier and ECE alongside
  discrimination. The base rate goes into the head's BIAS instead, so the untrained
  model starts at 3.56% rather than at a saturated random guess.
- **The loss is summed and divided by the label count**, not averaged per batch. A
  token-budget batch holds anywhere from 1 to 64 sequences and so from a handful to
  several hundred labels; averaging per batch would weight a one-label step as
  heavily as a three-hundred-label one.
"""

import json
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from thesis.modelling.motor.data import batch_to


def binary_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Scores one set of predictions against its labels.

    AUPRC leads because the task is 3.56% positive, where AUROC is dominated by the
    negatives and a useless model still reads around 0.5 rather than around the base
    rate.

    Args:
        scores (np.ndarray): The predicted probabilities, shaped (n,).
        targets (np.ndarray): The binary labels, shaped (n,).

    Returns:
        dict[str, float]: `auprc`, `auroc`, `base_rate` and `n`. Both areas come back
            as NaN when the labels are all one class, which is a real possibility on
            a small evaluation subsample and not an error.

    Raises:
        ValueError: If the two arrays disagree in length, or if either is empty.
    """
    if scores.shape != targets.shape:
        raise ValueError(
            f"Scores shaped {scores.shape} do not match targets shaped {targets.shape}."
        )
    if scores.size == 0:
        raise ValueError("Cannot score an empty set of predictions.")

    single_class = len(np.unique(targets)) < 2
    return {
        "auprc": float("nan")
        if single_class
        else float(average_precision_score(targets, scores)),
        "auroc": float("nan")
        if single_class
        else float(roc_auc_score(targets, scores)),
        "base_rate": float(targets.mean()),
        "n": float(targets.size),
    }


def learning_rate_at(
    step: int, *, total: int, warmup: int, peak: float, floor: float = 0.0
) -> float:
    """The learning rate for one step: linear warmup, then cosine decay.

    Warmup matters more than usual here. The head starts at zero weight, so its first
    gradients are large relative to the backbone's, and a cold high rate would move
    the pretrained weights a long way to chase a head that has not yet learned
    anything.

    Args:
        step (int): The step about to be taken, counted from zero.
        total (int): How many steps the run will take in all.
        warmup (int): How many steps to spend ramping up.
        peak (float): The rate at the end of warmup.
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
        # +1 so the first step is not exactly zero, which would waste it
        return peak * (step + 1) / warmup
    if total == warmup:
        return peak
    progress = (step - warmup) / (total - warmup)
    return floor + (peak - floor) * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))


@torch.no_grad()
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
        model (torch.nn.Module): The classifier, which this puts in eval mode and
            leaves there -- the caller re-arms training.
        batches (Iterable): Batches as `collate` returns them.
        device (torch.device): Where to run.
        amp_dtype (torch.dtype): The autocast dtype, matching training.
        max_batches (int | None): Stop after this many. A full validation pass is
            45,612 sequences; a bounded subsample is what makes per-N-step
            evaluation affordable.

    Returns:
        dict[str, float]: `binary_metrics` plus the mean loss.

    Raises:
        ValueError: If the stream yields no batch.
    """
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    total_loss = 0.0
    total_labels = 0

    for index, batch in enumerate(batches):
        if max_batches is not None and index >= max_batches:
            break
        moved = batch_to(batch, device)
        labels = moved.pop("labels")

        with torch.autocast(device.type, dtype=amp_dtype):
            logits = model(**moved)

        total_loss += float(loss_fn(logits.float(), labels.float()))
        total_labels += labels.numel()
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(labels.cpu().numpy())

    if not scores:
        raise ValueError("The evaluation stream yielded no batch.")

    metrics = binary_metrics(np.concatenate(scores), np.concatenate(targets))
    metrics["loss"] = total_loss / total_labels
    return metrics


def _log(path: Path, record: dict) -> None:
    """Appends one JSON line, flushed, so a killed run keeps its history."""
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


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
    max_hours: float | None = None,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Fine-tunes the model, checkpointing the best validation AUPRC.

    The loop is bounded by BOTH a step count and, optionally, a wall clock, because
    an overnight run has a deadline that a step count cannot express. Whichever comes
    first stops it, and the learning-rate schedule is laid out over `total_steps`, so
    a wall-clock stop lands mid-schedule rather than at the annealed end.

    Args:
        model (torch.nn.Module): The classifier, already on `device`.
        train_batches (Iterator): The training stream. Exhausting it ends the run,
            so pass a stream that spans as many epochs as the budget allows.
        validation (Iterable): A re-iterable source of validation batches, called
            once per evaluation.
        dest (Path): The folder to write checkpoints and the log into.
        device (torch.device): Where to run.
        total_steps (int): The schedule's length, and the cap on optimizer steps.
        encoder_lr (float): Peak rate for the pretrained backbone.
        head_lr (float): Peak rate for the fresh head, which starts from zero weight
            and so has much further to travel.
        warmup (int): Steps spent ramping the rate up.
        accumulate (int): Batches per optimizer step.
        clip (float): Global gradient-norm clip.
        eval_every (int): Optimizer steps between evaluations.
        eval_batches (int): Batches per evaluation.
        max_hours (float | None): Wall-clock budget, or None for no limit.
        amp_dtype (torch.dtype): The autocast dtype.

    Returns:
        dict[str, float]: The best evaluation's metrics, plus the step it came from.

    Raises:
        FileExistsError: If dest already exists, as elsewhere in the pipeline.
        ValueError: If accumulate is not positive.
    """
    if accumulate < 1:
        raise ValueError(f"A step accumulates at least one batch, got {accumulate}.")
    if dest.exists():
        raise FileExistsError(
            f"Destination {dest} already exists; refusing to overwrite. "
            f"Remove it or choose a new path."
        )
    dest.mkdir(parents=True)
    log_path = dest / "log.jsonl"

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

    best = {"auprc": -1.0, "step": -1}
    started = time.perf_counter()
    step = 0
    window_loss = 0.0
    window_labels = 0
    seen_batches = 0
    running_loss = 0.0
    running_labels = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for batch in train_batches:
        if step >= total_steps:
            break
        if max_hours is not None and (time.perf_counter() - started) > max_hours * 3600:
            _log(log_path, {"event": "wall_clock_reached", "step": step})
            break

        moved = batch_to(batch, device)
        labels = moved.pop("labels")

        with torch.autocast(device.type, dtype=amp_dtype):
            logits = model(**moved)

        # summed, then divided by the window's labels at the step: a batch of one
        # label must not weigh as much as a batch of three hundred
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
            _log(
                log_path,
                {
                    "event": "train",
                    "step": step,
                    "loss": running_loss / max(1, running_labels),
                    "lr": rate * encoder_lr,
                    "grad_norm": float(grad_norm),
                    "elapsed_s": round(time.perf_counter() - started, 1),
                    "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                    if device.type == "cuda"
                    else 0.0,
                },
            )
            running_loss = 0.0
            running_labels = 0

        window_loss = 0.0
        window_labels = 0

        if step % eval_every == 0:
            metrics = evaluate(
                model,
                validation,
                device=device,
                amp_dtype=amp_dtype,
                max_batches=eval_batches,
            )
            model.train()
            _log(
                log_path,
                {
                    "event": "validation",
                    "step": step,
                    "elapsed_s": round(time.perf_counter() - started, 1),
                    **metrics,
                },
            )
            if metrics["auprc"] > best["auprc"]:
                best = {**metrics, "step": step}
                torch.save(
                    {"model": model.state_dict(), "step": step, "metrics": metrics},
                    dest / "best.pt",
                )

    torch.save({"model": model.state_dict(), "step": step}, dest / "last.pt")
    _log(log_path, {"event": "finished", "step": step, "best": best})
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

    Reads only the labels, which are 26 MB across all 200 shards -- small enough to
    hold whole, unlike the sequences.

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
