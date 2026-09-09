"""This module is responsible for the movement of data.

It handles the batching of sequences, the moving of data to different
devices, and the retrieval of different subject folds.
"""

import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import torch

from thesis.modelling.finetune.batching import collate

DEFAULT_TOKEN_BUDGET: int = 8192


def sequence_lengths(sequences: pl.LazyFrame) -> pl.LazyFrame:
    """Measures each sequence, in positions and in padded positions.

    Args:
        sequences (pl.LazyFrame): Rows as `build_sequences` writes them.

    Returns:
        pl.LazyFrame: One row per sequence, columns `sequence_id`, `length` and
            `padded`, where `padded` is the next power of two at or above `length`.
    """
    return (
        sequences.group_by("sequence_id")
        .agg((pl.col("position").max() + 1).alias("length"))
        .with_columns(
            padded=pl.when(pl.col("length") <= 1)
            .then(1)
            .otherwise(
                pl.lit(2).pow((pl.col("length") - 1).log(2).floor() + 1).cast(pl.Int64)
            )
        )
    )


def assign_batches(
    lengths: pl.DataFrame, *, token_budget: int, seed: int
) -> pl.DataFrame:
    """Groups sequences of like length into batches of a fixed token budget.

    Args:
        lengths (pl.DataFrame): One row per sequence, as `sequence_lengths` returns.
        token_budget (int): The most padded positions one batch may hold.
        seed (int): Seeds the shuffle, so an epoch is reproducible.

    Returns:
        pl.DataFrame: Columns `sequence_id` and `batch_index`.

    Raises:
        ValueError: If the token budget is not positive.
    """
    if token_budget < 1:
        raise ValueError(f"A batch holds at least one position, got {token_budget}.")

    # 'lengths' might arrive shuffled, so need to sort first
    shuffled = lengths.sort("sequence_id").sample(fraction=1.0, shuffle=True, seed=seed)

    return (
        shuffled.with_columns(
            capacity=pl.max_horizontal(
                pl.lit(1), (pl.lit(token_budget) // pl.col("padded"))
            )
        )
        .with_columns(rank=pl.int_range(pl.len()).over("padded"))
        .with_columns(local=pl.col("rank") // pl.col("capacity"))
        .with_columns(
            batch_index=pl.struct("padded", "local").rank("dense").cast(pl.UInt32) - 1
        )
        .select("sequence_id", "batch_index")
    )


def _labelled_sequences(labels: pl.LazyFrame) -> pl.LazyFrame:
    """The ids of the sequences carrying at least one label."""
    return labels.select("sequence_id").unique()


def shard_batches(
    sequences: pl.LazyFrame,
    labels: pl.LazyFrame,
    expansion: pl.LazyFrame,
    *,
    subjects: pl.LazyFrame,
    token_budget: int,
    seed: int,
) -> Iterator[dict[str, torch.Tensor | int]]:
    """Turns one shard into collated batches.

    Args:
        sequences (pl.LazyFrame): One shard of sequences.
        labels (pl.LazyFrame): The matching shard of placed labels.
        expansion (pl.LazyFrame): `build_ancestor_expansion`'s table.
        subjects (pl.LazyFrame): One column, `subject_id`
        token_budget (int): Passed to `assign_batches`.
        seed (int): Seeds this shard's shuffle.

    Yields:
        dict[str, torch.Tensor | int]: One batch, as `collate` returns it.
    """
    fold_labels = labels.join(subjects, on="subject_id", how="inner")
    kept = _labelled_sequences(fold_labels)

    fold_sequences = sequences.join(kept, on="sequence_id", how="inner")
    lengths = sequence_lengths(fold_sequences).collect()
    if lengths.height == 0:
        return

    batches = assign_batches(lengths, token_budget=token_budget, seed=seed).lazy()

    resolved = fold_sequences.join(batches, on="sequence_id", how="inner").collect()
    resolved_labels = fold_labels.join(batches, on="sequence_id", how="inner").collect()

    by_batch = {
        int(frame["batch_index"][0]): frame
        for frame in resolved.partition_by("batch_index")
    }
    labels_by_batch = {
        int(frame["batch_index"][0]): frame
        for frame in resolved_labels.partition_by("batch_index")
    }

    for batch_index in sorted(by_batch):
        yield collate(
            by_batch[batch_index].drop("batch_index"),
            labels_by_batch[batch_index].drop("batch_index"),
            expansion,
        )


def iter_epoch(
    root: Path,
    expansion: pl.LazyFrame,
    *,
    subjects: pl.LazyFrame,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    seed: int = 0,
    prefetch: int = 3,
) -> Iterator[dict[str, torch.Tensor | int]]:
    """Streams one epoch's batches by shard.

    Args:
        root (Path): Path to parent directory of `sequences/` and `labels/`.
        expansion (pl.LazyFrame): `build_ancestor_expansion`'s table.
        subjects (pl.LazyFrame): One column, `subject_id`.
        token_budget (int): The maximum number of positions per batch.
        seed (int): Seeds the shard order and every shard's shuffle.
        prefetch (int): Number of batches to pre-emptively prepare.

    Yields:
        dict[str, torch.Tensor | int]: One batch, as `collate` returns it.

    Raises:
        FileNotFoundError: If the folder holds no shard, or a sequence shard has no
            matching label shard.
    """
    shards = sorted((root / "sequences").glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"Found no parquet shards in {root / 'sequences'}. Point root at stage "
            f"5.2's output folder."
        )
    for shard in shards:
        if not (root / "labels" / shard.name).is_file():
            raise FileNotFoundError(
                f"Sequence shard {shard.name} has no matching label shard in "
                f"{root / 'labels'}."
            )

    order = np.random.default_rng(seed).permutation(len(shards))
    ready: queue.Queue = queue.Queue(maxsize=prefetch)
    done = object()
    stop = threading.Event()

    def offer(item: object) -> bool:
        """Offers an object to the consumer."""
        while not stop.is_set():
            try:
                ready.put(item, timeout=0.2)  # deadlock guard
            except queue.Full:
                continue
            return True
        return False

    def produce() -> None:
        """Reads shards in the epoch's order and fills the queue."""
        try:
            for position, index in enumerate(order):
                shard = shards[index]
                for batch in shard_batches(
                    pl.scan_parquet(shard),
                    pl.scan_parquet(root / "labels" / shard.name),
                    expansion,
                    subjects=subjects,
                    token_budget=token_budget,
                    seed=seed * 1000 + position,
                ):
                    if not offer(batch):
                        return
                if stop.is_set():
                    return
        except BaseException as error:  # noqa: BLE001 - re-raised on the consumer
            offer(error)
        else:
            offer(done)

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()

    try:
        while True:
            item = ready.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # release worker in case of error
        stop.set()
        worker.join(timeout=5.0)


def fold_subjects(split: Path, fold: str) -> pl.LazyFrame:
    """The subject ids belonging to one fold.

    Args:
        split (Path): Path to the subject split parquet file.
        fold (str): "training", "validation" or "testing".

    Returns:
        pl.LazyFrame: One column, `subject_id`.

    Raises:
        FileNotFoundError: If the split file is absent.
        ValueError: If the fold names no subject, which means it is misspelled --
            silently training on nothing is the failure this prevents.
    """
    if not split.is_file():
        raise FileNotFoundError(f"Expected the subject split at {split}.")

    subjects = (
        pl.scan_parquet(split).filter(pl.col("fold") == fold).select("subject_id")
    )
    if subjects.select(pl.len()).collect().item() == 0:
        folds = (
            pl.scan_parquet(split).select("fold").unique().collect()["fold"].to_list()
        )
        raise ValueError(f"Fold {fold!r} names no subject; the file holds {folds}.")
    return subjects


def bag_subjects(subjects: pl.LazyFrame, fraction: float, seed: int) -> pl.LazyFrame:
    """One ensemble member's share of the training subjects.

    Args:
        subjects (pl.LazyFrame): One `subject_id` column, as `fold_subjects` returns.
        fraction (float): The share to keep, in (0, 1].
        seed (int): Seeds the draw; a different seed is a different member.

    Returns:
        pl.LazyFrame: The kept subjects, one column.

    Raises:
        ValueError: If the fraction is outside (0, 1], or if it would keep nobody.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"A bag holds a fraction in (0, 1], got {fraction}.")

    available = subjects.collect()
    drawn = available.sample(
        fraction=fraction, with_replacement=False, shuffle=True, seed=seed
    ).sort("subject_id")
    if drawn.height == 0:
        raise ValueError(
            f"A fraction of {fraction} draws no subject from {available.height}."
        )
    return drawn.lazy()


def batch_to(
    batch: dict[str, torch.Tensor | int], device: torch.device
) -> dict[str, torch.Tensor | int]:
    """Moves every tensor in a batch onto a device, leaving `seq_len` an int.

    Args:
        batch (dict[str, torch.Tensor | int]): A batch as `collate` returns it.
        device (torch.device): Where to put it.

    Returns:
        dict[str, torch.Tensor | int]: The same batch, on the device.
    """
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }
