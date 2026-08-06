"""Module turning positioned sequences into the tensors the encoder takes.

This is the boundary between the Polars side of the pipeline and the torch side:
frames in, tensors out.
"""

import numpy as np
import polars as pl
import torch

# Sequences are padded up to a power of two rather than to the longest in the batch,
# so the encoder sees a handful of distinct lengths instead of one per batch.
PAD_VALUE: float = 0.0


def padded_length(longest: int) -> int:
    """Rounds a sequence length up to the next power of two.

    Args:
        longest (int): The most positions any sequence in the batch holds.

    Returns:
        int: The padded length, which is `longest` itself when it is already a power
            of two.

    Raises:
        ValueError: If the length is not positive.
    """
    if longest < 1:
        raise ValueError(f"A sequence holds at least one position, got {longest}.")
    return 1 << (longest - 1).bit_length()


def collate(
    sequences: pl.DataFrame, labels: pl.DataFrame, expansion: pl.LazyFrame
) -> dict[str, torch.Tensor | int]:
    """Packs one batch of sequences into the encoder's arguments.

    Args:
        sequences (pl.DataFrame): Rows from `build_sequences` for the sequences in
            this batch, and no others.
        labels (pl.DataFrame): Rows from `place_labels` for those same sequences.
        expansion (pl.LazyFrame): The table `build_ancestor_expansion` returns.

    Returns:
        dict[str, torch.Tensor | int]: `indices`, `seq_len`, `ages`, `normed_ages`,
            `valid_tokens` and `segment_ids` are `MotorEncoder.forward`'s arguments
            by name; `label_indices` and `labels` carry the supervision.

    Raises:
        ValueError: If the batch holds no sequence, or if a label names a sequence
            the batch does not contain.
    """
    if sequences.height == 0:
        raise ValueError("A batch holds at least one sequence, got an empty frame.")

    ordered = sequences.sort("sequence_id", "position")
    batch_rows = ordered["sequence_id"].unique(maintain_order=False).sort()
    row_of = {sequence: row for row, sequence in enumerate(batch_rows)}

    batch_size = len(row_of)
    seq_len = padded_length(int(ordered["position"].max()) + 1)

    if unknown := set(labels["sequence_id"].unique()) - set(row_of):
        raise ValueError(
            f"{len(unknown)} label(s) name a sequence this batch does not hold: "
            f"{sorted(unknown)[:5]}. Their positions would index another sequence."
        )

    ordered = ordered.with_columns(
        row=pl.col("sequence_id").replace_strict(row_of, return_dtype=pl.UInt32)
    )
    rows = ordered["row"].to_numpy()
    positions = ordered["position"].to_numpy()

    def grid(column: str, dtype: type) -> torch.Tensor:
        """Scatters one per-position column into the padded grid."""
        filled = np.full((batch_size, seq_len), PAD_VALUE, dtype=dtype)
        filled[rows, positions] = ordered[column].to_numpy()
        return torch.from_numpy(filled)

    valid = np.zeros((batch_size, seq_len), dtype=bool)
    valid[rows, positions] = True

    pairs = (
        ordered.lazy()
        .with_columns(flat=pl.col("row") * seq_len + pl.col("position"))
        .join(expansion, on="index", how="inner")
        .select("token", "flat")
        .sort("flat", "token")
        .collect()
    )

    placed = labels.with_columns(
        flat=pl.col("sequence_id").replace_strict(row_of, return_dtype=pl.UInt32)
        * seq_len
        + pl.col("position")
    ).sort("flat")

    return {
        "indices": torch.from_numpy(pairs.to_numpy().astype(np.int64)),
        "seq_len": seq_len,
        "ages": grid("age", np.float32),
        "normed_ages": grid("normed_age", np.float32),
        "valid_tokens": torch.from_numpy(valid),
        "segment_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "label_indices": torch.from_numpy(placed["flat"].to_numpy().astype(np.int64)),
        "labels": torch.from_numpy(
            placed["boolean_value"].to_numpy().astype(np.float32)
        ),
    }
