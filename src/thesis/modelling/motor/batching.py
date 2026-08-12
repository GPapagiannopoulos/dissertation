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

LABEL_METADATA: tuple[str, ...] = ("labels", "label_subjects", "label_times")
"""Batch keys that are supervision or provenance, not `MotorClassifier.forward` args.

Every other key in a collated batch is splatted into the model, so anything added here
has to be taken out first -- `forward` names its arguments and an unexpected one is a
`TypeError` at the top of the step. Both loops in `training.py` pop this tuple.

`label_subjects` and `label_times` exist because a metric needs no provenance but a
confidence interval does: the bootstrap resamples SUBJECTS, and pairing this model's
predictions against another's needs a key both sides share.

`label_clocks` is deliberately NOT here: it is a model input, so it stays in the dict
and splats into `forward` alongside the encoder's own arguments.
"""

CLOCK_FEATURES: tuple[str, ...] = ("log1p_hours_since_event",)
"""The per-label scalars handed to the head alongside the pooled features.

MOTOR's head reads the hidden state at the last charted event at or before the
landmark, and nothing in the encoder's input says how long ago that was. Measured on
the validation fold, the gap runs to a **median 4.60 hours**, p90 21.9 and p99 94.
The XGBoost baseline gets this as `hours_since`, per code, and an ablation that
removed it cost the tree **0.0310 AUPRC [0.0261, 0.0359]** -- 79% of its margin over
the fine-tuned encoder. This is that signal, in the weakest form that does not
require touching the pretrained backbone.

The value is `log1p(hours)` rather than hours. The raw gap spans 0 to 792, which
alongside activations of order 1 would dominate the head's first gradient purely by
scale; the log compresses it to roughly 0-6.7 while keeping the ordering intact and
keeping zero at zero.

Adding `hours_since_admission` would need the labeller's `admittime` carried into
stage 5.2's label shards, which they do not currently hold -- a stage 5.2 change, not
a head change.
"""


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
            by name; `label_indices` and `label_clocks` are `MotorClassifier`'s two
            extra arguments; `labels` carries the supervision; and `label_subjects`
            and `label_times` carry each label's provenance, in the same order, so a
            caller can group predictions by patient or pair them against another
            model's. See `LABEL_METADATA` and `CLOCK_FEATURES`.

    Raises:
        ValueError: If the batch holds no sequence, if a label names a sequence the
            batch does not contain, or if a label names a position its sequence does
            not hold.
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

    # the time of the position each label was pinned to, which is what the encoder's
    # last visible event actually is; the difference against `prediction_time` is the
    # staleness the head is otherwise blind to
    pinned = ordered.select("sequence_id", "position", "time").unique(
        subset=["sequence_id", "position"], maintain_order=False
    )
    placed = (
        labels.join(pinned, on=["sequence_id", "position"], how="left")
        .with_columns(
            flat=pl.col("sequence_id").replace_strict(row_of, return_dtype=pl.UInt32)
            * seq_len
            + pl.col("position")
        )
        .sort("flat")
    )

    if placed["time"].null_count():
        raise ValueError(
            f"{placed['time'].null_count()} label(s) name a position their sequence "
            f"does not hold, so no event time could be found for them."
        )

    # clipped at zero because the landmark is placed by a BACKWARD asof join and can
    # never precede its own event; a negative here would mean that join had flipped
    hours = (
        (pl.col("prediction_time") - pl.col("time")).dt.total_seconds() / 3600.0
    ).clip(lower_bound=0.0)
    clocks = placed.select(hours.log1p().alias(CLOCK_FEATURES[0]))

    return {
        "indices": torch.from_numpy(pairs.to_numpy().astype(np.int64)),
        "seq_len": seq_len,
        "ages": grid("age", np.float32),
        "normed_ages": grid("normed_age", np.float32),
        "valid_tokens": torch.from_numpy(valid),
        "segment_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "label_indices": torch.from_numpy(placed["flat"].to_numpy().astype(np.int64)),
        "label_clocks": torch.from_numpy(
            clocks.to_numpy()
            .astype(np.float32)
            .reshape(placed.height, len(CLOCK_FEATURES))
        ),
        "labels": torch.from_numpy(
            placed["boolean_value"].to_numpy().astype(np.float32)
        ),
        "label_subjects": torch.from_numpy(
            placed["subject_id"].to_numpy().astype(np.int64)
        ),
        # microseconds since the epoch, because a batch is tensors and a tensor holds
        # no datetime. The unit is fixed rather than inherited so that two frames
        # stored at different precisions still produce the same integer for one
        # instant -- this column is a JOIN KEY across models, not a duration.
        "label_times": torch.from_numpy(
            placed["prediction_time"].dt.epoch("us").to_numpy().astype(np.int64)
        ),
    }
