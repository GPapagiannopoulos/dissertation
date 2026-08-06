"""Module assembling tokenised events into the sequences the encoder reads.

The tokeniser gives each event a row of the embedding table. This module gives it a
place in a sequence: an age, a position, and the sequence it belongs to. Everything
here is a pure LazyFrame transform, so the caller decides whether to materialise.
"""

import polars as pl

# femr's batch creator stores an integer age in minutes and divides by this to get the
# days the model actually sees. Reproduced exactly rather than computed from the
# timestamps directly, so the truncation happens in the same place.
MINUTES_PER_DAY = 1440

SEQUENCE_SCHEMA: dict[str, pl.DataType] = {
    "subject_id": pl.Int64,
    "sequence_id": pl.UInt32,
    "position": pl.UInt32,
    "subject_position": pl.UInt32,
    "index": pl.UInt32,
    "age": pl.Float32,
    "normed_age": pl.Float32,
}


def build_sequences(
    events: pl.LazyFrame,
    births: pl.LazyFrame,
    *,
    length: int,
    stride: int,
    age_mean: float,
    age_std: float,
) -> pl.LazyFrame:
    """Turns tokenised events into positioned, aged sequences.

    A subject's timeline is one sequence when it fits in `length`, and is cut into
    overlapping chunks when it does not. Attention reaches `attention_width` positions
    back, so a chunk that begins `stride` positions into the previous one gives every
    position past its first `stride` a full window. Events in the overlap therefore
    appear in two sequences, which is why this returns more rows than it is given.

    Subjects with no birth are dropped by the inner join. An event with no age has no
    place on the rotary clock, and a null age would reach the encoder as a NaN table.

    Args:
        events (pl.LazyFrame): Tokenised events carrying `subject_id`, `time` and the
            `index` column `assign_leaf_tokens` emits. Rows are expected in the
            shards' own order, which breaks ties inside a timestamp.
        births (pl.LazyFrame): One row per subject, `subject_id` and `birth`.
            `MEDS_BIRTH` carries no OMOP concept and so does not survive
            tokenisation; the origin has to arrive separately.
        length (int): The longest sequence to emit, in positions.
        stride (int): How far apart consecutive chunks start. Equal to `length` for
            no overlap; `length // 2` gives every chunk past the first a full
            attention window.
        age_mean (float): The dictionary's mean age in days.
        age_std (float): The dictionary's age standard deviation, in days.

    Returns:
        pl.LazyFrame: One row per (sequence, position), sorted by both. `position` is
            the place in the sequence the encoder sees; `subject_position` is the
            place in the subject's whole timeline, which is what labels join on.

    Raises:
        ValueError: If the length or stride are not positive, if the stride exceeds
            the length, or if the age standard deviation is not positive.
    """
    if length < 1:
        raise ValueError(f"The length must be at least 1, got {length}.")
    if not 1 <= stride <= length:
        raise ValueError(
            f"The stride must be between 1 and the length {length}, got {stride}."
        )
    if age_std <= 0:
        raise ValueError(f"The age standard deviation must be positive, got {age_std}.")

    age = (pl.col("time") - pl.col("birth")).dt.total_minutes() / MINUTES_PER_DAY

    positioned = (
        events.join(births, on="subject_id", how="inner")
        # a Polars sort is not stable, so the arrival order is made an explicit key:
        # events sharing a timestamp would otherwise be positioned differently on
        # every run, and the artifact would not be reproducible
        .with_row_index("arrival")
        .sort("subject_id", "time", "arrival")
        .with_columns(
            subject_position=pl.int_range(pl.len(), dtype=pl.UInt32).over("subject_id"),
            n_events=pl.len().over("subject_id"),
            age=age,
        )
    )

    # the chunks a subject needs, and then the ones each position falls inside: a
    # position belongs to every chunk that starts at or before it and has not yet
    # ended, which is at most two while the stride is at least half the length
    n_chunks = ((pl.col("n_events") - length + stride - 1) // stride).clip(0) + 1
    first = ((pl.col("subject_position").cast(pl.Int64) - length) // stride + 1).clip(0)
    last = pl.min_horizontal(pl.col("subject_position") // stride, n_chunks - 1)

    chunked = (
        positioned.with_columns(chunk=pl.int_ranges(first, last + 1))
        .explode("chunk")
        .with_columns(position=pl.col("subject_position") - pl.col("chunk") * stride)
        .sort("subject_id", "chunk", "position")
    )

    opens_a_sequence = (pl.col("subject_id") != pl.col("subject_id").shift(1)) | (
        pl.col("chunk") != pl.col("chunk").shift(1)
    )

    return (
        chunked.with_columns(
            sequence_id=opens_a_sequence.fill_null(True).cum_sum().cast(pl.UInt32) - 1
        )
        .with_columns(
            position=pl.col("position").cast(pl.UInt32),
            age=pl.col("age").cast(pl.Float32),
            normed_age=((pl.col("age") - age_mean) / age_std).cast(pl.Float32),
        )
        .select(*SEQUENCE_SCHEMA)
        .sort("sequence_id", "position")
    )
