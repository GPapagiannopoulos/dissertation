"""Module assembling tokenised events into the sequences the encoder reads.

The tokeniser gives each event a row of the embedding table. This module gives it a
place in a sequence: an age, a position, and the sequence it belongs to. Everything
here is a pure LazyFrame transform, so the caller decides whether to materialise.
"""

from pathlib import Path

import polars as pl

from thesis.modelling.motor.tokenizer import assign_leaf_tokens, load_token_table

# femr's batch creator stores an integer age in minutes and divides by this to get the
# days the model actually sees. Reproduced exactly rather than computed from the
# timestamps directly, so the truncation happens in the same place.
MINUTES_PER_DAY = 1440

# The MEDS code marking a subject's time origin. It carries no OMOP concept and so
# survives normalisation but never tokenisation.
BIRTH_CODE = "MEDS_BIRTH"

SEQUENCE_SCHEMA: dict[str, pl.DataType] = {
    "subject_id": pl.Int64,
    "sequence_id": pl.UInt32,
    "position": pl.UInt32,
    "subject_position": pl.UInt32,
    "index": pl.UInt32,
    "time": pl.Datetime("us"),
    "age": pl.Float32,
    "normed_age": pl.Float32,
}

LABEL_SCHEMA: dict[str, pl.DataType] = {
    "subject_id": pl.Int64,
    "sequence_id": pl.UInt32,
    "position": pl.UInt32,
    "prediction_time": pl.Datetime("us"),
    "boolean_value": pl.Boolean,
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
            `time` is carried through so a landmark can be placed against it and so
            an age in the written artifact can be audited against its timestamp.

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
    # ended
    n_chunks = (
        (pl.col("n_events").cast(pl.Int64) - length + stride - 1) // stride
    ).clip(0) + 1
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


def place_labels(sequences: pl.LazyFrame, labels: pl.LazyFrame) -> pl.LazyFrame:
    """Attaches each landmark to the position whose features it must be read from.

    An event in a chunk overlap sits in two sequences at two positions. The landmark
    takes the larger position, which is the chunk holding more of the subject's
    history before it.


    Notes:
        A landmark earlier than the subject's first tokenised event is dropped. It
            has no position to read, and there is nothing to predict from.
        Landmarks that pin to the same position are both kept. A 12-hour
            grid outruns the record whenever no event is charted in between, so two
            predictions sharing one history may disagree on the outcome.

    Args:
        sequences (pl.LazyFrame): The frame `build_sequences` returns.
        labels (pl.LazyFrame): One row per landmark, carrying `subject_id`,
            `prediction_time` and `boolean_value`.

    Returns:
        pl.LazyFrame: One row per placed landmark, sorted by sequence and position.
    """
    timeline = (
        sequences.select("subject_id", "subject_position", "time")
        .unique(subset=["subject_id", "subject_position"])
        .sort("subject_id", "time")
    )

    placed = (
        labels.select("subject_id", "prediction_time", "boolean_value")
        .with_row_index("landmark")
        .sort("subject_id", "prediction_time")
        .join_asof(
            timeline,
            by="subject_id",
            left_on="prediction_time",
            right_on="time",
            strategy="backward",
        )
        # a landmark before the subject's first event matches no position
        .join(
            sequences.select(
                "subject_id", "subject_position", "sequence_id", "position"
            ),
            on=["subject_id", "subject_position"],
            how="inner",
        )
    )

    return (
        placed.group_by("landmark")
        .agg(pl.all().sort_by("position").last())
        .select(*LABEL_SCHEMA)
        .sort("sequence_id", "position", "prediction_time")
    )


def subject_birth_times(events: pl.LazyFrame) -> pl.LazyFrame:
    """Lifts each subject's time origin out of the raw event stream.

    This has to run before tokenisation. `MEDS_BIRTH` carries no OMOP concept, so
    it is in no MOTOR vocabulary and `assign_leaf_tokens` drops it.

    A subject holds exactly one birth in a well-formed MEDS dataset. The minimum is
    taken anyway, so a duplicated origin gives one deterministic answer rather than
    multiplying every one of that subject's events through the join.

    Args:
        events (pl.LazyFrame): Raw MEDS events, before tokenisation.

    Returns:
        pl.LazyFrame: One row per subject, `subject_id` and `birth`.
    """
    return (
        events.filter(pl.col("code") == BIRTH_CODE)
        .group_by("subject_id")
        .agg(birth=pl.col("time").min())
    )


def restrict_to_labelled(events: pl.LazyFrame, labels: pl.LazyFrame) -> pl.LazyFrame:
    """Keeps only the events a prediction could ever be made from.

    Events after a subject's last landmark are dropped because attention is causal.
    No position can influence a prediction made before it, so they would
    be encoded and never read.

    Subjects carrying no landmark are also dropped. An unmatched subject has no
    landmark to compare against, and the comparison decides it either way.

    Args:
        events (pl.LazyFrame): Raw MEDS events.
        labels (pl.LazyFrame): The landmark labels, carrying `subject_id` and
            `prediction_time`.

    Returns:
        pl.LazyFrame: The events worth tokenising.
    """
    horizon = labels.group_by("subject_id").agg(
        last_landmark=pl.col("prediction_time").max()
    )
    return (
        events.join(horizon, on="subject_id", how="inner")
        .filter(pl.col("time") <= pl.col("last_landmark"))
        .drop("last_landmark")
    )


def run_build_sequences(
    events: Path,
    labels: Path,
    dest: Path,
    *,
    dictionary: Path,
    vocab_size: int,
    length: int,
    stride: int,
) -> Path:
    """Materialises the tokenised sequences and their placed labels, shard by shard.

    Shards are processed one at a time, as in stage 2.6: peak memory stays at one
    shard, and a crash at shard 137 leaves 137 readable outputs.

    `sequence_id` is dense within a call to `build_sequences`, so every shard would
    otherwise restart at zero and two subjects in different shards would share an id.
    A running offset makes it unique across the dataset, which is why the shards
    cannot be processed in parallel without reworking this.

    Args:
        events (Path): Stage 2.6's shard folder, i.e. <normalized>/data.
        labels (Path): The landmark labels parquet.
        dest (Path): The folder to create, which must not already exist.
        dictionary (Path): MOTOR's msgpack dictionary.
        vocab_size (int): The token count the checkpoint's config declares.
        length (int): The longest sequence to emit.
        stride (int): How far apart consecutive chunks start.

    Returns:
        Path: The folder written, holding `sequences/` and `labels/`.

    Raises:
        FileExistsError: If dest already exists.
        FileNotFoundError: If events holds no parquet, or an input file is missing.
    """
    if dest.exists():
        raise FileExistsError(
            f"Destination {dest} already exists; refusing to overwrite. "
            f"Remove it or choose a new path."
        )
    shards = sorted(events.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"Found no parquet shards in {events}. Point events at stage 2.6's "
            f"data/ folder."
        )
    for path in (labels, dictionary):
        if not path.is_file():
            raise FileNotFoundError(f"Expected a file at {path}.")

    table = load_token_table(dictionary, vocab_size=vocab_size)
    print(f"vocabulary holds {len(table.code_tokens)} code tokens")

    label_frame = pl.scan_parquet(labels)
    sequence_dir = dest / "sequences"
    label_dir = dest / "labels"
    sequence_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    offset = 0
    for position, shard in enumerate(shards, start=1):
        wanted = restrict_to_labelled(pl.scan_parquet(shard), label_frame)
        built = build_sequences(
            assign_leaf_tokens(wanted, table),
            subject_birth_times(wanted),
            length=length,
            stride=stride,
            age_mean=table.age_mean,
            age_std=table.age_std,
        ).collect()

        if built.height:
            built = built.with_columns(sequence_id=pl.col("sequence_id") + offset)
            offset = int(built["sequence_id"].max()) + 1
            placed = place_labels(built.lazy(), label_frame).collect()
            built.write_parquet(sequence_dir / shard.name)
            placed.write_parquet(label_dir / shard.name)
        else:
            placed = pl.DataFrame(schema=LABEL_SCHEMA)

        print(
            f"  [{position}/{len(shards)}] {shard.name}: "
            f"{built.height} positions, {placed.height} labels",
            flush=True,
        )

    print(f"{offset} sequences written")
    return dest
