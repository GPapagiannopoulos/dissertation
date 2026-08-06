"""Testing suite for packing sequences into the encoder's tensors.

Everything here is an index arithmetic, and index arithmetic fails silently. A write
index flattened with the wrong stride still lands inside the buffer, on a position
that belongs to another subject; a label gathered from the wrong row still yields a
finite logit and a loss that falls. The shapes stay right through all of it, so the
assertions have to be about *which cell*, not about how many.
"""

from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from thesis.modelling.motor.batching import collate, padded_length
from thesis.modelling.motor.sequences import LABEL_SCHEMA, SEQUENCE_SCHEMA

BIRTH = datetime(2100, 1, 1)


@pytest.fixture
def make_sequences() -> Callable:
    """Returns a factory for sequence rows, from (sequence_id, length) pairs.

    Token indices are 1, 2, 3 ... across the whole batch so a pair can be traced back
    to the exact position that produced it.
    """

    def _make(*sequences: tuple[int, int]) -> pl.DataFrame:
        rows: list[dict] = []
        token = 1
        for sequence_id, length in sequences:
            for position in range(length):
                rows.append(
                    {
                        "subject_id": sequence_id,
                        "sequence_id": sequence_id,
                        "position": position,
                        "subject_position": position,
                        "index": token,
                        "time": BIRTH + timedelta(days=position),
                        "age": float(position),
                        "normed_age": float(position) / 10,
                    }
                )
                token += 1
        return pl.DataFrame(rows, schema=SEQUENCE_SCHEMA)

    return _make


@pytest.fixture
def make_labels() -> Callable:
    """Returns a factory for placed labels, from (sequence_id, position, value)."""

    def _make(*labels: tuple[int, int, bool]) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "subject_id": sequence_id,
                    "sequence_id": sequence_id,
                    "position": position,
                    "prediction_time": BIRTH + timedelta(days=position),
                    "boolean_value": value,
                }
                for sequence_id, position, value in labels
            ],
            schema=LABEL_SCHEMA,
        )

    return _make


@pytest.fixture
def identity_expansion() -> pl.LazyFrame:
    """An expansion where every token is worth exactly itself.

    The fan-out is tested where it lives; here it would only obscure which position a
    pair came from.
    """
    return pl.DataFrame(
        {"index": list(range(1, 64)), "token": list(range(1, 64))},
        schema={"index": pl.UInt32, "token": pl.UInt32},
    ).lazy()


@pytest.mark.parametrize(
    ("longest", "expected"),
    [(1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8), (9, 16), (1000, 1024)],
)
def test_padded_length_rounds_up_to_a_power_of_two(longest: int, expected: int) -> None:
    """Bucketing keeps the number of distinct shapes small.

    A length already a power of two must not double: that would pad every full
    sequence to twice its size.
    """
    assert padded_length(longest) == expected


def test_each_sequence_becomes_a_row(
    make_sequences: Callable, make_labels: Callable, identity_expansion: pl.LazyFrame
) -> None:
    """Rows are assigned in ascending sequence id, which is what the pairs assume."""
    batch = collate(
        make_sequences((7, 3), (2, 3), (5, 3)), make_labels(), identity_expansion
    )

    assert batch["seq_len"] == 4
    assert batch["ages"].shape == (3, 4)
    # sequence 2 sorts first, so its tokens land in row 0
    assert batch["indices"][0].tolist() == [4, 0]


def test_shorter_sequences_are_padded_on_the_right(
    make_sequences: Callable, make_labels: Callable, identity_expansion: pl.LazyFrame
) -> None:
    """A batch mixes lengths, and the padding must never precede a real position.

    Padding on the left would put filler inside the causal window of every real
    position after it.
    """
    batch = collate(make_sequences((0, 5), (1, 2)), make_labels(), identity_expansion)

    assert batch["valid_tokens"].tolist() == [
        [True] * 5 + [False] * 3,
        [True] * 2 + [False] * 6,
    ]


def test_ages_land_in_their_own_cells(
    make_sequences: Callable, make_labels: Callable, identity_expansion: pl.LazyFrame
) -> None:
    """Scattered by (row, position), not laid down in row order.

    With ragged lengths the two differ, and laying them down in order shifts every
    sequence after the first onto another subject's clock.
    """
    batch = collate(make_sequences((0, 3), (1, 2)), make_labels(), identity_expansion)

    assert batch["ages"].tolist() == [[0.0, 1.0, 2.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    assert batch["normed_ages"].flatten().tolist() == pytest.approx(
        [0.0, 0.1, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0]
    )


def test_pair_write_indices_are_flattened_over_the_batch(
    make_sequences: Callable, make_labels: Callable, identity_expansion: pl.LazyFrame
) -> None:
    """`row * seq_len + position` is the contract MotorEncoder unflattens.

    Flattened by the count of real positions instead, every row after the first would
    be shifted by however much padding preceded it -- still inside the buffer, still a
    real position, and belonging to someone else.
    """
    batch = collate(make_sequences((0, 3), (1, 2)), make_labels(), identity_expansion)

    # tokens 1..3 are sequence 0's positions 0..2, tokens 4..5 sequence 1's 0..1
    assert batch["indices"].tolist() == [[1, 0], [2, 1], [3, 2], [4, 4], [5, 5]]


def test_padded_positions_receive_no_pairs(
    make_sequences: Callable, make_labels: Callable, identity_expansion: pl.LazyFrame
) -> None:
    """They embed as zero, which the encoder then overwrites with ones before norming.

    A pair pointing at padding would give it a real concept, and the position would
    stop being inert.
    """
    batch = collate(make_sequences((0, 3)), make_labels(), identity_expansion)

    written = batch["indices"][:, 1].tolist()
    assert written == [0, 1, 2]


def test_labels_are_gathered_from_their_own_position(
    make_sequences: Callable, make_labels: Callable, identity_expansion: pl.LazyFrame
) -> None:
    """The index has to name the cell the landmark was placed on, in the flat buffer.

    Checked by decomposing it back, because an index that reads plausibly can still
    be pointing at the wrong row.
    """
    labels = make_labels((0, 2, True), (1, 1, False))

    batch = collate(make_sequences((0, 3), (1, 2)), labels, identity_expansion)

    seq_len = batch["seq_len"]
    assert [(int(i) // seq_len, int(i) % seq_len) for i in batch["label_indices"]] == [
        (0, 2),
        (1, 1),
    ]


def test_label_indices_come_back_sorted(
    make_sequences: Callable, make_labels: Callable, identity_expansion: pl.LazyFrame
) -> None:
    """The gather declares them sorted, and the labels stay aligned to them.

    Sorting the indices without carrying the values along would pair each label with
    another landmark's outcome.
    """
    labels = make_labels((1, 1, True), (0, 2, False), (0, 0, True))

    batch = collate(make_sequences((0, 3), (1, 2)), labels, identity_expansion)

    assert batch["label_indices"].tolist() == [0, 2, 5]
    assert batch["labels"].tolist() == [1.0, 0.0, 1.0]


def test_labels_are_floats_for_the_loss(
    make_sequences: Callable, make_labels: Callable, identity_expansion: pl.LazyFrame
) -> None:
    """`binary_cross_entropy_with_logits` will not take a bool target."""
    batch = collate(
        make_sequences((0, 3)), make_labels((0, 1, True)), identity_expansion
    )

    assert batch["labels"].dtype == torch.float32


def test_returns_the_encoder_arguments_by_name(
    make_sequences: Callable, make_labels: Callable, identity_expansion: pl.LazyFrame
) -> None:
    """The batch is splatted into forward, so the keys are a contract."""
    batch = collate(make_sequences((0, 3)), make_labels(), identity_expansion)

    assert set(batch) == {
        "indices",
        "seq_len",
        "ages",
        "normed_ages",
        "valid_tokens",
        "segment_ids",
        "label_indices",
        "labels",
    }
    assert batch["segment_ids"].shape == (1, 4)
    assert not batch["segment_ids"].any()


def test_dtypes_are_what_the_encoder_asserts(
    make_sequences: Callable, make_labels: Callable, identity_expansion: pl.LazyFrame
) -> None:
    """`rotary_tables` rejects anything but float32, and indexing needs int64."""
    batch = collate(
        make_sequences((0, 3)), make_labels((0, 1, True)), identity_expansion
    )

    assert batch["ages"].dtype == torch.float32
    assert batch["normed_ages"].dtype == torch.float32
    assert batch["valid_tokens"].dtype == torch.bool
    assert batch["indices"].dtype == torch.int64
    assert batch["label_indices"].dtype == torch.int64


def test_rejects_an_empty_batch(
    make_labels: Callable, identity_expansion: pl.LazyFrame
) -> None:
    """A batch of nothing has no length to pad to; the caller has a bug."""
    empty = pl.DataFrame(schema=SEQUENCE_SCHEMA)

    with pytest.raises(ValueError, match="at least one sequence"):
        collate(empty, make_labels(), identity_expansion)


def test_rejects_a_label_from_outside_the_batch(
    make_sequences: Callable, make_labels: Callable, identity_expansion: pl.LazyFrame
) -> None:
    """Its position would be flattened against a row that holds someone else.

    Dropping it instead would shrink the training set invisibly.
    """
    with pytest.raises(ValueError, match="does not hold"):
        collate(make_sequences((0, 3)), make_labels((9, 1, True)), identity_expansion)


def test_a_pair_is_emitted_for_every_ancestor(
    make_sequences: Callable, make_labels: Callable
) -> None:
    """The fan-out still applies; only where the position comes from changes."""
    expansion = pl.DataFrame(
        {"index": [1, 1, 1, 2], "token": [1, 30, 31, 2]},
        schema={"index": pl.UInt32, "token": pl.UInt32},
    ).lazy()

    batch = collate(make_sequences((0, 2)), make_labels(), expansion)

    assert batch["indices"].tolist() == [[1, 0], [30, 0], [31, 0], [2, 1]]


def test_the_grid_matches_a_hand_scattered_reference(
    make_sequences: Callable, make_labels: Callable, identity_expansion: pl.LazyFrame
) -> None:
    """One end-to-end check that the whole grid is right, not only the cells asserted.

    Guards against a transposed scatter, which for a square batch would otherwise
    survive every shape assertion above.
    """
    batch = collate(make_sequences((0, 4), (1, 4)), make_labels(), identity_expansion)

    expected = np.zeros((2, 4), dtype=np.float32)
    expected[0] = [0.0, 1.0, 2.0, 3.0]
    expected[1] = [0.0, 1.0, 2.0, 3.0]
    assert np.array_equal(batch["ages"].numpy(), expected)
    assert batch["indices"][:, 0].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert batch["indices"][:, 1].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
