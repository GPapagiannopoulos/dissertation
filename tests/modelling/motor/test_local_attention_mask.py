"""Tests the attention mask against the pattern the released model actually applies.

The oracle's own attention_width (496) is wider than any sequence in the dump and its
packed length covers the whole buffer, so the block activations exercise neither the
window nor the segmentation. The probe uses two segments of 32 and a window of 16, so
both conditions bind and neither can hide the other.
"""

import numpy as np
import pytest
import torch
from numpy.lib.npyio import NpzFile

from thesis.modelling.motor.layers import local_attention_mask

SEQ_LEN = 64
WIDTH = 16
SEGMENT_LENGTH = 32


@pytest.fixture
def segments() -> torch.Tensor:
    """Two packed sequences of 32, the layout the attention probe was dumped under."""
    return torch.arange(SEQ_LEN) // SEGMENT_LENGTH


def test_is_causal(segments: torch.Tensor) -> None:
    """No query may see a key that comes after it."""
    mask = local_attention_mask(segments, WIDTH)

    assert not mask.triu(diagonal=1).any()


def test_every_query_sees_itself(segments: torch.Tensor) -> None:
    """Guards the softmax: an all-masked row would return nan, not zero."""
    assert local_attention_mask(segments, WIDTH).diagonal().all()


def test_window_is_inclusive(segments: torch.Tensor) -> None:
    """A query reaches width positions back, so it sees width + 1 keys in all.

    Off by one here is the difference between MOTOR's 497-key window and a 496-key one.
    """
    mask = local_attention_mask(segments, WIDTH)

    assert mask[SEQ_LEN - 1].sum() == WIDTH + 1
    assert mask[SEQ_LEN - 1, SEQ_LEN - 1 - WIDTH]
    assert not mask[SEQ_LEN - 1, SEQ_LEN - 2 - WIDTH]


def test_segments_do_not_attend_across(segments: torch.Tensor) -> None:
    """The first query of a packed sequence sees nothing before it.

    This is what stops one subject reading another's events out of a shared buffer.
    """
    mask = local_attention_mask(segments, WIDTH)

    assert mask[SEGMENT_LENGTH].sum() == 1
    assert not mask[:SEGMENT_LENGTH, SEGMENT_LENGTH:].any()
    assert not mask[SEGMENT_LENGTH:, :SEGMENT_LENGTH].any()


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        pytest.param(0, 1, id="self_only"),
        pytest.param(1, 2, id="one_back"),
        pytest.param(SEQ_LEN, SEGMENT_LENGTH, id="wider_than_the_segment"),
    ],
)
def test_width_counts_keys(segments: torch.Tensor, width: int, expected: int) -> None:
    """The last query of a segment sees min(width + 1, segment_length) keys."""
    mask = local_attention_mask(segments, width)

    assert mask[SEQ_LEN - 1].sum() == expected


def test_a_single_sequence_is_plain_causal_windowing() -> None:
    """All-zero segment ids impose no boundary at all."""
    mask = local_attention_mask(torch.zeros(SEQ_LEN, dtype=torch.long), SEQ_LEN)

    assert torch.equal(mask, torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool).tril())


@pytest.mark.parametrize(
    ("segment_ids", "width", "match"),
    [
        pytest.param(torch.zeros(2, 2), 4, "one dimensional", id="2d_segments"),
        pytest.param(torch.zeros(4), -1, "negative", id="negative_width"),
    ],
)
def test_rejects_malformed_arguments(
    segment_ids: torch.Tensor, width: int, match: str
) -> None:
    """A negative width would silently mask every key including the diagonal."""
    with pytest.raises(ValueError, match=match):
        local_attention_mask(segment_ids, width)


def test_matches_jax_oracle(jax_oracle: NpzFile) -> None:
    """Reproduces femr's mask exactly, derived from its own length bitmask.

    The probe was dumped with q = k = 0 and v = I, which makes every allowed key share
    the weight equally, so a nonzero entry of the output IS an allowed (i, j) pair.
    """
    bitmask = int(jax_oracle["attention_probe_length_mask"])
    segment_ids = torch.arange(SEQ_LEN) & bitmask

    mask = local_attention_mask(segment_ids, int(jax_oracle["attention_probe_width"]))

    expected = torch.from_numpy(jax_oracle["attention_probe_pattern"][0] > 0)
    assert torch.equal(mask, expected)


def test_probe_arrays_are_what_the_oracle_dumped(jax_oracle: NpzFile) -> None:
    """Guards the probe's parameters, which the constants above hardcode.

    If the dump is ever regenerated with a width that no longer binds, these fail
    rather than the mask tests quietly weakening.
    """
    assert int(jax_oracle["attention_probe_width"]) == WIDTH
    assert int(jax_oracle["attention_probe_length_mask"]) == np.uint32(
        ~np.uint32(SEGMENT_LENGTH - 1)
    )
    assert jax_oracle["attention_probe_pattern"].shape == (1, SEQ_LEN, SEQ_LEN)
