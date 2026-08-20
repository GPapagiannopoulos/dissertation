"""Fixtures for the fine-tuning stack: a toy classifier and one collated batch."""

from collections.abc import Callable

import pytest
import torch

from thesis.modelling.backbone.model import MotorEncoder
from thesis.modelling.finetune.head import MotorClassifier

SMALL_ENCODER: dict[str, int] = {
    "vocab_size": 32,
    "hidden_size": 8,
    "intermediate_size": 16,
    "n_heads": 2,
    "n_layers": 2,
    "attention_width": 3,
}


SMALL_SEQ_LEN: int = 4


@pytest.fixture
def make_classifier() -> Callable:
    """Returns a factory for a toy `MotorClassifier`; one seed gives one backbone."""

    def _make(seed: int = 0) -> MotorClassifier:
        torch.manual_seed(seed)
        return MotorClassifier(MotorEncoder(**SMALL_ENCODER), positive_rate=0.05)

    return _make


@pytest.fixture
def classifier_batch() -> dict[str, torch.Tensor | int]:
    """One batch of `MotorClassifier.forward` arguments at the toy widths.

    Shaped as `collate` emits one: a single sequence with a label on its last
    position.
    """
    tokens = torch.arange(1, SMALL_SEQ_LEN + 1)
    ages = torch.arange(SMALL_SEQ_LEN, dtype=torch.float32).unsqueeze(0) * 10.0
    return {
        "indices": torch.stack((tokens, torch.arange(SMALL_SEQ_LEN))).T.long(),
        "seq_len": SMALL_SEQ_LEN,
        "ages": ages,
        "normed_ages": (ages - ages.mean()) / ages.std(),
        "valid_tokens": torch.ones(1, SMALL_SEQ_LEN, dtype=torch.bool),
        "segment_ids": torch.zeros(1, SMALL_SEQ_LEN, dtype=torch.long),
        "label_indices": torch.tensor([SMALL_SEQ_LEN - 1], dtype=torch.int64),
        "label_clocks": torch.tensor([[0.5]], dtype=torch.float32),
    }
