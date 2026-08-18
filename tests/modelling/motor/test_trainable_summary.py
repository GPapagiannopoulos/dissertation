"""Testing suite for counting what a run will actually move."""

from collections.abc import Callable

import pytest
import torch

from thesis.modelling.motor.lora import apply_lora, lora_config, trainable_summary

# 2 blocks x (q, v), each an (8, 10) A and an (8, 8) B
TOY_ADAPTER_PARAMS: int = 576
# Linear(2 * 8 + 1, 1): the position vector, its pooled window, and the clock
TOY_HEAD_PARAMS: int = 18


def test_it_counts_the_adapters_and_the_head(make_classifier: Callable) -> None:
    """Composed of two things, so both are named rather than totalled."""
    model = apply_lora(make_classifier(), lora_config())

    summary = trainable_summary(model)

    assert summary["trainable"] == TOY_ADAPTER_PARAMS + TOY_HEAD_PARAMS
    assert summary["adapters"] == 4
    assert summary["fraction"] == summary["trainable"] / summary["total"]


def test_an_adapter_is_counted_once_not_per_tensor(make_classifier: Callable) -> None:
    """`lora_A` and `lora_B` are one adapter on one module, not two."""
    model = apply_lora(make_classifier(), lora_config(target_modules=("q_proj",)))

    assert trainable_summary(model)["adapters"] == 2


def test_an_unadapted_model_reports_itself_as_fully_trainable(
    make_classifier: Callable,
) -> None:
    """Under the LoRA driver this reading is the signal that the wrap was skipped."""
    summary = trainable_summary(make_classifier())

    assert summary["adapters"] == 0
    assert summary["fraction"] == 1.0
    assert summary["trainable"] == summary["total"]


def test_a_model_with_no_parameters_is_refused() -> None:
    """The fraction would be 0/0, which reads as a NaN nobody looks at twice."""
    with pytest.raises(ValueError, match="no parameters"):
        trainable_summary(torch.nn.Module())
