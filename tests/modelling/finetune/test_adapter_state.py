"""Testing suite for the subset of a model that a LoRA run checkpoints."""

from collections.abc import Callable

import pytest
import torch

from thesis.modelling.finetune.head import MotorClassifier
from thesis.modelling.finetune.lora import adapter_state, apply_lora, lora_config


class Compiled(torch.nn.Module):
    """Stands in for what `torch.compile` returns: the module under `_orig_mod`."""

    def __init__(self, inner: torch.nn.Module) -> None:
        """Holds the wrapped module under the attribute name compile uses."""
        super().__init__()
        self._orig_mod = inner


def test_it_holds_the_trainable_tensors_and_nothing_else(
    make_classifier: Callable,
) -> None:
    """The adapters and the head, which is exactly what moved during training."""
    model = apply_lora(make_classifier(), lora_config())

    state = adapter_state(model)

    assert set(state) == {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(".lora_" in name or name.startswith("head.") for name in state)


def test_no_backbone_tensor_rides_along(make_classifier: Callable) -> None:
    """Stated as disjointness from the frozen set, not as a size ratio.

    At the toy widths r=8 is nearly full rank on a 10-wide projection, so the
    fraction only means something at the released widths.
    """
    model = apply_lora(make_classifier(), lora_config())
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}

    state = adapter_state(model)

    assert frozen
    assert not set(state) & frozen
    assert not any(".base_layer." in name for name in state)


def test_the_values_are_the_model_s_own(make_classifier: Callable) -> None:
    """Selecting the right keys is worth nothing if the tensors are stale copies."""
    model = apply_lora(make_classifier(), lora_config())
    with torch.no_grad():
        model.head.bias.fill_(1.25)

    assert float(adapter_state(model)["head.bias"]) == pytest.approx(1.25)


def test_compiled_names_are_normalised(make_classifier: Callable) -> None:
    """Scoring is always eager, so the compiled naming can never be the one on disk."""
    model = apply_lora(make_classifier(), lora_config())
    expected = set(adapter_state(model))
    model.encoder = Compiled(model.encoder)

    state = adapter_state(model)

    assert not any("_orig_mod" in name for name in state)
    assert set(state) == expected


def test_a_model_with_nothing_trainable_is_refused(make_classifier: Callable) -> None:
    """Otherwise the run writes an empty file at every checkpoint and says nothing."""
    model: MotorClassifier = make_classifier()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(ValueError, match="no adapter to save"):
        adapter_state(model)
