"""Testing suite for loading an adapter checkpoint back into a model."""

from collections.abc import Callable

import pytest
import torch

from thesis.modelling.finetune.lora import (
    adapter_state,
    apply_lora,
    load_adapter,
    lora_config,
)


def trained(model: torch.nn.Module, seed: int = 7) -> torch.nn.Module:
    """Moves every trainable tensor off its initial value.

    A fresh model has `lora_B` and the head weight at zero, so an adapter saved
    before training is indistinguishable from one that never loaded.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.copy_(torch.rand(parameter.shape, generator=generator))
    return model


def test_a_trained_adapter_transfers(
    make_classifier: Callable, classifier_batch: dict
) -> None:
    """Both start from the same seed, so only the adapter and head can differ."""
    source = trained(apply_lora(make_classifier(seed=0), lora_config())).eval()
    target = apply_lora(make_classifier(seed=0), lora_config()).eval()

    with torch.no_grad():
        assert not torch.equal(source(**classifier_batch), target(**classifier_batch))
        load_adapter(target, adapter_state(source))
        assert torch.equal(source(**classifier_batch), target(**classifier_batch))


def test_the_frozen_backbone_is_untouched(make_classifier: Callable) -> None:
    """A non-strict load must not be a route for the file to rewrite the backbone."""
    source = trained(apply_lora(make_classifier(seed=0), lora_config()))
    target = apply_lora(make_classifier(seed=1), lora_config())
    name = "encoder.base_model.model.blocks.0.input_proj.q_proj.base_layer.weight"
    before = target.state_dict()[name].clone()

    load_adapter(target, adapter_state(source))

    assert torch.equal(target.state_dict()[name], before)


def test_a_key_matching_nothing_is_refused(make_classifier: Callable) -> None:
    """Silently dropping it would train one configuration and score another."""
    model = apply_lora(make_classifier(), lora_config())
    state = {**adapter_state(model), "encoder.blocks.9.nonsense": torch.zeros(1)}

    with pytest.raises(ValueError, match="match nothing in the model"):
        load_adapter(model, state)


def test_an_unadapted_model_is_refused(make_classifier: Callable) -> None:
    """Every adapter key is unexpected, so the load would be a silent no-op."""
    state = adapter_state(apply_lora(make_classifier(), lora_config()))

    with pytest.raises(ValueError, match="match nothing in the model"):
        load_adapter(make_classifier(), state)


def test_a_wider_rank_is_refused(make_classifier: Callable) -> None:
    """The rank is per-member, so a mismatched pairing is a live possibility."""
    source = apply_lora(make_classifier(), lora_config(r=4))
    target = apply_lora(make_classifier(), lora_config(r=8))

    with pytest.raises(RuntimeError, match="size mismatch"):
        load_adapter(target, adapter_state(source))
