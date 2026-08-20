"""Testing suite for injecting adapters into a built classifier."""

from collections.abc import Callable

import pytest
import torch
from peft import PeftModel

from thesis.modelling.finetune import lora as lora_module
from thesis.modelling.finetune.head import MotorClassifier
from thesis.modelling.finetune.lora import apply_lora, lora_config

# 2 blocks x (q, v) at the toy widths
TOY_ADAPTERS: int = 4


def adapted_modules(model: MotorClassifier) -> set[str]:
    """The module each adapter hangs off, with the peft scaffolding stripped."""
    return {
        name.split(".lora_")[0].removeprefix("encoder.base_model.model.")
        for name, _ in model.named_parameters()
        if ".lora_" in name
    }


def test_adapters_land_on_the_named_modules_only(make_classifier: Callable) -> None:
    """Query and value in every block, and nothing else."""
    model = apply_lora(make_classifier(), lora_config())

    assert adapted_modules(model) == {
        "blocks.0.input_proj.q_proj",
        "blocks.0.input_proj.v_proj",
        "blocks.1.input_proj.q_proj",
        "blocks.1.input_proj.v_proj",
    }


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        (("q_proj",), 2),
        (("q_proj", "v_proj"), 4),
        (("o_proj",), 2),
        (("q_proj", "k_proj", "v_proj", "ff_proj"), 8),
    ],
)
def test_the_configured_targets_are_the_ones_adapted(
    make_classifier: Callable, targets: tuple[str, ...], expected: int
) -> None:
    """Per-member target sets are one of the ensemble's diversity mechanisms."""
    model = apply_lora(make_classifier(), lora_config(target_modules=targets))

    assert len(adapted_modules(model)) == expected


def test_only_the_adapters_and_the_head_train(make_classifier: Callable) -> None:
    """Everything else is the released backbone, frozen."""
    model = apply_lora(make_classifier(), lora_config())

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}

    assert {"head.weight", "head.bias"} <= trainable
    assert all(".lora_" in name for name in trainable - {"head.weight", "head.bias"})
    assert len(trainable) == 2 * TOY_ADAPTERS + 2


def test_the_wrapped_encoder_starts_bit_identical(
    make_classifier: Callable, classifier_batch: dict
) -> None:
    """LoRA's B is zero-initialised, so at step zero the adapter adds exactly zero.

    Bit-identical, not close: any drift is the wrap having changed the computation,
    which would mean the backbone is no longer the released model.
    """
    plain = make_classifier(seed=0).eval()
    adapted = apply_lora(make_classifier(seed=0), lora_config()).eval()

    with torch.no_grad():
        assert torch.equal(plain(**classifier_batch), adapted(**classifier_batch))


def test_no_gradient_reaches_the_backbone(
    make_classifier: Callable, classifier_batch: dict
) -> None:
    """A frozen weight that still accumulates gradient is updated by the optimizer."""
    model = apply_lora(make_classifier(), lora_config())

    model(**classifier_batch).sum().backward()

    with_gradient = {n for n, p in model.named_parameters() if p.grad is not None}
    assert with_gradient == {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any(".base_layer." in name for name in with_gradient)


def test_wrapping_twice_is_refused(make_classifier: Callable) -> None:
    """Nesting one PeftModel in another trains adapters against a moving target."""
    model = apply_lora(make_classifier(), lora_config())

    with pytest.raises(ValueError, match="already carries adapters"):
        apply_lora(model, lora_config())


def test_the_encoder_becomes_a_peft_model(make_classifier: Callable) -> None:
    """The wrapper is what `load_adapter` and multi-adapter serving hang off."""
    model = apply_lora(make_classifier(), lora_config())

    assert isinstance(model.encoder, PeftModel)
    assert not isinstance(model.head, PeftModel)


def test_a_leaked_backbone_parameter_is_refused(
    make_classifier: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full fine-tune reported as LoRA produces entirely plausible metrics."""
    monkeypatch.setattr(lora_module, "get_peft_model", lambda model, config: model)

    with pytest.raises(ValueError, match="still trainable"):
        apply_lora(make_classifier(), lora_config())
