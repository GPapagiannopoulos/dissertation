"""Testing suite for building a LoRA configuration from the registered defaults."""

import json
from typing import Any

import pytest
from peft import LoraConfig

from thesis.modelling.finetune.lora import LORA_DEFAULTS, config_record, lora_config


def test_the_registered_defaults_reach_the_configuration() -> None:
    """r=8 on q and v with alpha=32 is the design of record."""
    config = lora_config()

    assert isinstance(config, LoraConfig)
    assert config.r == 8
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.0
    assert config.bias == "none"
    assert set(config.target_modules) == {"q_proj", "v_proj"}


def test_the_targets_are_not_left_as_a_tuple() -> None:
    """A list is normalised to a set; any other type reaches the matcher as is."""
    assert not isinstance(lora_config().target_modules, tuple)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("r", 16),
        ("lora_alpha", 64),
        ("lora_dropout", 0.1),
        ("target_modules", ("q_proj", "k_proj", "v_proj", "ff_proj")),
    ],
)
def test_an_override_wins_over_the_default(field: str, value: Any) -> None:
    """Per-member ranks and targets are the ensemble's diversity mechanism."""
    config = lora_config(**{field: value})

    actual = getattr(config, field)
    assert set(actual) == set(value) if field == "target_modules" else actual == value


def test_the_defaults_are_not_mutated_by_an_override() -> None:
    """The constant is module level, so only the second call would be wrong."""
    before = dict(LORA_DEFAULTS)

    lora_config(r=32, target_modules=("o_proj",))

    assert LORA_DEFAULTS == before
    assert lora_config().r == 8


def test_a_config_record_rebuilds_the_configuration() -> None:
    """A checkpoint carries this; alpha cannot be read back off the saved tensors."""
    original = lora_config(r=16, lora_alpha=64, target_modules=("q_proj", "o_proj"))

    rebuilt = lora_config(**config_record(original))

    assert rebuilt.r == original.r
    assert rebuilt.lora_alpha == original.lora_alpha
    assert rebuilt.lora_dropout == original.lora_dropout
    assert set(rebuilt.target_modules) == set(original.target_modules)


def test_a_config_record_survives_json() -> None:
    """It rides in the manifest as well as the checkpoint, and a set would not."""
    record = config_record(lora_config())

    assert json.loads(json.dumps(record)) == record
