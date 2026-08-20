"""Testing suite for assembling the released backbone under adapters.

These need the real checkpoint, so they skip on a clean checkout as the oracle
fixtures do.
"""

from pathlib import Path

import pytest
import torch

from thesis.modelling.backbone.checkpoint import RELEASED_CONFIG, load_released_params
from thesis.modelling.backbone.model import MotorEncoder
from thesis.modelling.finetune.lora import lora_classifier

_ORACLE_PATH: Path = (
    Path(__file__).resolve().parents[3] / "motor_output" / "oracle_fp32.npz"
)

# 12 blocks x (q, v), each an (8, 770) A and a (768, 8) B, plus Linear(1537, 1)
RELEASED_TRAINABLE: int = 296_834


@pytest.fixture
def oracle_path() -> Path:
    """The fp32 dump the released weights are read out of, or a skip."""
    if not _ORACLE_PATH.exists():
        pytest.skip(
            f"{_ORACLE_PATH} not found; regenerate with '.venv-motor-v1/bin/python "
            f"scripts/tools/dump_motor_oracle.py --dtype fp32'"
        )
    return _ORACLE_PATH


def test_the_released_weights_survive_the_wrap(oracle_path: Path) -> None:
    """Wrapped before the load, the base layer is a fresh Kaiming draw instead.

    Finite, correctly shaped, and trainable to a plausible-looking result.
    """
    expected = MotorEncoder(**RELEASED_CONFIG)
    expected.load_haiku(load_released_params(oracle_path))

    model = lora_classifier(oracle_path, 0.0356)

    adapted = model.get_submodule("encoder.base_model.model.blocks.0.input_proj.q_proj")
    assert torch.equal(
        adapted.base_layer.weight, expected.blocks[0].input_proj.q_proj.weight
    )


def test_the_released_model_trains_two_tenths_of_a_percent(oracle_path: Path) -> None:
    """296,834 of 135,481,346, which is what makes a member 1.13 MB."""
    model = lora_classifier(oracle_path, 0.0356)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == RELEASED_TRAINABLE


def test_the_head_starts_at_the_base_rate(oracle_path: Path) -> None:
    """The head is outside peft's freeze, so it still initialises from prevalence."""
    model = lora_classifier(oracle_path, 0.0356)

    assert model.head.weight.requires_grad
    assert not torch.any(model.head.weight)
    assert model.head.bias.item() == pytest.approx(-3.2992, abs=1e-4)
