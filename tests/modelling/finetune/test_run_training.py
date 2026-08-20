"""Testing suite for what `run_training` writes into a checkpoint.

Only the `save_state` hook is covered here; the loop itself still has no tests.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from thesis.modelling.finetune.lora import (
    adapter_state,
    apply_lora,
    config_record,
    lora_config,
)
from thesis.modelling.finetune.training import run_training


@pytest.fixture
def training_batch(classifier_batch: dict) -> dict:
    """A forward batch with the supervision `run_training` pops off it."""
    return {
        **classifier_batch,
        "labels": torch.tensor([0.0]),
        "label_subjects": torch.tensor([11]),
        "label_times": torch.tensor([100]),
    }


def _train(model: torch.nn.Module, dest: Path, training_batch: dict, **kwargs) -> dict:
    """Runs two steps and returns the final checkpoint."""
    run_training(
        model,
        iter([training_batch for _ in range(4)]),
        [training_batch],
        dest,
        device=torch.device("cpu"),
        total_steps=2,
        warmup=1,
        accumulate=1,
        eval_every=1,
        eval_batches=1,
        checkpoint_every=1,
        verbose=False,
        **kwargs,
    )
    return torch.load(dest / "last.pt", map_location="cpu", weights_only=False)


def test_the_whole_state_dict_is_the_default(
    make_classifier: Callable, training_batch: dict, tmp_path: Path
) -> None:
    """The full fine-tune's files must not change shape because a hook exists."""
    model = make_classifier()

    saved = _train(model, tmp_path / "run", training_batch)

    assert set(saved["model"]) == set(model.state_dict())


def test_the_hook_decides_what_is_written(
    make_classifier: Callable, training_batch: dict, tmp_path: Path
) -> None:
    """A LoRA run writes the adapters and the head, not the frozen backbone."""
    model = apply_lora(make_classifier(), lora_config())

    saved = _train(model, tmp_path / "run", training_batch, save_state=adapter_state)

    assert set(saved["model"]) == set(adapter_state(model))
    assert not any(".base_layer." in name for name in saved["model"])


def test_checkpoint_extra_rides_in_every_file(
    make_classifier: Callable, training_batch: dict, tmp_path: Path
) -> None:
    """A file has to say what must be rebuilt to load it, including a killed run's."""
    dest = tmp_path / "run"
    model = apply_lora(make_classifier(), lora_config())
    record = config_record(lora_config())

    _train(model, dest, training_batch, checkpoint_extra={"lora": record})

    for name in ("last.pt", "step_000001.pt"):
        saved = torch.load(dest / name, map_location="cpu", weights_only=False)
        assert saved["lora"] == record


def test_the_periodic_checkpoints_use_the_hook_too(
    make_classifier: Callable, training_batch: dict, tmp_path: Path
) -> None:
    """These are the run's candidates, so a full-sized one defeats the point."""
    dest = tmp_path / "run"
    model = apply_lora(make_classifier(), lora_config())

    _train(model, dest, training_batch, save_state=adapter_state)

    periodic = torch.load(
        dest / "step_000001.pt", map_location="cpu", weights_only=False
    )
    assert set(periodic["model"]) == set(adapter_state(model))
