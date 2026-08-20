"""Stage 8: LoRA adapters over the frozen MOTOR backbone.

Everything but the model construction is shared with the full fine-tune: the data
stream, the collate, the head, the loop and the metrics are the same objects.

At the released configuration this trains 296,834 of 135,481,346 parameters (0.219%)
over 24 adapters, and a checkpoint is 1.13 MB against the full fine-tune's 517 MB.
"""

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from thesis.modelling.backbone.checkpoint import released_encoder, strip_compile_prefix
from thesis.modelling.finetune.head import MotorClassifier

LORA_DEFAULTS: dict[str, Any] = {
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": ("q_proj", "v_proj"),
    "bias": "none",
}
"""The configuration the experimental design registered, before any tuning.

Targets match by name suffix, so two strings reach all twelve blocks. Dropout is a
knob rather than a decision: the PEFT arm needs diversity between ensemble members.
"""


def lora_config(**overrides: Any) -> LoraConfig:
    """Builds a LoRA configuration from the registered defaults.

    Args:
        **overrides (Any): Any `LoraConfig` field, overriding `LORA_DEFAULTS`.

    Returns:
        LoraConfig: The configuration, with no `task_type` -- MOTOR is not one of
            transformers' task models.
    """
    settings = {**LORA_DEFAULTS, **overrides}
    # peft normalises a list of targets to a set and leaves any other type alone
    settings["target_modules"] = list(settings["target_modules"])
    return LoraConfig(**settings)


def config_record(config: LoraConfig) -> dict[str, Any]:
    """The fields `lora_config` needs to rebuild this configuration, JSON-safe.

    `alpha` is the reason this exists: it scales the adapter by alpha/r and is not
    recoverable from the saved tensors, so a checkpoint that does not carry it can
    only be rebuilt by guessing.

    Args:
        config (LoraConfig): The configuration a run trained under.

    Returns:
        dict[str, Any]: Keyword arguments for `lora_config`.
    """
    return {
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": sorted(config.target_modules),
    }


def apply_lora(model: MotorClassifier, config: LoraConfig) -> MotorClassifier:
    """Freezes the backbone in place and injects adapters into it.

    The encoder is wrapped rather than the classifier, so the head keeps the name
    `run_training` gives its own learning rate to and stays out of the freeze.

    Args:
        model (MotorClassifier): A classifier holding the weights it starts from.
        config (LoraConfig): Which modules to adapt, and at what rank.

    Returns:
        MotorClassifier: The same object, its encoder now a `PeftModel`.

    Raises:
        ValueError: If the encoder already carries adapters, or if any non-adapter
            parameter is left trainable.
    """
    if isinstance(model.encoder, PeftModel):
        raise ValueError(
            "This classifier's encoder already carries adapters. Wrapping twice "
            "nests one PeftModel in another; build a fresh classifier instead."
        )

    model.encoder = get_peft_model(model.encoder, config)

    # a freeze that silently fails is a full fine-tune reported as LoRA, which shows
    # up in no metric
    leaked = [
        name
        for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad and ".lora_" not in name
    ]
    if leaked:
        raise ValueError(
            f"{len(leaked)} backbone parameter(s) are still trainable after "
            f"wrapping, starting with {leaked[:3]}."
        )
    return model


def lora_classifier(
    oracle: Path, positive_rate: float, config: LoraConfig | None = None
) -> MotorClassifier:
    """Builds the released backbone, a fresh head, and adapters over the two.

    Args:
        oracle (Path): `motor_output/oracle_fp32.npz`.
        positive_rate (float): The training fold's prevalence, for the head's bias.
        config (LoraConfig | None): Defaults to `lora_config()`.

    Returns:
        MotorClassifier: The pretrained backbone, frozen and adapted, under a
            zero-initialised head.
    """
    # weights in, THEN wrap: `load_haiku` walks module names, and wrapping moves
    # every one of them under `base_model.model.*` and `base_layer`
    model = MotorClassifier(released_encoder(oracle), positive_rate=positive_rate)
    return apply_lora(model, config if config is not None else lora_config())


def trainable_summary(model: torch.nn.Module) -> dict[str, float]:
    """Counts what will actually move, for the run's log line and its manifest.

    Args:
        model (torch.nn.Module): Any model, adapted or not.

    Returns:
        dict[str, float]: `total`, `trainable`, the `fraction` that trains, and how
            many `adapters` were injected.

    Raises:
        ValueError: If the model holds no parameter at all.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    if not total:
        raise ValueError("This model holds no parameters; there is nothing to count.")

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    # one adapter contributes several tensors, so count the module they hang off
    adapters = {
        name.split(".lora_")[0]
        for name, _ in model.named_parameters()
        if ".lora_" in name
    }
    return {
        "total": float(total),
        "trainable": float(trainable),
        "fraction": trainable / total,
        "adapters": float(len(adapters)),
    }


def adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The trainable tensors alone -- the adapters and the head.

    Args:
        model (torch.nn.Module): An adapted classifier, compiled or not.

    Returns:
        dict[str, torch.Tensor]: The trainable parameters, keyed as an uncompiled
            model names them.

    Raises:
        ValueError: If nothing in the model is trainable.
    """
    trainable = strip_compile_prefix(
        {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
    )
    if not trainable:
        raise ValueError(
            "No parameter in this model requires grad, so there is no adapter to "
            "save. Was apply_lora called?"
        )

    state = strip_compile_prefix(model.state_dict())
    return {name: state[name] for name in trainable}


def load_adapter(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Loads an adapter checkpoint into a model built the same way.

    The frozen backbone is absent from the file by design, so this loads
    non-strictly and checks the other direction: every key has to land.

    Args:
        model (torch.nn.Module): A classifier at the same configuration the state
            was saved from, **before** `torch.compile`.
        state (dict[str, torch.Tensor]): An `adapter_state` mapping.

    Raises:
        ValueError: If any key in the state matches nothing in the model.
    """
    unexpected = model.load_state_dict(state, strict=False).unexpected_keys
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} key(s) in this adapter match nothing in the model, "
            f"starting with {sorted(unexpected)[:3]}. Either the LoRA configuration "
            f"differs from the one it was trained under, or the model has already "
            f"been compiled -- load the adapter first."
        )
