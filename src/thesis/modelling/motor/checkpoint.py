"""Getting the released MOTOR weights into a PyTorch encoder.

The release is a JAX/haiku pickle that only `.venv-motor-v1` can open, so the weights
reach this environment through the oracle dump `scripts/dump_motor_oracle.py` writes:
an npz holding every intermediate activation AND the flat parameter tree. Only the
parameters are read here.

`np.load` is lazy, so opening the 580 MB dump costs nothing until a key is touched;
the ~135 M parameters this pulls out are about 540 MB, and they are freed as soon as
`load_haiku` has copied them in.
"""

from pathlib import Path

import numpy as np
import torch

from thesis.modelling.motor.model import MotorEncoder

# every parameter in the dump is keyed "param::{module}::{leaf}", and the encoder's
# modules all hang off this haiku scope; the task head's sit on a sibling one
HAIKU_SCOPE: str = "EHRTransformer/~/TransformerFeaturizer/~/Transformer/~/"

# the released configuration, as its config.json declares it
RELEASED_CONFIG: dict[str, int] = {
    "vocab_size": 65536,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "n_heads": 12,
    "n_layers": 12,
    "attention_width": 496,
}


def load_released_params(oracle: Path) -> dict[str, torch.Tensor]:
    """Pulls the encoder's parameters out of an oracle dump.

    Args:
        oracle (Path): `motor_output/oracle_fp32.npz`.

    Returns:
        dict[str, torch.Tensor]: The parameters, keyed below the haiku scope exactly
            as `MotorEncoder.load_haiku` expects them.

    Raises:
        FileNotFoundError: If the dump is absent, with the command that writes it.
        KeyError: If the dump holds no parameter under the encoder's scope, which
            means it was written by a different script than it appears.
    """
    if not oracle.is_file():
        raise FileNotFoundError(
            f"{oracle} not found; regenerate with '.venv-motor-v1/bin/python "
            f"scripts/dump_motor_oracle.py --dtype fp32'."
        )

    prefix = f"param::{HAIKU_SCOPE}"
    with np.load(oracle) as dump:
        params = {
            key.removeprefix(prefix): torch.from_numpy(dump[key])
            for key in dump.files
            if key.startswith(prefix)
        }

    if not params:
        raise KeyError(f"{oracle} holds no parameter under {HAIKU_SCOPE!r}.")
    return params


def released_encoder(oracle: Path) -> MotorEncoder:
    """Builds the released encoder at its own widths and loads the weights in.

    Args:
        oracle (Path): `motor_output/oracle_fp32.npz`.

    Returns:
        MotorEncoder: The pretrained backbone, in float32.
    """
    encoder = MotorEncoder(**RELEASED_CONFIG)
    encoder.load_haiku(load_released_params(oracle))
    return encoder
