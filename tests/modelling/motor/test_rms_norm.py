"""Asserts torch's RMSNorm still computes what haiku's does.

torch.nn.RMSNorm already implements haiku's convention. The port therefore calls torch's
layer directly. The test is a guard against future implementations drifting.
"""

from collections.abc import Callable

import numpy as np
import torch
from numpy.lib.npyio import NpzFile

from thesis.modelling.motor.constants import MOTOR_RMS_EPS


def test_pytorch_rmsnorm_matches_oracle(
    jax_oracle: NpzFile, oracle_param: Callable[[str], np.ndarray]
) -> None:
    """Compares torch's RMSNorm against the JAX model's own in_norm output."""
    # embedded is in_norm's input only because every token in the synthetic batch is
    # valid; the model substitutes ones at invalid positions before normalising.
    assert jax_oracle["batch_valid_tokens"].all()

    scale = torch.from_numpy(oracle_param("rms_norm::scale"))
    hidden_size = scale.shape[0]

    norm = torch.nn.RMSNorm(hidden_size, eps=MOTOR_RMS_EPS)
    with torch.no_grad():
        norm.weight.copy_(scale)

    result = norm(torch.from_numpy(jax_oracle["embedded"]))

    torch.testing.assert_close(
        result,
        torch.from_numpy(jax_oracle["after_in_norm"]),
        rtol=0,
        atol=1e-6,
    )
