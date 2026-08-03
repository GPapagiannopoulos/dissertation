"""Module implementing the ported MOTOR layers."""

import torch


def rotary_tables(
    ages: torch.Tensor, dim: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds the rotary tables from the patients' ages.

    MOTOR does not see time. Instead, it rotates adjacent channels
    within one event's vector in the attention head, proportional to
    the subject's age.

    Args:
        ages (torch.Tensor): Tensor shaped (seq_len,) containing the subject's
            age at each event.
        dim (int): The width of one attention head.
        dtype (torch.dtype): The tensor dtype

    Returns:
        tuple[torch.Tensor, torch.Tensor]: two tensors containing the
            sine and cosine values. Each has shape (seq_len, dim)
    """
    assert ages.dtype == torch.float32
    assert ages.ndim == 1

    # Generate 32 angular frequency values from 1.0 to 1e-8 radians per minute of age
    # We use different "clocks" to account for variable time differences between events
    # Fast clocks contribute resolution, slow clocks contribute range
    inv_freq = 1.0 / (10000 ** torch.linspace(0, 2, dim // 2, device=ages.device))
    angles = ages.unsqueeze(1) * inv_freq.unsqueeze(0)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype)
    cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype)
    return sin, cos
