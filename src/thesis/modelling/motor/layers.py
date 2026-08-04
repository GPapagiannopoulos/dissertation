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

    Raises:
        ValueError: If ages is not a one dimensional float32 tensor.
    """
    if ages.dtype != torch.float32:
        raise ValueError(f"Ages must be float32, got {ages.dtype}.")
    if ages.ndim != 1:
        raise ValueError(f"Ages must be one dimensional, got {ages.ndim} dimensions.")

    # Generate 32 angular frequency values from 1.0 to 1e-8 radians per minute of age
    # We use different "clocks" to account for variable time differences between events
    # Fast clocks contribute resolution, slow clocks contribute range
    inv_freq = 1.0 / (10000 ** torch.linspace(0, 2, dim // 2, device=ages.device))
    angles = ages.unsqueeze(1) * inv_freq.unsqueeze(0)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype)
    cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype)
    return sin, cos


def apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Rotates each adjacent channel pair of x by the angle for its event.

    Args:
        x (torch.Tensor): Queries/ keys shaped (..., seq_len, dim)
        sin: Sine table output from 'rotary_tables' shaped (seq_len, dim)
        cos: Cosine table output from 'rotary_tables' shaped (seq_len, dim)

    Returns:
        torch.Tensor: x rotated shaped (..., seq_len, dim)

    Raises:
        ValueError: If any of the parameters have different dtypes, or
            if the inputs last two dimensions do not match with the tables'
            dimensions.
    """
    if not x.dtype == sin.dtype == cos.dtype:
        raise ValueError("The dtypes of the tables or the input do not match.")
    if not x.shape[-2:] == sin.shape == cos.shape:
        raise ValueError("The shapes of the tables or the input do not match.")

    rotated = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    return x * cos + rotated * sin
