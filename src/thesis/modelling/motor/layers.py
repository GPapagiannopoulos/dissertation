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


def split_fused_projection(
    weight: torch.Tensor, bias: torch.Tensor, hidden_size: int
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Cuts the checkpoint's one fused projection into four named ones.

    MOTOR trains q, k, v and the feed-forward as a single matmul, so the checkpoint
    holds one (in_features, 3 * hidden_size + intermediate_size) matrix. peft resolves
    LoRA targets by module name, and a fused weight can only be adapted as a whole.
    Therefore the port keeps them apart.

    Args:
        weight (torch.Tensor): The fused weight, shaped
            (in_features, 3 * hidden_size + intermediate_size).
        bias (torch.Tensor): The fused bias, shaped (out_features,).
        hidden_size (int): The model width, which is also q, k and v's width.

    Returns:
        dict[str, tuple[torch.Tensor, torch.Tensor]]: A (weight, bias) tuple per
            projection, keyed by the names InputProjection uses. Each
            weight is contiguous and shaped (out_features, in_features).

    Raises:
        ValueError: If the weight is not two dimensional, if the bias does not match
            its output width, or if there is no room for a feed-forward slice.
    """
    if weight.ndim != 2:
        raise ValueError(f"The weight must be two dimensional, got {weight.ndim}.")
    if bias.shape != weight.shape[-1:]:
        raise ValueError(
            f"The bias must be shaped {tuple(weight.shape[-1:])}, "
            f"got {tuple(bias.shape)}."
        )
    if weight.shape[1] <= 3 * hidden_size:
        raise ValueError(
            f"The weight must be wider than the {3 * hidden_size} columns q, k and v "
            f"claim, got {weight.shape[1]}."
        )

    bounds = {
        "q_proj": (0, hidden_size),
        "k_proj": (hidden_size, 2 * hidden_size),
        "v_proj": (2 * hidden_size, 3 * hidden_size),
        "ff_proj": (3 * hidden_size, weight.shape[1]),
    }
    return {
        name: (weight[:, low:high].T.contiguous(), bias[low:high].clone())
        for name, (low, high) in bounds.items()
    }


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Gives each attention head its own slice of the channels.

    Head h owns a contiguous block of channels, h * head_dim to (h + 1) * head_dim.

    Args:
        x (torch.Tensor): Queries, keys or values shaped (..., seq_len, hidden_size).
        n_heads (int): How many heads share the channels.

    Returns:
        torch.Tensor: The same values shaped (..., n_heads, seq_len, hidden_size //
            n_heads).

    Raises:
        ValueError: If the head count does not divide the channels.
    """
    if x.shape[-1] % n_heads:
        raise ValueError(
            f"{n_heads} heads do not divide {x.shape[-1]} channels evenly."
        )

    return x.unflatten(-1, (n_heads, -1)).transpose(-3, -2)


class InputProjection(torch.nn.Module):
    """The four projections at the front of a MOTOR block.

    q, k and v feed attention. ff is the feed-forward branch's expansion, which runs
    in parallel with attention. All four read the same input, so the checkpoint fuses
    them into one matmul; see split_fused_projection for why the port does not.

    Attributes:
        q_proj (torch.nn.Linear): Query projection, named for peft's target strings.
        k_proj (torch.nn.Linear): Key projection.
        v_proj (torch.nn.Linear): Value projection.
        ff_proj (torch.nn.Linear): Feed-forward expansion.
    """

    def __init__(
        self, in_features: int, hidden_size: int, intermediate_size: int
    ) -> None:
        """Builds the four projections.

        Args:
            in_features (int): The block's width plus the concatenated age columns.
            hidden_size (int): The model width, which q, k and v each return.
            intermediate_size (int): The feed-forward expansion's width.
        """
        super().__init__()
        self.q_proj = torch.nn.Linear(in_features, hidden_size)
        self.k_proj = torch.nn.Linear(in_features, hidden_size)
        self.v_proj = torch.nn.Linear(in_features, hidden_size)
        self.ff_proj = torch.nn.Linear(in_features, intermediate_size)

    def load_fused(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        """Copies one fused checkpoint projection into the four submodules.

        Copies rather than assigns, so the parameters keep their own storage and never
        alias the checkpoint buffer.

        Args:
            weight (torch.Tensor): The fused weight, in haiku's (in, out) layout.
            bias (torch.Tensor): The fused bias.

        Raises:
            ValueError: If a slice does not fit the submodule it belongs to.
        """
        parts = split_fused_projection(weight, bias, self.q_proj.out_features)
        with torch.no_grad():
            for name, (part_weight, part_bias) in parts.items():
                projection: torch.nn.Linear = getattr(self, name)
                if part_weight.shape != projection.weight.shape:
                    raise ValueError(
                        f"{name} expects a weight shaped "
                        f"{tuple(projection.weight.shape)}, got "
                        f"{tuple(part_weight.shape)}."
                    )
                projection.weight.copy_(part_weight)
                projection.bias.copy_(part_bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Projects the normed input, ages already concatenated, four ways.

        Args:
            x (torch.Tensor): The block input shaped (..., seq_len, in_features).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: q, k and v
                shaped (..., seq_len, hidden_size), then ff shaped
                (..., seq_len, intermediate_size).
        """
        return self.q_proj(x), self.k_proj(x), self.v_proj(x), self.ff_proj(x)
