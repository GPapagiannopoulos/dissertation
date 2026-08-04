"""Module implementing the ported MOTOR layers."""

import torch

from thesis.modelling.motor.constants import MOTOR_AGE_FEATURES, MOTOR_RMS_EPS


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


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Lays the heads back out as channels, undoing split_heads.

    Args:
        x (torch.Tensor): Attended values shaped (..., n_heads, seq_len, head_dim).

    Returns:
        torch.Tensor: The same values shaped (..., seq_len, n_heads * head_dim).

    Raises:
        ValueError: If there are not enough dimensions to hold heads.
    """
    if x.ndim < 3:
        raise ValueError(f"Expected at least three dimensions, got {x.ndim}.")

    return x.transpose(-3, -2).flatten(-2)


def load_haiku_linear(
    module: torch.nn.Linear, weight: torch.Tensor, bias: torch.Tensor
) -> None:
    """Copies haiku linear's parameters into a torch one.

    haiku stores (in_features, out_features) and computes x @ w + b.
    torch stores (out_features, in_features) and computes x @ w.T + b.

    Args:
        module (torch.nn.Linear): The destination
        weight (torch.Tensor): The checkpoint weight, in haiku's (in, out) layout.
        bias (torch.Tensor): The checkpoint bias.

    Raises:
        ValueError: If either parameter does not fit the module.
    """
    expected = (module.in_features, module.out_features)
    if tuple(weight.shape) != expected:
        raise ValueError(
            f"Expected a weight shaped {expected} in haiku's layout, "
            f"got {tuple(weight.shape)}."
        )
    if bias.shape != module.bias.shape:
        raise ValueError(
            f"Expected a bias shaped {tuple(module.bias.shape)}, "
            f"got {tuple(bias.shape)}."
        )

    with torch.no_grad():
        module.weight.copy_(weight.T)
        module.bias.copy_(bias)


def local_attention_mask(
    segment_ids: torch.Tensor, width: int, device: torch.device | None = None
) -> torch.Tensor:
    """Builds the boolean mask MOTOR's attention runs under.

    Three conditions, all of which must hold for query i to see key j:

    - causal, j <= i;
    - local, i - j <= width, inclusive, so a query sees width + 1 keys;
    - same segment, because MOTOR packs several subjects into one flat buffer and
      they must not attend across each other.

    Args:
        segment_ids (torch.Tensor): Which packed sequence each position belongs to,
            shaped (seq_len,). A single sequence is all zeros.
        width (int): How many positions back a query may reach.
        device (torch.device | None): Where to build the mask. Defaults to the
            segment ids' device.

    Returns:
        torch.Tensor: A (seq_len, seq_len) boolean mask, True where attention is
            allowed, in the sense torch's attn_mask takes.

    Raises:
        ValueError: If the segment ids are not one dimensional, or the width negative.
    """
    if segment_ids.ndim != 1:
        raise ValueError(
            f"The segment ids must be one dimensional, got {segment_ids.ndim}."
        )
    if width < 0:
        raise ValueError(f"The width must not be negative, got {width}.")

    device = device or segment_ids.device
    positions = torch.arange(segment_ids.shape[0], device=device)
    offsets = positions.unsqueeze(1) - positions.unsqueeze(0)

    within_window = (offsets >= 0) & (offsets <= width)
    same_segment = segment_ids.to(device).unsqueeze(1) == segment_ids.to(
        device
    ).unsqueeze(0)
    return within_window & same_segment


def local_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Runs masked attention over one head's queries, keys and values.

    No row can be entirely masked, because j = i satisfies all three conditions, so
    there is no all-infinite softmax to guard against.

    Args:
        queries (torch.Tensor): Shaped (..., n_heads, seq_len, head_dim).
        keys (torch.Tensor): The same shape as the queries.
        values (torch.Tensor): The same shape as the queries.
        mask (torch.Tensor): A boolean (seq_len, seq_len) mask from
            `local_attention_mask`, broadcast over the heads.

    Returns:
        torch.Tensor: The attended values, shaped like the queries.

    Raises:
        ValueError: If the mask is not boolean, or the three inputs disagree on shape.
    """
    if mask.dtype != torch.bool:
        raise ValueError(f"The mask must be boolean, got {mask.dtype}.")
    if not queries.shape == keys.shape == values.shape:
        raise ValueError("The queries, keys and values must share one shape.")

    # equivalent to femr's implementation
    return torch.nn.functional.scaled_dot_product_attention(
        queries, keys, values, attn_mask=mask
    )


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


class MotorBlock(torch.nn.Module):
    """One MOTOR transformer block.

    The attention and feed-forward branches run in parallel, both reading the one
    normed input rather than the feed-forward reading attention's output. That is why
    a block holds a single norm, and why its output projection is fed the two branches
    concatenated.

    The rotary tables and the attention mask are passed as arguments because
    they are the same for all twelve blocks.

    Attributes:
        norm (torch.nn.RMSNorm): The block's normalisation.
        input_proj (InputProjection): q, k, v and the feed-forward expansion.
        o_proj (torch.nn.Linear): Projects the two branches back to the model width.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_heads: int,
        eps: float = MOTOR_RMS_EPS,
    ) -> None:
        """Builds a block at the released model's widths.

        Args:
            hidden_size (int): The model width.
            intermediate_size (int): The feed-forward expansion's width.
            n_heads (int): How many attention heads share the channels.
            eps (float): The norm's epsilon, inside the square root.
        """
        super().__init__()
        self.n_heads = n_heads
        self.norm = torch.nn.RMSNorm(hidden_size, eps=eps)
        self.input_proj = InputProjection(
            hidden_size + MOTOR_AGE_FEATURES, hidden_size, intermediate_size
        )
        self.o_proj = torch.nn.Linear(hidden_size + intermediate_size, hidden_size)

    def load_haiku(
        self,
        norm_scale: torch.Tensor,
        fused_weight: torch.Tensor,
        fused_bias: torch.Tensor,
        output_weight: torch.Tensor,
        output_bias: torch.Tensor,
    ) -> None:
        """Copies one checkpoint block's five parameters in.

        Args:
            norm_scale (torch.Tensor): The RMSNorm scale, shaped (hidden_size,).
            fused_weight (torch.Tensor): The fused input projection, haiku's `linear`.
            fused_bias (torch.Tensor): Its bias.
            output_weight (torch.Tensor): The output projection, haiku's `linear_1`.
            output_bias (torch.Tensor): Its bias.

        Raises:
            ValueError: If any parameter does not fit its module.
        """
        if norm_scale.shape != self.norm.weight.shape:
            raise ValueError(
                f"Expected a scale shaped {tuple(self.norm.weight.shape)}, "
                f"got {tuple(norm_scale.shape)}."
            )

        with torch.no_grad():
            self.norm.weight.copy_(norm_scale)
        self.input_proj.load_fused(fused_weight, fused_bias)
        load_haiku_linear(self.o_proj, output_weight, output_bias)

    def forward(
        self,
        x: torch.Tensor,
        normed_ages: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Runs the block and adds its residual.

        Args:
            x (torch.Tensor): The residual stream, shaped (..., seq_len, hidden_size).
            normed_ages (torch.Tensor): The z-scored age per position, shaped
                (..., seq_len). Concatenated with its square onto every block's input.
            sin (torch.Tensor): The rotary sine table, shaped (seq_len, head_dim).
            cos (torch.Tensor): The rotary cosine table, the same shape.
            mask (torch.Tensor): The boolean attention mask, shaped
                (seq_len, seq_len).

        Returns:
            torch.Tensor: The updated residual stream, shaped like x.

        Raises:
            ValueError: If the ages do not match the stream's dtype. torch would
                promote silently, widening every tensor below this point.
        """
        if normed_ages.dtype != x.dtype:
            raise ValueError(
                f"The ages must match the stream's dtype, {x.dtype}, "
                f"got {normed_ages.dtype}."
            )

        normed = self.norm(x)
        with_ages = torch.cat(
            (normed, normed_ages.unsqueeze(-1), (normed_ages**2).unsqueeze(-1)), dim=-1
        )

        queries, keys, values, feed_forward = self.input_proj(with_ages)
        queries, keys, values = (
            split_heads(projected, self.n_heads)
            for projected in (queries, keys, values)
        )
        attention = merge_heads(
            local_attention(
                apply_rotary(queries, sin, cos),
                apply_rotary(keys, sin, cos),
                values,
                mask,
            )
        )

        # jax.nn.gelu defaults to the tanh approximation, torch's to the exact erf.
        # They differ by ~1e-4, which compounds over twelve blocks.
        feed_forward = torch.nn.functional.gelu(feed_forward, approximate="tanh")

        return x + self.o_proj(torch.cat((attention, feed_forward), dim=-1))
