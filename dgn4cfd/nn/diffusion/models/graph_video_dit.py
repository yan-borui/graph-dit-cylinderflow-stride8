"""Joint future-block Diffusion Transformer for graph UVP latents.

The model receives one clean posterior-mean latent frame and one noisy block of
future absolute latent frames.  Every ``(frame, latent-node)`` pair is a token,
and every token in one graph participates in the same non-causal self-attention
operation.  No future-clean tensor is accepted by ``forward`` or ``sample``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


CHECKPOINT_FORMAT_V1 = "dgn4cfd.graph_video_dit.joint.v1"
CHECKPOINT_FORMAT_V2 = "dgn4cfd.graph_video_dit.joint.v2"
CHECKPOINT_FORMAT = "dgn4cfd.graph_video_dit.joint.v3"
DEFAULT_FUTURE_FRAMES = 64
DEFAULT_DIFFUSION_STEPS = 1000
DEFAULT_SAMPLING_STEPS = 20
NeighborLayout = tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], torch.Tensor]


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Embed an arbitrary-shaped scalar tensor on a sinusoidal basis."""

    if dim < 2:
        raise ValueError("embedding dimension must be at least two")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = values.to(torch.float32).unsqueeze(-1) * frequencies
    result = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if result.size(-1) < dim:
        result = F.pad(result, (0, dim - result.size(-1)))
    return result


def _modulate(
    values: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return values * (1.0 + scale) + shift


def _slot_modulation(
    layer: nn.Module, condition: torch.Tensor, total_slots: int
) -> torch.Tensor:
    """Project clean/future diffusion conditions once, then broadcast over nodes."""
    pair = layer(condition)
    return torch.cat(
        (pair[:, :1], pair[:, 1:].expand(-1, total_slots - 1, -1)), dim=1
    ).unsqueeze(2)


class JointDiTBlock(nn.Module):
    """Standard full-attention DiT block with token-wise AdaLN-Zero."""

    def __init__(self, width: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.attention_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, width),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def _neighbor_attention(
        self, inputs: torch.Tensor, total_slots: int, layout: NeighborLayout
    ) -> torch.Tensor:
        """Evaluate exactly the unmasked H1 keys, grouped by spatial degree."""
        batch_size, token_count, width = inputs.shape
        num_nodes = token_count // total_slots
        heads = self.attention.num_heads
        head_width = width // heads
        query, key, value = F.linear(
            inputs, self.attention.in_proj_weight, self.attention.in_proj_bias
        ).chunk(3, dim=-1)
        query = query.reshape(batch_size, total_slots, num_nodes, heads, head_width)
        query = query.permute(0, 2, 3, 1, 4).reshape(
            batch_size * num_nodes, heads, total_slots, head_width
        )

        def node_values(tensor: torch.Tensor) -> torch.Tensor:
            return (
                tensor.reshape(batch_size, total_slots, num_nodes, heads, head_width)
                .permute(0, 2, 1, 3, 4)
                .reshape(batch_size * num_nodes, total_slots, heads, head_width)
            )

        key, value = node_values(key), node_values(value)
        groups, original_node_order = layout
        outputs = []
        for centers, neighbors in groups:
            count, degree = neighbors.shape

            def neighbor_values(tensor: torch.Tensor) -> torch.Tensor:
                # Time precedes neighbor index, matching the dense mask's key order.
                return (
                    tensor[neighbors]
                    .permute(0, 3, 2, 1, 4)
                    .reshape(count, heads, total_slots * degree, head_width)
                )

            outputs.append(
                F.scaled_dot_product_attention(
                    query[centers],
                    neighbor_values(key),
                    neighbor_values(value),
                    dropout_p=0.0,
                    is_causal=False,
                )
            )
        output = torch.cat(outputs, dim=0)[original_node_order]
        output = (
            output.reshape(batch_size, num_nodes, heads, total_slots, head_width)
            .permute(0, 3, 1, 2, 4)
            .reshape(batch_size, token_count, width)
        )
        return self.attention.out_proj(output)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        neighbor_layout: NeighborLayout | None = None,
    ) -> torch.Tensor:
        """Process [B, slots, nodes, width] with the original token ordering."""

        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = _slot_modulation(self.modulation, condition, tokens.size(1)).chunk(
            6, dim=-1
        )
        attention_input = _modulate(
            self.attention_norm(tokens), attention_shift, attention_scale
        ).flatten(1, 2)
        if neighbor_layout is not None:
            attention_output = self._neighbor_attention(
                attention_input, tokens.size(1), neighbor_layout
            )
        else:
            attention_kwargs = {"need_weights": False, "is_causal": False}
            if attention_bias is not None:
                attention_kwargs["attn_mask"] = attention_bias
            attention_output, _ = self.attention(
                attention_input,
                attention_input,
                attention_input,
                **attention_kwargs,
            )
        tokens = tokens + attention_gate * attention_output.reshape_as(tokens)
        mlp_input = _modulate(self.mlp_norm(tokens), mlp_shift, mlp_scale)
        return tokens + mlp_gate * self.mlp(mlp_input)


class JointFinalLayer(nn.Module):
    """AdaLN-modulated projection from transformer width to latent features."""

    def __init__(self, width: int, latent_features: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output = nn.Linear(width, latent_features)
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = _slot_modulation(
            self.modulation, condition, tokens.size(1)
        ).chunk(2, dim=-1)
        return self.output(_modulate(self.norm(tokens), shift, scale))


class GraphVideoDiT(nn.Module):
    """Jointly predict DDPM noise for a fixed block of future graph latents.

    Public tensor axes are ``B`` graphs, ``S`` frame slots, ``N`` coarsest graph
    nodes, and ``C`` latent features. Personal retraining uses ``B=1``, ``S=1+64``
    and ``C=4``; legacy representations remain configurable. H1 evaluates the
    exact graph-hop neighborhood, including all frame slots for every neighbor.
    """

    def __init__(
        self,
        *,
        latent_features: int = 1,
        condition_features: int = 126,
        width: int = 256,
        blocks: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        future_frames: int = DEFAULT_FUTURE_FRAMES,
        diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        attention_mode: Literal[
            "full", "graph_distance_bias", "graph_hop_mask"
        ] = "full",
        graph_bias_alpha: float = 0.0,
        graph_hop_limit: int | None = None,
    ) -> None:
        super().__init__()
        if latent_features < 1 or condition_features < 1:
            raise ValueError("latent and condition feature counts must be positive")
        if width < 1 or blocks < 1 or heads < 1 or width % heads:
            raise ValueError("invalid DiT width/block/head configuration")
        if mlp_ratio <= 0 or future_frames < 1 or diffusion_steps < 2:
            raise ValueError("invalid MLP, future-frame, or diffusion configuration")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("beta schedule endpoints must satisfy 0 < start < end < 1")
        if attention_mode not in (
            "full",
            "graph_distance_bias",
            "graph_hop_mask",
        ):
            raise ValueError(
                "attention_mode must be full, graph_distance_bias, or graph_hop_mask"
            )
        if not math.isfinite(graph_bias_alpha) or graph_bias_alpha < 0.0:
            raise ValueError("graph_bias_alpha must be finite and non-negative")
        if attention_mode == "full" and graph_bias_alpha != 0.0:
            raise ValueError("full attention requires graph_bias_alpha=0")
        if attention_mode != "graph_distance_bias" and graph_bias_alpha != 0.0:
            raise ValueError("graph_bias_alpha is only valid for graph_distance_bias")
        if attention_mode == "graph_hop_mask":
            if graph_hop_limit not in (1, 2):
                raise ValueError("graph_hop_mask requires graph_hop_limit=1 or 2")
        elif graph_hop_limit is not None:
            raise ValueError("graph_hop_limit is only valid for graph_hop_mask")

        self.latent_features = int(latent_features)
        self.condition_features = int(condition_features)
        self.width = int(width)
        self.num_blocks = int(blocks)
        self.heads = int(heads)
        self.mlp_ratio = float(mlp_ratio)
        self.future_frames = int(future_frames)
        self.diffusion_steps = int(diffusion_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.attention_mode = attention_mode
        self.graph_bias_alpha = float(graph_bias_alpha)
        self.graph_hop_limit = graph_hop_limit
        self.total_slots = self.future_frames + 1
        # Execution policy is recorded in the training config, not weight shapes.
        self.activation_checkpointing = False

        self.latent_encoder = nn.Linear(self.latent_features, self.width)
        self.condition_encoder = nn.Linear(self.condition_features, self.width)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
        )
        self.timestep_encoder = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
        )
        self.blocks = nn.ModuleList(
            [
                JointDiTBlock(self.width, self.heads, self.mlp_ratio)
                for _ in range(self.num_blocks)
            ]
        )
        self.final = JointFinalLayer(self.width, self.latent_features)

        betas = torch.linspace(
            self.beta_start, self.beta_end, self.diffusion_steps, dtype=torch.float32
        )
        alphas = 1.0 - betas
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0), persistent=True)
        self.register_buffer(
            "latent_mean",
            torch.zeros(1, 1, 1, self.latent_features),
            persistent=True,
        )
        self.register_buffer(
            "latent_std",
            torch.ones(1, 1, 1, self.latent_features),
            persistent=True,
        )

    def architecture(self) -> dict[str, int | float | str]:
        """Return the complete constructor configuration for checkpoints."""

        return {
            "latent_features": self.latent_features,
            "condition_features": self.condition_features,
            "width": self.width,
            "blocks": self.num_blocks,
            "heads": self.heads,
            "mlp_ratio": self.mlp_ratio,
            "future_frames": self.future_frames,
            "diffusion_steps": self.diffusion_steps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "attention_mode": self.attention_mode,
            "graph_bias_alpha": self.graph_bias_alpha,
            "graph_hop_limit": self.graph_hop_limit,
        }

    def set_latent_statistics(
        self, mean: torch.Tensor | float, std: torch.Tensor | float
    ) -> None:
        """Set absolute-latent statistics computed exclusively from Train."""

        mean_tensor = torch.as_tensor(
            mean, dtype=self.latent_mean.dtype, device=self.latent_mean.device
        ).reshape(1, 1, 1, -1)
        std_tensor = torch.as_tensor(
            std, dtype=self.latent_std.dtype, device=self.latent_std.device
        ).reshape(1, 1, 1, -1)
        if mean_tensor.numel() == 1:
            mean_tensor = mean_tensor.expand_as(self.latent_mean)
        if std_tensor.numel() == 1:
            std_tensor = std_tensor.expand_as(self.latent_std)
        if mean_tensor.shape != self.latent_mean.shape:
            raise ValueError("latent mean feature count is incorrect")
        if std_tensor.shape != self.latent_std.shape or torch.any(std_tensor <= 0):
            raise ValueError("latent standard deviations must be positive")
        self.latent_mean.copy_(mean_tensor)
        self.latent_std.copy_(std_tensor.clamp_min(1e-8))

    def normalise_latent(self, latent: torch.Tensor) -> torch.Tensor:
        mean = self.latent_mean.reshape(
            *([1] * (latent.ndim - 1)), self.latent_features
        )
        std = self.latent_std.reshape(*([1] * (latent.ndim - 1)), self.latent_features)
        return (latent - mean) / std

    def denormalise_latent(self, latent: torch.Tensor) -> torch.Tensor:
        mean = self.latent_mean.reshape(
            *([1] * (latent.ndim - 1)), self.latent_features
        )
        std = self.latent_std.reshape(*([1] * (latent.ndim - 1)), self.latent_features)
        return latent * std + mean

    def _validate_inputs(
        self,
        clean_z0: torch.Tensor,
        noisy_future: torch.Tensor,
        diffusion_step: torch.Tensor,
        node_context: torch.Tensor,
        positions: torch.Tensor,
        graph_distance: torch.Tensor | None,
        graph_hops: torch.Tensor | None,
    ) -> None:
        if clean_z0.ndim != 4 or clean_z0.size(1) != 1:
            raise ValueError("clean_z0 must have shape [B, 1, N, C]")
        if noisy_future.ndim != 4 or noisy_future.size(1) != self.future_frames:
            raise ValueError(
                f"noisy_future must have shape [B, {self.future_frames}, N, C]"
            )
        if clean_z0.shape[0:1] + clean_z0.shape[2:] != (
            noisy_future.shape[0:1] + noisy_future.shape[2:]
        ):
            raise ValueError("clean and future latent axes do not agree")
        if clean_z0.size(-1) != self.latent_features:
            raise ValueError("latent feature count differs from the architecture")
        batch_size, _, num_nodes, _ = noisy_future.shape
        if diffusion_step.shape != (batch_size,):
            raise ValueError("diffusion_step must have shape [B]")
        if diffusion_step.dtype not in (torch.int32, torch.int64):
            raise TypeError("diffusion_step must be an integer tensor")
        if torch.any(diffusion_step < 0) or torch.any(
            diffusion_step >= self.diffusion_steps
        ):
            raise ValueError("diffusion_step is outside the training schedule")
        if node_context.shape != (
            batch_size,
            num_nodes,
            self.condition_features,
        ):
            raise ValueError("node_context has the wrong shape")
        if positions.shape != (batch_size, num_nodes, 2):
            raise ValueError("positions must have shape [B, N, 2]")
        if self.attention_mode == "graph_distance_bias":
            if graph_distance is None:
                raise ValueError(
                    "graph_distance is required for graph-distance attention"
                )
            if graph_distance.shape != (batch_size, num_nodes, num_nodes):
                raise ValueError("graph_distance must have shape [B, N, N]")
            if not graph_distance.is_floating_point():
                raise TypeError("graph_distance must be floating point")
            if not bool(torch.isfinite(graph_distance).all().item()):
                raise ValueError("graph_distance must be finite")
            tolerance = 1e-6
            if bool(torch.any(graph_distance < -tolerance).item()) or bool(
                torch.any(graph_distance > 1.0 + tolerance).item()
            ):
                raise ValueError("graph_distance must lie in [0, 1]")
        if self.attention_mode == "graph_hop_mask":
            if graph_hops is None:
                raise ValueError("graph_hops is required for graph-hop attention")
            if graph_hops.shape != (batch_size, num_nodes, num_nodes):
                raise ValueError("graph_hops must have shape [B, N, N]")
            if graph_hops.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise TypeError("graph_hops must be an integer tensor")
            if bool(torch.any(graph_hops < -1).item()):
                raise ValueError("graph_hops must use -1 for disconnected pairs")
            diagonal = torch.diagonal(graph_hops, dim1=-2, dim2=-1)
            if bool(torch.any(diagonal != 0).item()):
                raise ValueError("graph_hops diagonal must be zero")
            allowed = (graph_hops >= 0) & (graph_hops <= int(self.graph_hop_limit))
            if not bool(torch.all(allowed.any(dim=-1)).item()):
                raise ValueError("graph-hop mask contains a fully masked row")

    def _neighbor_layout(self, graph_hops: torch.Tensor) -> NeighborLayout:
        """Group rows by degree without padding or dropping an allowed connection."""
        batch_size, num_nodes, _ = graph_hops.shape
        allowed = (graph_hops >= 0) & (graph_hops <= 1)
        node_indices = torch.arange(num_nodes, device=graph_hops.device)
        neighbors = torch.where(allowed, node_indices, num_nodes).sort(dim=-1).values
        neighbors = neighbors.reshape(batch_size * num_nodes, num_nodes)
        degrees = allowed.sum(dim=-1).reshape(-1).cpu().tolist()
        groups = []
        for degree in sorted(set(degrees)):
            centers = torch.tensor(
                [index for index, count in enumerate(degrees) if count == degree],
                device=graph_hops.device,
                dtype=torch.long,
            )
            selected = neighbors[centers, :degree]
            selected = selected + (centers // num_nodes)[:, None] * num_nodes
            groups.append((centers, selected))
        original_order = torch.argsort(torch.cat([centers for centers, _ in groups]))
        return tuple(groups), original_order

    def _attention_bias(
        self,
        graph_distance: torch.Tensor | None,
        graph_hops: torch.Tensor | None,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Expand one spatial relation over every non-causal frame-slot pair."""

        if self.attention_mode == "full":
            return None
        if self.attention_mode == "graph_distance_bias":
            if graph_distance is None:
                raise ValueError("graph_distance is required")
            batch_size, num_nodes, _ = graph_distance.shape
            spatial_bias = -self.graph_bias_alpha * graph_distance.to(dtype=dtype)
        else:
            if graph_hops is None:
                raise ValueError("graph_hops is required")
            batch_size, num_nodes, _ = graph_hops.shape
            allowed = (graph_hops >= 0) & (graph_hops <= int(self.graph_hop_limit))
            spatial_bias = torch.zeros(
                graph_hops.shape,
                dtype=dtype,
                device=graph_hops.device,
            ).masked_fill(~allowed, float("-inf"))
        token_bias = (
            spatial_bias[:, None, :, None, :]
            .expand(
                batch_size,
                self.total_slots,
                num_nodes,
                self.total_slots,
                num_nodes,
            )
            .reshape(
                batch_size,
                self.total_slots * num_nodes,
                self.total_slots * num_nodes,
            )
        )
        if batch_size == 1:
            return token_bias[0]
        token_count = self.total_slots * num_nodes
        return (
            token_bias[:, None]
            .expand(-1, self.heads, -1, -1)
            .reshape(batch_size * self.heads, token_count, token_count)
        )

    @staticmethod
    def _normalise_positions(positions: torch.Tensor) -> torch.Tensor:
        lower = positions.amin(dim=1, keepdim=True)
        scale = (positions.amax(dim=1, keepdim=True) - lower).clamp_min(
            torch.finfo(positions.dtype).eps
        )
        return 2.0 * (positions - lower) / scale - 1.0

    def _diffusion_conditions(
        self,
        diffusion_step: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Only diffusion step 0 and t differ; physical-time positions stay separate."""
        pair_steps = torch.stack(
            (torch.zeros_like(diffusion_step), diffusion_step), dim=1
        )
        return self.timestep_encoder(sinusoidal_embedding(pair_steps, self.width)).to(
            dtype
        )

    def _position_features(
        self, positions: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = positions.shape
        normalized = self._normalise_positions(positions)
        spatial = normalized[:, None].expand(-1, self.total_slots, -1, -1)
        offsets = torch.linspace(
            0.0,
            1.0,
            self.total_slots,
            device=positions.device,
            dtype=positions.dtype,
        )[None, :, None, None].expand(batch_size, -1, num_nodes, -1)
        coordinate_time = torch.cat([spatial, offsets], dim=-1)
        return self.position_encoder(coordinate_time).to(dtype)

    def forward(
        self,
        clean_z0: torch.Tensor,
        noisy_future: torch.Tensor,
        diffusion_step: torch.Tensor,
        node_context: torch.Tensor,
        positions: torch.Tensor,
        graph_distance: torch.Tensor | None = None,
        graph_hops: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict ``epsilon[B, future_frames, N, C]`` for one joint clip."""

        self._validate_inputs(
            clean_z0,
            noisy_future,
            diffusion_step,
            node_context,
            positions,
            graph_distance,
            graph_hops,
        )
        slots = torch.cat([clean_z0, noisy_future], dim=1)
        batch_size, _, num_nodes, _ = slots.shape
        tokens = (
            self.latent_encoder(slots)
            + self.condition_encoder(node_context)[:, None]
            + self._position_features(positions, dtype=slots.dtype)
        )
        diffusion_condition = self._diffusion_conditions(
            diffusion_step, dtype=tokens.dtype
        )
        neighbor_layout = None
        attention_bias = None
        if self.attention_mode == "graph_hop_mask" and self.graph_hop_limit == 1:
            neighbor_layout = self._neighbor_layout(graph_hops)
        else:
            attention_bias = self._attention_bias(
                graph_distance, graph_hops, dtype=tokens.dtype
            )
        for block in self.blocks:
            if (
                self.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                tokens = checkpoint(
                    block,
                    tokens,
                    diffusion_condition,
                    attention_bias,
                    neighbor_layout,
                    use_reentrant=False,
                )
            else:
                tokens = block(
                    tokens, diffusion_condition, attention_bias, neighbor_layout
                )
        predicted = self.final(tokens, diffusion_condition).reshape(
            batch_size,
            self.total_slots,
            num_nodes,
            self.latent_features,
        )
        return predicted[:, 1:]

    def corrupt_future(
        self,
        clean_future: torch.Tensor,
        diffusion_step: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one shared DDPM level and independent elementwise noise."""

        if clean_future.ndim != 4 or clean_future.size(1) != self.future_frames:
            raise ValueError("clean_future has the wrong frame axes")
        if diffusion_step.shape != (clean_future.size(0),):
            raise ValueError("diffusion_step must have shape [B]")
        noise = torch.randn(
            clean_future.shape,
            dtype=clean_future.dtype,
            device=clean_future.device,
            generator=generator,
        )
        alpha = self.alpha_bar[diffusion_step].to(clean_future.dtype)[
            :, None, None, None
        ]
        noisy = torch.sqrt(alpha) * clean_future + torch.sqrt(1.0 - alpha) * noise
        return noisy, noise

    def training_loss(
        self,
        clean_z0_raw: torch.Tensor,
        clean_future_raw: torch.Tensor,
        node_context: torch.Tensor,
        positions: torch.Tensor,
        *,
        graph_distance: torch.Tensor | None = None,
        graph_hops: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return plain epsilon MSE over future slots only."""

        batch_size = clean_z0_raw.size(0)
        steps = torch.randint(
            0,
            self.diffusion_steps,
            (batch_size,),
            device=clean_z0_raw.device,
            generator=generator,
        )
        clean_z0 = self.normalise_latent(clean_z0_raw)
        clean_future = self.normalise_latent(clean_future_raw)
        noisy, noise = self.corrupt_future(clean_future, steps, generator=generator)
        prediction = self(
            clean_z0,
            noisy,
            steps,
            node_context,
            positions,
            graph_distance=graph_distance,
            graph_hops=graph_hops,
        )
        return torch.mean((prediction - noise) ** 2)

    @torch.no_grad()
    def sample(
        self,
        clean_z0_raw: torch.Tensor,
        node_context: torch.Tensor,
        positions: torch.Tensor,
        *,
        graph_distance: torch.Tensor | None = None,
        graph_hops: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        sampling_steps: int = DEFAULT_SAMPLING_STEPS,
    ) -> torch.Tensor:
        """Jointly sample raw absolute future latents from pure Gaussian noise."""

        if sampling_steps < 2:
            raise ValueError("sampling_steps must be at least two")
        self.eval()
        clean_z0 = self.normalise_latent(clean_z0_raw)
        batch_size, _, num_nodes, _ = clean_z0.shape
        future = torch.randn(
            (
                batch_size,
                self.future_frames,
                num_nodes,
                self.latent_features,
            ),
            dtype=clean_z0.dtype,
            device=clean_z0.device,
            generator=generator,
        )
        schedule = (
            torch.linspace(
                self.diffusion_steps - 1,
                0,
                sampling_steps,
                device=clean_z0.device,
            )
            .round()
            .long()
        )
        schedule = torch.unique_consecutive(schedule)
        for index, step in enumerate(schedule):
            step_value = int(step.item())
            step_tensor = torch.full(
                (batch_size,),
                step_value,
                dtype=torch.long,
                device=clean_z0.device,
            )
            predicted_noise = self(
                clean_z0,
                future,
                step_tensor,
                node_context,
                positions,
                graph_distance=graph_distance,
                graph_hops=graph_hops,
            )
            alpha = self.alpha_bar[step_value].to(future.dtype)
            clean_estimate = (
                future - torch.sqrt(1.0 - alpha) * predicted_noise
            ) / torch.sqrt(alpha).clamp_min(torch.finfo(future.dtype).eps)
            if index == len(schedule) - 1:
                future = clean_estimate
            else:
                next_alpha = self.alpha_bar[int(schedule[index + 1].item())].to(
                    future.dtype
                )
                future = (
                    torch.sqrt(next_alpha) * clean_estimate
                    + torch.sqrt(1.0 - next_alpha) * predicted_noise
                )
        return self.denormalise_latent(future)

    def checkpoint_payload(self, **metadata) -> dict:
        """Build a stable model checkpoint payload."""

        return {
            "format": CHECKPOINT_FORMAT,
            "arch": self.architecture(),
            "state": {
                name: value.detach().cpu() for name, value in self.state_dict().items()
            },
            **metadata,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path | dict,
        *,
        device: torch.device | str = "cpu",
    ) -> tuple["GraphVideoDiT", dict]:
        """Load and strictly validate a joint-video checkpoint."""

        payload = (
            torch.load(checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict)
            else checkpoint
        )
        checkpoint_format = payload.get("format")
        if checkpoint_format not in (
            CHECKPOINT_FORMAT_V1,
            CHECKPOINT_FORMAT_V2,
            CHECKPOINT_FORMAT,
        ):
            raise ValueError("unsupported Graph-Video DiT checkpoint format")
        architecture = dict(payload["arch"])
        if checkpoint_format == CHECKPOINT_FORMAT_V1:
            architecture.setdefault("attention_mode", "full")
            architecture.setdefault("graph_bias_alpha", 0.0)
        if checkpoint_format in (CHECKPOINT_FORMAT_V1, CHECKPOINT_FORMAT_V2):
            architecture.setdefault("graph_hop_limit", None)
        model = cls(**architecture)
        model.load_state_dict(payload["state"], strict=True)
        model.to(device)
        return model, payload
