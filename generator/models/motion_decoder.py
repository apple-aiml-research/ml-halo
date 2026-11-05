#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
DiT (Diffusion Transformer) motion decoder with flow matching.

GR00T N1-style architecture: self-attention among action tokens,
cross-attention to VLM conditioning tokens, and AdaLN for
timestep conditioning.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class TimestepEncoder(nn.Module):
    """Encode diffusion timestep into an embedding vector."""

    def __init__(self, hidden_dim: int, frequency_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_dim = frequency_dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestep: [B] scalar-per-sample, or [B, T] per-token timesteps in
                [0, 1] (per-token enables prefix frames at t=1 for prev_chunk
                inpainting; see AdaLayerNorm).

        Returns:
            [B, hidden_dim] or [B, T, hidden_dim] timestep embeddings
        """
        # Sinusoidal frequency encoding. A trailing feature dim is appended, so
        # this works for both [B] -> [B, D] and [B, T] -> [B, T, D].
        half_dim = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=timestep.device, dtype=timestep.dtype) / half_dim
        )
        args = timestep.unsqueeze(-1) * freqs  # [..., half_dim]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [..., frequency_dim]

        return self.mlp(embedding)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep embedding."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, 2 * hidden_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] input features
            t_emb: [B, D] (one timestep for the whole sequence) or [B, T, D]
                (per-token timestep, e.g. prev_chunk prefix at t=1)

        Returns:
            [B, T, D] normalized and modulated features
        """
        # Compute scale and shift from timestep. [B, D] -> [B, 1, 2D] broadcasts
        # across tokens; [B, T, D] -> [B, T, 2D] modulates each token separately.
        params = self.proj(t_emb)
        if params.dim() == 2:
            params = params.unsqueeze(1)  # [B, 1, 2D]
        scale, shift = params.chunk(2, dim=-1)  # each [B, 1|T, D]

        return self.norm(x) * (1 + scale) + shift


class DiTBlock(nn.Module):
    """
    Single DiT transformer block.

    1. AdaLN → Self-Attention (among action tokens) → residual
    2. AdaLN → Cross-Attention (to VLM conditioning) → residual
    3. AdaLN → FFN → residual
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        feedforward_dim: Optional[int] = None,
    ):
        super().__init__()

        if feedforward_dim is None:
            feedforward_dim = hidden_dim * 4

        # Self-attention
        self.norm1 = AdaLayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention
        self.norm2 = AdaLayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN
        self.norm3 = AdaLayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        t_emb: torch.Tensor,
        conditioning_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] action tokens
            conditioning: [B, N, D] VLM conditioning tokens
            t_emb: [B, D] timestep embedding
            conditioning_mask: Optional [B, N] padding mask for conditioning

        Returns:
            [B, T, D] updated action tokens
        """
        # Self-attention
        normed = self.norm1(x, t_emb)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + attn_out

        # Cross-attention to VLM tokens
        normed = self.norm2(x, t_emb)
        cross_out, _ = self.cross_attn(
            normed, conditioning, conditioning,
            key_padding_mask=conditioning_mask,
        )
        x = x + cross_out

        # FFN
        normed = self.norm3(x, t_emb)
        x = x + self.ffn(normed)

        return x


class DiTMotionDecoder(nn.Module):
    """
    Diffusion Transformer decoder for motion trajectory generation.

    Uses flow matching to iteratively denoise noisy trajectories,
    conditioned on VLM features via cross-attention and diffusion
    timestep via adaptive layer normalization.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        output_dim: int = 36,
        num_layers: int = 8,
        num_heads: int = 8,
        trajectory_length: int = 30,
        dropout: float = 0.1,
        feedforward_dim: Optional[int] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.trajectory_length = trajectory_length

        # Project noisy trajectory into hidden space
        self.action_in_proj = nn.Linear(output_dim, hidden_dim)

        # Temporal positional encoding for action tokens
        self.temporal_pe = SinusoidalPositionalEncoding(hidden_dim)

        # Timestep encoder
        self.timestep_encoder = TimestepEncoder(hidden_dim)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                feedforward_dim=feedforward_dim,
            )
            for _ in range(num_layers)
        ])

        # Final norm and projection
        self.final_norm = AdaLayerNorm(hidden_dim)
        self.action_out_proj = nn.Linear(hidden_dim, output_dim)

        # Initialize output projection to near-zero for stable training
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        conditioning: torch.Tensor,
        timestep: torch.Tensor,
        conditioning_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity field for flow matching.

        Args:
            noisy_trajectory: [B, T, output_dim] noisy trajectory
            conditioning: [B, D] or [B, N, D] VLM conditioning features
            timestep: [B] (one per sample) or [B, T] (per-token; prev_chunk
                prefix frames at t=1) diffusion timestep in [0, 1]
            conditioning_mask: Optional [B, N] padding mask

        Returns:
            [B, T, output_dim] predicted velocity
        """
        T = self.trajectory_length
        device = noisy_trajectory.device

        # Project noisy trajectory to hidden dim
        x = self.action_in_proj(noisy_trajectory)  # [B, T, D]

        # Add temporal positional encoding
        x = x + self.temporal_pe(T, device)

        # Encode timestep
        t_emb = self.timestep_encoder(timestep)  # [B, D]

        # Ensure conditioning is 3D for cross-attention
        if conditioning.dim() == 2:
            conditioning = conditioning.unsqueeze(1)  # [B, 1, D]

        # Pass through DiT blocks
        for block in self.blocks:
            x = block(x, conditioning, t_emb, conditioning_mask)

        # Final norm and projection
        x = self.final_norm(x, t_emb)
        velocity = self.action_out_proj(x)  # [B, T, output_dim]

        return velocity


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return self.pe[:seq_len].to(device)
