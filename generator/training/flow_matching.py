#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Flow matching utilities for DiT-based motion generation.

Implements the flow matching training objective and inference sampling
following the conditional flow matching framework.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class FlowMatchingScheduler:
    """
    Flow matching scheduler for training and inference.

    Uses Beta distribution for timestep sampling (biased toward noisier states)
    and linear interpolation for the probability path.
    """

    def __init__(
        self,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        clip_min: float = 1e-5,
        clip_max: float = 1.0 - 1e-5,
    ):
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.clip_min = clip_min
        self.clip_max = clip_max

    def sample_timestep(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample flow matching timesteps from Beta distribution.

        Args:
            batch_size: Number of timesteps to sample
            device: Target device

        Returns:
            [B] timesteps in [clip_min, clip_max]
        """
        dist = torch.distributions.Beta(self.beta_alpha, self.beta_beta)
        t = dist.sample((batch_size,)).to(device)
        return t.clamp(self.clip_min, self.clip_max)

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create noisy sample via linear interpolation.

        noisy = t * clean + (1 - t) * noise

        At t=0: pure noise. At t=1: clean sample.

        Args:
            clean: [B, T, D] clean trajectories
            noise: [B, T, D] random noise
            t: [B] timesteps

        Returns:
            [B, T, D] noisy trajectories
        """
        t = t.view(-1, 1, 1)  # [B, 1, 1] for broadcasting
        return t * clean + (1 - t) * noise

    def get_velocity_target(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the velocity target for flow matching.

        velocity = clean - noise
        (the direction from noise to clean data)

        Args:
            clean: [B, T, D] clean trajectories
            noise: [B, T, D] random noise

        Returns:
            [B, T, D] velocity targets
        """
        return clean - noise

class FlowMatchingLoss(nn.Module):
    """
    Flow matching loss: MSE between predicted and target velocity.
    """

    def forward(
        self,
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching loss.

        Args:
            predicted_velocity: [B, T, D] predicted velocity
            target_velocity: [B, T, D] target velocity
            mask: Optional [B, T] validity mask

        Returns:
            Dict with loss values
        """
        diff = (predicted_velocity - target_velocity) ** 2

        if mask is not None:
            mask = mask.unsqueeze(-1)  # [B, T, 1]
            loss = (diff * mask).sum() / (mask.sum() * predicted_velocity.shape[-1] + 1e-8)
        else:
            loss = diff.mean()

        return {"loss": loss, "velocity_mse": loss}
