#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Data transforms: the motion normalizer and the image transform for the VLM input.
"""

import torch
import torchvision.transforms as T
from typing import Tuple, Optional, Dict
import numpy as np


# ── Quaternion utilities ([w, x, y, z] convention) ─────────────────────────


class MotionNormalizer:
    """
    Normalize and denormalize motion trajectories.

    Computes statistics from training data and applies z-score normalization.
    Quaternions are not normalized (they're already unit quaternions).
    """

    def __init__(
        self,
        joint_mean: Optional[np.ndarray] = None,
        joint_std: Optional[np.ndarray] = None,
        pelvis_pos_mean: Optional[np.ndarray] = None,
        pelvis_pos_std: Optional[np.ndarray] = None,
    ):
        self.joint_mean = joint_mean
        self.joint_std = joint_std
        self.pelvis_pos_mean = pelvis_pos_mean
        self.pelvis_pos_std = pelvis_pos_std
        self._fitted = joint_mean is not None

    def fit(self, trajectories: np.ndarray):
        """
        Compute normalization statistics from data.

        Args:
            trajectories: [N, T, 36] array of trajectories
        """
        # Flatten to [N*T, 36]
        flat = trajectories.reshape(-1, 36)

        # Joint positions (0:29)
        joints = flat[:, :29]
        self.joint_mean = joints.mean(axis=0).astype(np.float32)
        self.joint_std = joints.std(axis=0).astype(np.float32) + 1e-8

        # Pelvis position (29:32)
        pelvis_pos = flat[:, 29:32]
        self.pelvis_pos_mean = pelvis_pos.mean(axis=0).astype(np.float32)
        self.pelvis_pos_std = pelvis_pos.std(axis=0).astype(np.float32) + 1e-8

        # Quaternions (32:36) - don't normalize, they're already unit
        self._fitted = True

    def normalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Normalize trajectory.

        Args:
            trajectory: [T, 36] or [B, T, 36] tensor

        Returns:
            Normalized trajectory of same shape
        """
        if not self._fitted:
            raise RuntimeError("MotionNormalizer not fitted. Call fit() first.")

        device = trajectory.device
        out = trajectory.clone()

        # Convert stats to tensors
        joint_mean = torch.tensor(self.joint_mean, device=device)
        joint_std = torch.tensor(self.joint_std, device=device)
        pelvis_pos_mean = torch.tensor(self.pelvis_pos_mean, device=device)
        pelvis_pos_std = torch.tensor(self.pelvis_pos_std, device=device)

        # Normalize joints
        out[..., :29] = (out[..., :29] - joint_mean) / joint_std

        # Normalize pelvis position
        out[..., 29:32] = (out[..., 29:32] - pelvis_pos_mean) / pelvis_pos_std

        # Quaternions unchanged (32:36)
        return out

    def denormalize(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Denormalize trajectory back to original scale.

        Args:
            trajectory: [T, 36] or [B, T, 36] normalized tensor

        Returns:
            Denormalized trajectory of same shape
        """
        if not self._fitted:
            raise RuntimeError("MotionNormalizer not fitted. Call fit() first.")

        device = trajectory.device
        out = trajectory.clone()

        # Convert stats to tensors
        joint_mean = torch.tensor(self.joint_mean, device=device)
        joint_std = torch.tensor(self.joint_std, device=device)
        pelvis_pos_mean = torch.tensor(self.pelvis_pos_mean, device=device)
        pelvis_pos_std = torch.tensor(self.pelvis_pos_std, device=device)

        # Denormalize joints
        out[..., :29] = out[..., :29] * joint_std + joint_mean

        # Denormalize pelvis position
        out[..., 29:32] = out[..., 29:32] * pelvis_pos_std + pelvis_pos_mean

        return out

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Get normalizer state for saving."""
        return {
            'joint_mean': self.joint_mean,
            'joint_std': self.joint_std,
            'pelvis_pos_mean': self.pelvis_pos_mean,
            'pelvis_pos_std': self.pelvis_pos_std,
        }

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]):
        """Load normalizer state."""
        self.joint_mean = state_dict['joint_mean']
        self.joint_std = state_dict['joint_std']
        self.pelvis_pos_mean = state_dict['pelvis_pos_mean']
        self.pelvis_pos_std = state_dict['pelvis_pos_std']
        self._fitted = True


def get_vlm_image_transform(
    image_size: Tuple[int, int] = (224, 224),
) -> T.Compose:
    """
    Get image transform for VLM backbone.

    Only resizes — no ImageNet normalization. The VLM processor
    handles its own preprocessing internally.

    Args:
        image_size: Target (H, W) size

    Returns:
        Composed transforms
    """
    return T.Compose([
        T.Resize(image_size),
    ])
