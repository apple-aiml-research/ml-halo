#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Evaluation metrics for motion trajectory prediction.
"""

import math
import torch
from typing import Dict


def mean_per_joint_position_error(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Mean per-joint error (MPJPE) over joint angles.

    Args:
        pred: [B, T, 29] predicted joint angles (rad)
        target: [B, T, 29] target joint angles (rad)

    Returns:
        Scalar mean absolute joint-angle error (rad)
    """
    return torch.abs(pred - target).mean()


def average_displacement_error(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Average Displacement Error (ADE).

    Measures the average L2 distance between predicted and target
    positions across all timesteps.

    Args:
        pred: [B, T, D] predicted positions (e.g., pelvis position)
        target: [B, T, D] target positions

    Returns:
        Scalar ADE value
    """
    return torch.norm(pred - target, p=2, dim=-1).mean()


def final_displacement_error(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Final Displacement Error (FDE).

    Measures the L2 distance at the final timestep.

    Args:
        pred: [B, T, D] predicted positions
        target: [B, T, D] target positions

    Returns:
        Scalar FDE value (averaged over batch)
    """
    # Get final timestep
    pred_final = pred[:, -1]  # [B, D]
    target_final = target[:, -1]  # [B, D]

    # Compute L2 distance
    displacement = torch.norm(pred_final - target_final, p=2, dim=-1)  # [B]
    return displacement.mean()


def quaternion_angular_error(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute angular error between predicted and target quaternions.

    Args:
        pred: [B, T, 4] predicted quaternions (w, x, y, z)
        target: [B, T, 4] target quaternions

    Returns:
        Scalar angular error in radians
    """
    # Normalize quaternions
    pred_norm = pred / (torch.norm(pred, dim=-1, keepdim=True) + 1e-8)
    target_norm = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-8)

    # Compute dot product (handles q = -q equivalence)
    dot = torch.abs((pred_norm * target_norm).sum(dim=-1))  # [B, T]
    dot = torch.clamp(dot, -1.0, 1.0)

    # Angular error = 2 * arccos(|dot|)
    return (2.0 * torch.acos(dot)).mean()


def velocity_error(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute velocity error between predicted and target trajectories.

    Args:
        pred: [B, T, D] predicted trajectory
        target: [B, T, D] target trajectory

    Returns:
        Scalar velocity error
    """
    # Compute velocities
    pred_vel = pred[:, 1:] - pred[:, :-1]  # [B, T-1, D]
    target_vel = target[:, 1:] - target[:, :-1]

    return torch.norm(pred_vel - target_vel, p=2, dim=-1).mean()


def heading_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    min_displacement: float = 0.10,
) -> Dict[str, float]:
    """Direction-of-travel metrics: does the plan head the right WAY?

    The other metrics are dominated by the 29 joints and by the magnitude of
    pelvis motion, which can be predicted well without looking at the scene
    (e.g. from the elapsed time or the task alone). Turn DIRECTION has no such
    shortcut, and it is the typical failure mode: policies that walk a sensible
    distance but do not steer toward the target.

    Uses channels 29:31 of the [B, T, 36] layout — pelvis displacement relative
    to frame 0 and de-yawed into the robot's heading frame (data/traj_extract.py)
    — so 'y' is signed lateral offset: +left, -right. Requires PHYSICAL units;
    per-axis normalisation would rescale x and y differently and corrupt the
    angle. model.predict() denormalises, so callers there are fine.

    Samples whose ground-truth endpoint displacement is below min_displacement
    are skipped: the robot barely moved, so its heading is ill-defined. This
    matters on datasets where trajectory_length exceeds the episode length, since
    those windows are padded with a static tail.

    Returns:
        heading_err_deg   mean |angle(pred_end) - angle(gt_end)|, degrees
        turn_sign_acc     fraction whose lateral sign matches (went the right way)
        lateral_err_m     mean |pred_end_y - gt_end_y|, metres
        heading_n_valid   fraction of samples that cleared min_displacement
    """
    pred_end = pred[:, -1, 29:31]
    gt_end = target[:, -1, 29:31]

    gt_norm = torch.linalg.norm(gt_end, dim=-1)
    valid = gt_norm >= min_displacement
    frac = valid.float().mean().item()
    if int(valid.sum()) == 0:
        return {"heading_err_deg": float("nan"), "turn_sign_acc": float("nan"),
                "lateral_err_m": float("nan"), "heading_n_valid": frac}

    pe, ge = pred_end[valid], gt_end[valid]
    ang = torch.atan2(pe[:, 1], pe[:, 0]) - torch.atan2(ge[:, 1], ge[:, 0])
    # wrap to (-pi, pi] so a 359-degree error does not read as 359
    ang = torch.atan2(torch.sin(ang), torch.cos(ang))

    # Sign agreement, ignoring GT endpoints that are essentially straight ahead
    # (|y| tiny) where "left vs right" carries no information.
    lat = ge[:, 1].abs() >= 0.05
    sign_acc = (float("nan") if int(lat.sum()) == 0 else
                (torch.sign(pe[lat, 1]) == torch.sign(ge[lat, 1])).float().mean().item())

    return {
        "heading_err_deg": torch.rad2deg(ang.abs()).mean().item(),
        "turn_sign_acc": sign_acc,
        "lateral_err_m": (pe[:, 1] - ge[:, 1]).abs().mean().item(),
        "heading_n_valid": frac,
    }


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for motion prediction.

    Args:
        pred: [B, T, 36] predicted trajectory
        target: [B, T, 36] target trajectory

    Returns:
        Dict with all metric values
    """
    # Split trajectory components
    pred_joints = pred[..., :29]
    target_joints = target[..., :29]

    pred_pelvis_pos = pred[..., 29:32]
    target_pelvis_pos = target[..., 29:32]

    pred_pelvis_quat = pred[..., 32:36]
    target_pelvis_quat = target[..., 32:36]

    metrics = {}

    # Joint metrics
    metrics["mpjpe"] = mean_per_joint_position_error(
        pred_joints, target_joints
    ).item()

    metrics["joint_ade"] = average_displacement_error(
        pred_joints, target_joints
    ).item()

    # Pelvis position metrics
    metrics["pelvis_pos_ade"] = average_displacement_error(
        pred_pelvis_pos, target_pelvis_pos
    ).item()

    metrics["pelvis_pos_fde"] = final_displacement_error(
        pred_pelvis_pos, target_pelvis_pos
    ).item()

    # Pelvis orientation metrics
    metrics["pelvis_quat_error"] = quaternion_angular_error(
        pred_pelvis_quat, target_pelvis_quat
    ).item()

    # Velocity metrics
    metrics["joint_vel_error"] = velocity_error(
        pred_joints, target_joints
    ).item()

    metrics["pelvis_vel_error"] = velocity_error(
        pred_pelvis_pos, target_pelvis_pos
    ).item()

    # Overall trajectory metrics
    metrics["trajectory_ade"] = average_displacement_error(
        pred, target
    ).item()

    metrics["trajectory_fde"] = final_displacement_error(
        pred, target
    ).item()

    # Direction of travel — the only component without a free shortcut.
    metrics.update(heading_metrics(pred, target))

    return metrics


class MetricTracker:
    """
    Track and accumulate metrics over batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self._metrics = {}
        self._counts = {}

    def update(self, metrics: Dict[str, float], count: int = 1):
        """
        Update metrics with new values.

        Args:
            metrics: Dict of metric name -> value
            count: Number of samples in this batch
        """
        for name, value in metrics.items():
            # Skip non-finite values instead of accumulating them: a single NaN
            # would otherwise poison this metric's running sum for the whole run.
            # heading_metrics returns NaN by design when a batch has no sample
            # with enough displacement to define a heading, which happens
            # whenever trajectory_length exceeds the episode length and the
            # window is padded with a static tail. Counts are per-metric, so the
            # average is simply taken over the samples that were valid.
            if not math.isfinite(value):
                continue
            if name not in self._metrics:
                self._metrics[name] = 0.0
                self._counts[name] = 0
            self._metrics[name] += value * count
            self._counts[name] += count

    def compute(self) -> Dict[str, float]:
        """
        Compute average metrics.

        Returns:
            Dict of averaged metrics
        """
        return {
            name: self._metrics[name] / max(self._counts[name], 1)
            for name in self._metrics
        }
