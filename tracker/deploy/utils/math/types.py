#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Core data types shared by the G1 deployment (MuJoCo and real robot).
"""

from typing import List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import yaml

# Joint count of the G1 the deployment is built for (the C++ deployment is compiled for the same N).
G1_NUM_JOINTS = 29


def _vector(raw: dict, key: str, n: int) -> np.ndarray:
    """A required config vector: exactly ``n`` numbers, no default."""
    value = raw.get(key)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"config: missing or non-list field {key!r} (need {n} values)")
    if len(value) != n:
        raise ValueError(f"config: field {key!r} has {len(value)} entries, expected {n}")
    return np.array(value, dtype=np.float32)


@dataclass
class RobotState:
    """Robot state as read from a backend (simulation or real robot)."""
    joint_positions: np.ndarray      # (num_joints,) in deployment order
    joint_velocities: np.ndarray     # (num_joints,) in deployment order
    imu_quaternion: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], dtype=np.float32))  # wxyz
    imu_angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    odom_position: Optional[np.ndarray] = None       # (3,) xyz position from odom
    odom_quaternion: Optional[np.ndarray] = None     # (4,) wxyz quaternion from odom


@dataclass
class MotorCommand:
    """Per-joint motor command (PD targets, gains and feedforward torque)."""
    positions: np.ndarray           # Target positions (deployment order)
    velocities: np.ndarray          # Target velocities (deployment order)
    kp: np.ndarray                  # Position gains
    kd: np.ndarray                  # Velocity/damping gains
    torques: np.ndarray             # Feedforward torques
    modes: np.ndarray               # Per-joint control mode (0 = position; the G1 uses position only)


class ControllerConfig:
    """Controller configuration loaded from a YAML file (see ``g1/configs/``).

    Known keys are parsed into typed attributes below; every other top-level key is
    passed through as an attribute unchanged (e.g. ``observation``, ``depth_camera``).
    """

    def __init__(self, config_path: str, root_dir: Optional[str] = None):
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file (relative paths resolve against the CWD).
            root_dir: Value substituted for ``{ROOT_DIR}`` in ``xml_path``
                (default: the repository root, which holds ``assets/``).
        """
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        if root_dir is None:
            root_dir = str(Path(__file__).resolve().parents[4])  # repository root

        xml_path = raw.get("xml_path", "")
        self.xml_path = xml_path.replace("{ROOT_DIR}", root_dir)

        self.robot_name = raw.get("robot_name", "unknown")

        # Joint configuration
        self.deployment_joint_order = raw.get("deployment_joint_order", [])
        self.training_joint_order = raw.get("training_joint_order", [])
        self.num_joints = raw.get("num_joints", len(self.deployment_joint_order))
        if self.num_joints != G1_NUM_JOINTS:
            raise ValueError(f"config: num_joints={self.num_joints}, but the G1 deployment needs {G1_NUM_JOINTS}")

        # Default pose, gains, action scale and standing pose (deployment order). Required, one value per joint.
        n = self.num_joints
        self.default_angles = _vector(raw, "default_angles", n)
        self.joint_kps = _vector(raw, "joint_kps", n)
        self.joint_kds = _vector(raw, "joint_kds", n)
        self.action_scale = _vector(raw, "action_scale", n)
        self.max_torque = _vector(raw, "max_torque", n)
        self.standing_joint_pos = _vector(raw, "standing_joint_pos", n)
        self.standing_base_pos = _vector(raw, "standing_base_pos", 3)
        self.standing_base_quat_wxyz = _vector(raw, "standing_base_quat_wxyz", 4)
        # SIM ONLY: the sim plant's identified drive. The MuJoCo motor realises joint_kps x sim_kp_scale and
        # joint_kds x sim_kd_scale (deployment order), the gains training simulates; the real robot ignores both.
        self.sim_kp_scale = _vector(raw, "sim_kp_scale", n) if "sim_kp_scale" in raw else None
        self.sim_kd_scale = _vector(raw, "sim_kd_scale", n) if "sim_kd_scale" in raw else None
        for key in ("sim_kp_scale", "sim_kd_scale"):
            scale = getattr(self, key)
            if scale is not None and not (np.all(np.isfinite(scale)) and np.all(scale > 0)):
                raise ValueError(f"config: {key} must be finite and > 0")

        # Control timing
        self.simulation_dt = raw.get("simulation_dt", 0.005)
        self.decimation = raw.get("control_decimation", 4)
        self.control_dt = raw.get("control_dt", self.simulation_dt * self.decimation)

        # Policy I/O clipping and ONNX Runtime provider
        self.clip_action = raw.get("clip_action", 100.0)
        self.clip_observation = raw.get("clip_observation", 100.0)
        self.use_gpu = bool(raw.get("use_gpu", True))

        # The parsed YAML as loaded, for helpers that consume a whole block (e.g.
        # variant_support.depth_camera_rig reads `depth_camera:`). Set before the
        # pass-through below so a stray `raw:` key cannot shadow it.
        self.raw = raw

        # Pass through any extra keys as attributes
        for key, value in raw.items():
            if not hasattr(self, key):
                setattr(self, key, value)


def build_index_map(origin_joint_order: List[str], target_joint_order: List[str]) -> np.ndarray:
    """Build index mapping for fast joint reordering.

    Args:
        origin_joint_order: Source joint order
        target_joint_order: Target joint order

    Returns:
        numpy array of indices such that target[i] = origin[index_map[i]]
    """
    index_map = np.zeros(len(target_joint_order), dtype=np.int32)
    for i, target_joint in enumerate(target_joint_order):
        try:
            index_map[i] = origin_joint_order.index(target_joint)
        except ValueError:
            raise ValueError(f"Target joint '{target_joint}' not found in origin_joint_order.")
    return index_map