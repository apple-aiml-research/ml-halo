#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Scene PostRL env cfgs — multi-critic RL fine-tune of the distilled student on the SuperScene.

Extends the distillation env (``G1SceneDistillEnvCfg_*``: same SuperScene, motion set, obs groups and terminations)
for PPO: a wider reset-state DR (joint positions, root velocities, restitution), anchor-orientation and end-effector Z
termination thresholds of 1.2 and 0.375 m, ``power_safety`` re-enabled (distillation zeroes it, BC has no reward
gradient), plus the sim-to-real DR around the identified robot (``_apply_sysid_dr``) and the floor DR (rigid and
compliant floors, ``_apply_floor_dr``) and the push DR (``_apply_push_dr``). Variants:
``G1ScenePostRLEnvCfg_Proprio`` (blind) / ``_ProprioDepth`` (+ torso depth).
"""

from __future__ import annotations

import dataclasses

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .motion_tracking_scene_distill_cfg import (
    G1SceneDistillEnvCfg_Proprio,
    G1SceneDistillEnvCfg_ProprioDepth,
)
from .motion_tracking_scene_env_cfg import apply_play_overrides, make_floor_dr_plate
from .robot_cfg import JitterDelayedPDActuatorCfg


# Reset root-velocity range, 1.5x the one of PreRL / distillation.
_AGGRESSIVE_INIT_VEL_RANGE = {
    "x": (-0.75, 0.75),
    "y": (-0.75, 0.75),
    "z": (-0.30, 0.30),
    "roll": (-0.78, 0.78),
    "pitch": (-0.78, 0.78),
    "yaw": (-1.17, 1.17),
}


def _apply_postrl_overrides(cfg) -> None:
    """Apply the PostRL DR, termination and reward settings to the distillation env (in place)."""
    # Wider reset-state DR (the periodic push_robot keeps its range).
    cfg.commands.motion.joint_position_range = (-0.15, 0.15)
    cfg.commands.motion.velocity_range = dict(_AGGRESSIVE_INIT_VEL_RANGE)
    if getattr(cfg.events, "physics_material", None) is not None:
        cfg.events.physics_material.params["restitution_range"] = (0.0, 0.5)
    # Termination thresholds (guarded; ee_body_pos is the distillation env's Z-only term).
    if getattr(cfg.terminations, "anchor_ori", None) is not None:
        cfg.terminations.anchor_ori.params["threshold"] = 1.2
    if getattr(cfg.terminations, "ee_body_pos", None) is not None:
        cfg.terminations.ee_body_pos.params["threshold"] = 0.375
    # Reward re-tune for PPO: modest action-rate + re-enable the knee-torque safety penalty (BC had zeroed it).
    cfg.rewards.action_rate_l2.weight = -0.05
    cfg.rewards.power_safety.weight = -10.0
    _apply_sysid_dr(cfg)
    _apply_floor_dr(cfg)
    _apply_push_dr(cfg)


_LIMB_LINKS = [".*_hip_.*", ".*_knee_link", ".*_ankle_.*", ".*_shoulder_.*", ".*_elbow_link", ".*_wrist_.*"]
_PARALLEL_JOINTS = r"((left|right)_ankle_(pitch|roll)|waist_(roll|pitch))_joint"
_SYSID_DR_EVENTS = ("armature_dr", "parallel_armature_dr", "effort_limit_dr", "link_mass_dr", "link_com_dr",
                    "payload_mass_dr", "payload_inertia_dr", "joint_friction_dr", "floor_compliance")
# Measured joint stiction (N·m) per motor family, from the lab robot's system identification; each joint draws
# U[0, ceiling] per reset.
_JOINT_FRICTION_CEILING = {"hip_knee": 1.3, "hip_yaw": 0.89, "ankle_waist_rp": 2.46, "arms": 0.37, "wrist_py": 0.15}
_JOINT_FRICTION_GROUPS = {
    "hip_knee": r"(left|right)_(hip_pitch|hip_roll|knee)_joint",
    "hip_yaw": r"((left|right)_hip_yaw|waist_yaw)_joint",
    "ankle_waist_rp": r"((left|right)_ankle_(pitch|roll)|waist_(roll|pitch))_joint",
    "arms": r"(left|right)_(shoulder_(pitch|roll|yaw)|elbow|wrist_roll)_joint",
    "wrist_py": r"(left|right)_wrist_(pitch|yaw)_joint",
}
# Bodies the reference-gated contact penalty watches.
_GATED_CONTACT_LINKS = ["pelvis", "torso_link", "left_hip_roll_link", "right_hip_roll_link", "left_knee_link",
                        "right_knee_link", "left_shoulder_roll_link", "right_shoulder_roll_link", "left_elbow_link",
                        "right_elbow_link"]


def _apply_sysid_dr(cfg) -> None:
    """Sim-to-real DR around the system-identified robot (``robot_cfg.get_g1_sysid_cfg``).

    Actuator / mass / payload / joint-stiction DR around the identified values, 15-40 ms command latency re-drawn about
    every 20 ms, the stance-slip penalty, and the reference-gated contact penalty in place of the plain one. Latency
    and re-draw rate are given in ms and converted to this env's sim.dt.
    """
    cfg.events.robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.85, 1.15), "damping_distribution_params": (0.75, 1.3),
                "operation": "scale", "distribution": "log_uniform"})
    cfg.events.armature_dr = EventTerm(
        func=mdp.randomize_joint_parameters, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[rf"^(?!{_PARALLEL_JOINTS}$).*$"]),
                "armature_distribution_params": (0.9, 1.2), "operation": "scale"})
    cfg.events.parallel_armature_dr = EventTerm(
        func=mdp.randomize_joint_parameters, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[_PARALLEL_JOINTS]),
                "armature_distribution_params": (0.67, 1.33), "operation": "scale"})
    cfg.events.effort_limit_dr = EventTerm(
        func=mdp.randomize_actuator_effort_scale, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": (0.85, 1.05), "not_reached_min": 0.8,
                "not_reached_joints": (r".*_hip_(pitch|roll)_joint", r".*_ankle_roll_joint", r"waist_pitch_joint",
                                       r".*_wrist_.*_joint")})
    cfg.events.link_mass_dr = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=_LIMB_LINKS),
                "mass_distribution_params": (0.9, 1.1), "operation": "scale", "recompute_inertia": True})
    cfg.events.link_com_dr = EventTerm(
        func=mdp.randomize_rigid_body_com_per_body, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=_LIMB_LINKS),
                "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)}})
    # Payload (backpack / battery): torso mass +-0.5 kg (inertia rescaled), then torso inertia x0.6-1.6 with the mass
    # kept, and the torso / pelvis CoM +-5 cm on every axis.
    cfg.events.payload_mass_dr = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
                "mass_distribution_params": (-0.5, 0.5), "operation": "add", "recompute_inertia": True})
    cfg.events.payload_inertia_dr = EventTerm(
        func=mdp.randomize_rigid_body_inertia_only, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
                "inertia_distribution_params": (0.6, 1.6)})
    if getattr(cfg.events, "base_com", None) is not None:
        cfg.events.base_com.params["com_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}
    cfg.events.joint_friction_dr = EventTerm(
        func=mdp.randomize_joint_friction, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "friction_ceiling_by_group": _JOINT_FRICTION_CEILING,
                "group_joint_regex": _JOINT_FRICTION_GROUPS})
    step_ms = cfg.sim.dt * 1000.0
    cfg.scene.robot.actuators = {
        name: JitterDelayedPDActuatorCfg(
            **{f.name: getattr(act, f.name) for f in dataclasses.fields(act) if f.name != "class_type"}
            | {"min_delay": round(15 / step_ms), "max_delay": round(40 / step_ms),
               "resample_prob": min(1.0, step_ms / 20.0)})
        for name, act in cfg.scene.robot.actuators.items()}
    feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
    cfg.rewards.feet_stance_slip = RewTerm(
        func=mdp.feet_stance_slip_motion_track, weight=-0.2,
        params={"command_name": "motion", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=feet),
                "asset_cfg": SceneEntityCfg("robot", body_names=feet)})
    # Sit / kneel references touch the floor on purpose: charge only contacts the reference does not make.
    cfg.rewards.undesired_contacts = None
    cfg.rewards.undesired_contacts_ref_gated = RewTerm(
        func=mdp.undesired_contacts_ref_gated, weight=-0.05,
        params={"command_name": "motion",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_GATED_CONTACT_LINKS)})


def _apply_push_dr(cfg) -> None:
    """Wide box DR (friction 0.3-0.9, mass -0.4..+5 kg) and push clips sampled with probability 0.42 (3.5 : 2.5)."""
    if getattr(cfg.events, "object_material", None) is not None:
        cfg.events.object_material.params["static_friction_range"] = (0.3, 0.9)
        cfg.events.object_material.params["dynamic_friction_range"] = (0.3, 0.9)
    if getattr(cfg.events, "object_mass", None) is not None:
        cfg.events.object_mass.params["mass_distribution_params"] = (-0.4, 5.0)
    cfg.commands.motion.object_sampling_share = 2.5 / (2.5 + 3.5)


def _apply_floor_dr(cfg) -> None:
    """Rigid and compliant (carpet) floors: every flat-plate env stands on a wide, thick plate
    (``make_floor_dr_plate``), and a third of them get a compliant-contact floor, stiffness 1.3-6.5 kN/m and damping
    ratio 0.3-1.0 (``mdp.randomize_floor_compliance``). The sofa envs keep the rigid ground."""
    spawn = cfg.scene.terrain_object.spawn
    assets = [make_floor_dr_plate() if isinstance(a, sim_utils.MeshCuboidCfg) else a for a in spawn.assets_cfg]
    cfg.scene.terrain_object = cfg.scene.terrain_object.replace(spawn=spawn.replace(assets_cfg=assets))
    cfg.events.floor_compliance = EventTerm(
        func=mdp.randomize_floor_compliance, mode="prestartup",
        params={"fraction": 1.0 / 3.0, "stiffness_range": (1300.0, 6500.0), "damping_ratio_range": (0.3, 1.0)})


def _apply_postrl_play(cfg) -> None:
    """Play / eval: the shared play overrides plus the sim-to-real DR terms they do not cover (the gain DR is
    already one of theirs). The jitter actuator keeps one lag per reset, drawn from [min_delay, max_delay]."""
    apply_play_overrides(cfg)
    # Evaluate with the distillation box and uniform clip sampling.
    if getattr(cfg.events, "object_material", None) is not None:
        cfg.events.object_material.params["static_friction_range"] = (0.6, 0.8)
        cfg.events.object_material.params["dynamic_friction_range"] = (0.6, 0.8)
    if getattr(cfg.events, "object_mass", None) is not None:
        cfg.events.object_mass.params["mass_distribution_params"] = (0.0, 0.5)
    cfg.commands.motion.object_sampling_share = None
    for ev in _SYSID_DR_EVENTS:
        if hasattr(cfg.events, ev):
            setattr(cfg.events, ev, None)
    for act in cfg.scene.robot.actuators.values():
        if isinstance(act, JitterDelayedPDActuatorCfg):
            act.resample_prob = 0.0


@configclass
class G1ScenePostRLEnvCfg_Proprio(G1SceneDistillEnvCfg_Proprio):
    """Scene PostRL — blind (proprio-only) actor on the SuperScene."""

    def __post_init__(self):
        super().__post_init__()
        _apply_postrl_overrides(self)


@configclass
class G1ScenePostRLEnvCfg_ProprioDepth(G1SceneDistillEnvCfg_ProprioDepth):
    """Scene PostRL — proprio + torso-depth actor on the SuperScene."""

    def __post_init__(self):
        super().__post_init__()
        _apply_postrl_overrides(self)


@configclass
class G1ScenePostRLEnvCfg_Proprio_Play(G1ScenePostRLEnvCfg_Proprio):
    """Play-mode — few envs, relaxed terminations, robot DR off (one fixed command lag per reset)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_postrl_play(self)


@configclass
class G1ScenePostRLEnvCfg_ProprioDepth_Play(G1ScenePostRLEnvCfg_ProprioDepth):
    """Play-mode — few envs, relaxed terminations, robot DR off (one fixed command lag per reset)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_postrl_play(self)
