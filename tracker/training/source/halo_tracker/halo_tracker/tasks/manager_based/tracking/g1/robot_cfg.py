#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on BeyondMimic whole_body_tracking (Copyright (c) 2024, The Isaac Lab Project Developers, MIT License).

"""Unitree G1 (29 DoF) articulation configs for the HALO motion tracker.

* ``G1_29DOF_CFG``: implicit-PD base (gains from rotor inertia, see below). Single source of truth for the nominal
  gains / armature / limits / init pose.
* ``get_g1_sysid_cfg()``: the training robot, used by every stage and by the motion-data converter. A
  system-identified lab G1 (Mac mini backpack + head cameras) with an explicit DelayedPD actuator, per-joint
  identified gains / armature / torque limits, and the colliders of its MJCF.

PD gains follow the rotor-inertia rule: every joint gets the same natural frequency (10 Hz) and damping
ratio (2.0), ``kp = I * w^2`` and ``kd = 2 * zeta * I * w``. The action scale is ``0.25 * tau_max / kp``.
"""

import copy
import dataclasses
import functools
import json
import re

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from halo_tracker.utils.usd_dir import converted_usd_dir

from . import ASSETS_DIR

G1_ASSET_DIR = f"{ASSETS_DIR}/robots/g1"

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{G1_ASSET_DIR}/urdf/g1_29dof_rev_1_0.urdf",
        usd_dir=converted_usd_dir(f"{G1_ASSET_DIR}/urdf/g1_29dof_rev_1_0.urdf"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
)

"""Unitree G1 (29 DoF), implicit PD with rotor-inertia gains."""

G1_ACTION_SCALE = {}
for a in G1_29DOF_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            G1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]


# =============================================================================
# Training robot: a G1 carrying a Mac mini backpack + head cameras (37.56 kg), identified on hardware.
#
# Built from the base ``G1_29DOF_CFG`` with two changes:
#   1. Actuator: ImplicitActuatorCfg -> IsaacLab's DelayedPDActuatorCfg. In-model PD with a motor-torque
#      clip plus a randomized command latency of ``min_delay..max_delay`` PHYSICS steps (5-25 ms at
#      sim.dt = 5 ms). Gains, armature and torque limits are then overwritten per joint from the sysid json.
#   2. Body: the sysid URDF, whose colliders match the robot's MJCF. Self-collision is off: with it on, the
#      hands catch on the body while pushing (the push-box teacher learns far slower) and the sim is slower.
# The action scale stays the nominal ``G1_ACTION_SCALE``, so checkpoints and the deploy configs agree.
# Built lazily: the json is only read when an env is created.
# =============================================================================

# Command latency window, in PHYSICS steps (5-25 ms at sim.dt = 5 ms), covering the real robot's command transport
# delay; the robot always has >=1 step of delay.
DELAYED_PD_MIN_DELAY: int = 1
DELAYED_PD_MAX_DELAY: int = 5

G1_SYSID_STEM = "g1_29dof_macmini_sysid"


def _to_delayed_pd_cfg(
    impl_cfg: ImplicitActuatorCfg,
    *,
    min_delay: int = DELAYED_PD_MIN_DELAY,
    max_delay: int = DELAYED_PD_MAX_DELAY,
) -> DelayedPDActuatorCfg:
    """Convert an ``ImplicitActuatorCfg`` to an explicit, frictionless ``DelayedPDActuatorCfg``.

    Copies every field of ``impl_cfg`` (joint expr, gains, armature, limits) and re-homes the torque limit for an
    explicit actuator: the implicit group's ``effort_limit_sim`` is the motor limit, so it moves to ``effort_limit``
    (the in-model clip) and ``effort_limit_sim`` is cleared to None so IsaacLab auto-resolves it to 1e9 (no double
    clip in PhysX). ``velocity_limit_sim`` is kept as the solver cap.
    """
    fields = {f.name: getattr(impl_cfg, f.name) for f in dataclasses.fields(impl_cfg) if f.name != "class_type"}
    fields["effort_limit"] = fields.get("effort_limit_sim")
    fields["effort_limit_sim"] = None
    fields.update(min_delay=min_delay, max_delay=max_delay, friction=0.0, dynamic_friction=0.0,
                  viscous_friction=0.0)
    return DelayedPDActuatorCfg(**fields)


@functools.cache
def get_g1_sysid_cfg() -> ArticulationCfg:
    """The training robot (see the block comment above)."""
    with open(f"{G1_ASSET_DIR}/urdf/{G1_SYSID_STEM}_isaaclab.json") as f:
        table = json.load(f)["joints"]
    urdf = f"{G1_ASSET_DIR}/urdf/{G1_SYSID_STEM}.urdf"
    cfg = G1_29DOF_CFG.replace(
        actuators={name: _to_delayed_pd_cfg(act) for name, act in G1_29DOF_CFG.actuators.items()},
        spawn=G1_29DOF_CFG.spawn.replace(
            asset_path=urdf,
            usd_dir=converted_usd_dir(urdf),
            collider_type="convex_hull",            # convex hulls of the URDF's collision meshes; no GPU VHACD
            replace_cylinders_with_capsules=False,  # keep the flat cylinder foot rods (accurate ground contact)
            articulation_props=G1_29DOF_CFG.spawn.articulation_props.replace(enabled_self_collisions=False),
        ),
    )
    cfg = copy.deepcopy(cfg)
    cfg.actuators["legs"].velocity_limit_sim[".*_hip_pitch_joint"] = 20.0
    for wrist in (".*_wrist_pitch_joint", ".*_wrist_yaw_joint"):
        cfg.actuators["arms"].velocity_limit_sim[wrist] = 27.0
    claimed = set()
    for act in cfg.actuators.values():
        members = [n for n in table if any(re.fullmatch(e, f"{n}_joint") for e in act.joint_names_expr)]
        claimed.update(members)

        def per(key, members=members):
            return {f"{n}_joint": float(table[n][key]) for n in members}

        act.stiffness, act.damping = per("stiffness"), per("damping")
        act.armature, act.effort_limit = per("armature"), per("effort")
    missing = set(table) - claimed
    if missing:
        raise ValueError(f"sysid robot: joints not covered by any actuator group: {sorted(missing)}")
    return cfg


# DelayedPD whose command lag is re-drawn U[min_delay, max_delay] with probability ``resample_prob`` per env
# every physics step (the real robot's latency jitters per tick). Used by the PostRL sim-to-real DR.
class JitterDelayedPDActuator(DelayedPDActuator):
    cfg: "JitterDelayedPDActuatorCfg"

    def compute(self, control_action, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        p = self.cfg.resample_prob
        if p > 0.0 and self.cfg.max_delay > self.cfg.min_delay:
            ids = self._ALL_INDICES[torch.rand(self._num_envs, device=self._device) < p]
            if len(ids):
                lags = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (len(ids),),
                                     dtype=torch.int, device=self._device)
                for buf in (self.positions_delay_buffer, self.velocities_delay_buffer, self.efforts_delay_buffer):
                    buf.set_time_lag(lags, ids)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class JitterDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    class_type: type = JitterDelayedPDActuator
    resample_prob: float = 0.0
