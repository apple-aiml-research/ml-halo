#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on BeyondMimic whole_body_tracking (Copyright (c) 2024, The Isaac Lab Project Developers, MIT License).

"""Scene motion-tracking env cfgs: the privileged PreRL teachers and the shared building blocks
(scene assets, rewards, terminations, events, privileged observation groups) reused by the
distillation and PostRL envs.

Three PreRL scene cases, one teacher each:

* ``G1FlatEnvCfg_PreRL_CNN``: robot only, flat ground.
* ``G1ObjectEnvCfg_PreRL_CNN``: robot + push box (``push_v2_*`` clips) on flat ground.
* ``G1TerrainEnvCfg_PreRL_CNN``: robot + lab sofa (``labsofa_*`` sit clips).

``G1SceneEnvCfg`` is the multi-asset SuperScene (flat plates + sofa, push box) the distillation and PostRL envs
build on. Scene assets live in ``<repo>/assets/objects`` (``pushbox``, ``labsofa``); the flat plates are generated
in code.
"""

from __future__ import annotations

import glob
import os
import tempfile
from collections.abc import Callable
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from halo_tracker.utils.usd_dir import converted_usd_dir

from . import ASSETS_DIR, mdp, robot_cfg


VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


Cmd_Horizon = 10


@configclass
class MotionTrackingSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        # debug_vis must stay False: the markers of every body of every env crash a multi-GPU sim start.
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=False
    )


# Motion-command settings shared by the robot-only and the scene commands.
_MOTION_COMMAND_KWARGS = {
    "asset_name": "robot",
    "resampling_time_range": (1.0e9, 1.0e9),
    "debug_vis": False,
    "pose_range": {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
    },
    "velocity_range": VELOCITY_RANGE,
    "joint_position_range": (-0.1, 0.1),
    "joint_velocity_range": (-0.01, 0.01),
    "use_adaptive_sampling": True,
    # A resample that lands on a clip's first frame holds the reference there with 5% probability (stand /
    # hold-pose recovery).
    "freeze_at_t0_prob": 0.05,
    "horizon": Cmd_Horizon,
    "flatten_horizon": True,
}


@configclass
class MotionTrackingCommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(**_MOTION_COMMAND_KWARGS, adaptive_uniform_ratio=0.1, adaptive_kernel_size=3)


@configclass
class MotionTrackingActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class MotionTrackingEventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.4),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.02, 0.02),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link", "pelvis"]),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )

    # reset
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.7, 1.5),
            "damping_distribution_params": (0.7, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class MotionTrackingRewardsCfg:
    """Reward terms for the MDP."""

    # Global anchor (pelvis) tracking weighted like the body terms: anchor drift is the dominant
    # termination mode.
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    # Velocity-tracking exp kernels: std 0.5 (lin) / 1.5 (ang) keeps the gradient informative in the
    # typical-error band (wider kernels saturate near reward 1.0).
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.5},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.5},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        # Clipped, so a single physics blow-up step cannot swamp the reward.
        func=mdp.joint_pos_limits_clipped,
        weight=-100.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "clip_max": 10.0,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )

    survival = RewTerm(func=mdp.survival_reward, weight=0.1)

    power_safety = RewTerm(
        func=mdp.power_safety_penalty,
        weight=-10.0,
        # Knees only: they take the braking loads of landings and recoveries.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
                "deadband": 150.0, "normalizer": 500.0},
    )


@configclass
class MotionTrackingTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # ``time_out=True``: a clip's last frame is a truncation (success), not a failure. The motion command
    # normally chains the next clip on that same step, so episodes end on ``time_out`` or a failure term and
    # this fires only when a reset lands on a clip's last frame. The drift/fall terms below stay
    # time_out=False (real failures: value → 0 is correct).
    motion_ends = DoneTerm(
        func=mdp.motion_ends,
        params={"command_name": "motion"},
        time_out=True,
    )
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos,
        params={"command_name": "motion", "threshold": 0.5},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )
    # Safety circuit-breaker for diverged physics: real G1 joints stay below 20 rad/s, so 30 rad/s fires only on
    # blow-ups.
    joint_vel_out_of_limit = DoneTerm(
        func=mdp.joint_vel_out_of_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "threshold": 30.0,
        },
    )


@configclass
class G1FlatEnvCfg(ManagerBasedRLEnvCfg):
    """Base G1 motion-tracking env: flat ground, robot only. The PreRL / distillation / PostRL envs build on it."""

    # Scene settings
    scene: MotionTrackingSceneCfg = MotionTrackingSceneCfg(num_envs=8192, env_spacing=2.5)
    # Basic settings
    observations: CnnObservationsCfg = MISSING  # each concrete PreRL subclass sets this (CnnObservationsCfg)
    actions: MotionTrackingActionsCfg = MotionTrackingActionsCfg()
    commands: MotionTrackingCommandsCfg = MotionTrackingCommandsCfg()
    # MDP settings
    rewards: MotionTrackingRewardsCfg = MotionTrackingRewardsCfg()
    terminations: MotionTrackingTerminationsCfg = MotionTrackingTerminationsCfg()
    events: MotionTrackingEventCfg = MotionTrackingEventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # Replace IsaacLab's remote plane asset with a locally generated flat ground (no network).
        # Runs after physics_material is captured above; keeps it intact.
        apply_local_ground(self)
        self.sim.physx.gpu_max_rigid_patch_count = 2**22
        self.sim.physx.gpu_heap_capacity = 2 ** 27
        self.sim.physx.gpu_temp_buffer_capacity = 2 ** 26
        self.sim.physx.gpu_max_num_partitions = 8
        # Object scenes park the unused push box in a hidden below-floor "graveyard" grid; the extra AABBs
        # inflate the GPU broadphase pair budget, so raise these caps to avoid
        # "increase foundLostPairsCapacity" startup errors.
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**26
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**26
        self.sim.physx.gpu_collision_stack_size = 2**28
        # viewer settings: a wide world-frame view of the env grid
        self.viewer.eye = (12.0, 12.0, 8.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.origin_type = "world"

        # Robot: the system-identified lab G1 (robot_cfg.get_g1_sysid_cfg).
        self.scene.robot = robot_cfg.get_g1_sysid_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = robot_cfg.G1_ACTION_SCALE
        # Anchor body for motion_anchor obs, body-relative tracking, and the anchor reward/termination.
        # Pinned to "pelvis": only link with a physical IMU on the real G1, so the anchor obs is
        # directly measured and deploy runs anchor_mode=pelvis exactly.
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


def apply_play_overrides(cfg, num_envs: int = 64) -> None:
    """Mutate ``cfg`` for evaluation-time motion playback.

    Uniform random start frames (no failure weighting, no freeze-at-t0); zero reset perturbation;
    corruption off on every ObsGroup; DR events nulled; mid-motion terminations replaced by lenient
    Z-only / large-angle checks that still catch falls; 40 s episodes; shrink num_envs for the viewer.
    """
    # 1. Scene
    cfg.scene.num_envs = num_envs

    # 2. Motion command — uniform random start frames. Keeps use_adaptive_sampling=True: the
    # non-adaptive branch of MotionCommand starts every clip at frame 0 (the mocap calibration pose),
    # while the adaptive sampler with adaptive_uniform_ratio=1 picks start frames ~uniformly. The
    # scene motion command samples within each env's compatible clips either way.
    if hasattr(cfg, "commands") and hasattr(cfg.commands, "motion"):
        m = cfg.commands.motion
        m.use_adaptive_sampling = True  # degenerates to uniform-over-frames in play
        m.adaptive_uniform_ratio = 1.0
        m.freeze_at_t0_prob = 0.0
        m.debug_vis = False  # no goal-anchor / body-target markers in the video
        m.pose_range = {}
        m.velocity_range = {}
        m.joint_position_range = (0.0, 0.0)
        m.joint_velocity_range = (0.0, 0.0)

    # 3. Observation noise off — every ObsGroup attribute on cfg.observations
    if hasattr(cfg, "observations") and cfg.observations is not None:
        for group_name in vars(cfg.observations):
            group = getattr(cfg.observations, group_name)
            if hasattr(group, "enable_corruption"):
                group.enable_corruption = False

    # 4. Disable the base DR events (PostRL play also drops its sim-to-real DR; the depth corruption stays on)
    for ev in (
        "physics_material",
        "add_joint_default_pos",
        "base_com",
        "push_robot",
        "robot_joint_stiffness_and_damping",
    ):
        if hasattr(cfg.events, ev):
            setattr(cfg.events, ev, None)

    # 5. Relax the mid-motion terminations so the policy plays the whole clip: position checks become Z-only
    # variants so play survives sideways drift (stage-specific terms such as the distillation envs'
    # anchor_pos_3d stay); the full 3D orientation check still catches real falls.
    for term in ("ee_body_pos", "anchor_pos", "anchor_ori", "object_pos", "object_ori"):
        if hasattr(cfg.terminations, term):
            setattr(cfg.terminations, term, None)
    cfg.terminations.anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.35},
    )
    cfg.terminations.anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 2.0},
    )
    cfg.terminations.ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.4,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )
    # 6. Camera framing for the recorded video, anchored in WORLD coords (no asset tracking) so it
    # stays put. Eye (9,9,6) looking at origin frames the central env-grid cluster (~10 robots at
    # env_spacing 2.5 m; object/terrain runs at 5.0 m show fewer but still usable).
    cfg.viewer.origin_type = "world"
    cfg.viewer.eye = (9.0, 9.0, 6.0)
    cfg.viewer.lookat = (0.0, 0.0, 0.0)

    # 7. Longer play-mode episodes (40 s vs train's 20 s): fewer resets while clips chain.
    cfg.episode_length_s = 40.0


# ── Scene object assets (<repo>/assets/objects) ──────────────────────────────
# Terrains are per-env fixed URDFs (the SuperScene's flat plates are generated in code); the push box is a free
# rigid body. Tasks check their assets in ``__post_init__`` (``_require_assets``).
OBJECT_DIR: str = f"{ASSETS_DIR}/objects"


# Lab sofa (fixed terrain asset) for the ``labsofa_*`` sit clips.
TERRAIN_LABSOFA_URDFS: list[str] = sorted(glob.glob(f"{OBJECT_DIR}/labsofa/labsofa.urdf"))


# Push box (0.46 x 0.50 x 0.56 m, 1.0 kg) for the ``push_v2_*`` clips; its link is ``box_link``.
PUSHBOX_URDF: str = f"{OBJECT_DIR}/pushbox/pushbox.urdf"


def _require_assets(urdfs: list[str], what: str, task: str) -> None:
    """Fail at config time (not deep inside the spawner) when a task's scene assets are missing."""
    if not urdfs:
        raise FileNotFoundError(
            f"{task} needs {what} under {OBJECT_DIR}, but none were found. Add the assets "
            f"(see tracker/training/README.md, 'Scene assets') or train a different scene task."
        )


def _require_pushbox(task: str) -> None:
    _require_assets([PUSHBOX_URDF] if os.path.isfile(PUSHBOX_URDF) else [], "pushbox/pushbox.urdf", task)


_HIDDEN_OBJECT_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False, retain_accelerations=False,
    linear_damping=0.01, angular_damping=0.01,
    max_linear_velocity=1000.0, max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)


_OBJECT_ARTICULATION = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,
    solver_position_iteration_count=8, solver_velocity_iteration_count=4,
)


_OBJECT_DRIVE = sim_utils.UrdfConverterCfg.JointDriveCfg(
    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0))


def _make_terrain_asset(asset_path: str) -> sim_utils.UrdfFileCfg:
    return sim_utils.UrdfFileCfg(
        fix_base=True, replace_cylinders_with_capsules=True,
        asset_path=asset_path, usd_dir=converted_usd_dir(asset_path), activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, retain_accelerations=False,
            linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=0.0, max_angular_velocity=0.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Terrain URDFs declare per-box fixed joints to a `world` root link. Articulation MUST
            # stay enabled for PhysX to enforce them — else every box becomes a free rigid body and
            # scatters on first contact. Terrain is AssetBaseCfg (not RigidObjectCfg), so the
            # articulation-root rejection RigidObject does doesn't apply.
            articulation_enabled=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=8, solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0),
        ),
    )


def _make_flat_plate() -> sim_utils.MeshCuboidCfg:
    """SuperScene flat-ground terrain: a static 4x4 m plate, 2 cm thick, centred at z=0 (top face at z=+0.01).

    It has no ``asset_path``, which ``MotionCommandObjectTerrain`` reads as "no terrain", so its envs play the
    robot-only and push-box clips. Its mesh is the env's ``TerrainMesh`` ray-cast target, so height scans and depth
    over it read flat ground.
    """
    return sim_utils.MeshCuboidCfg(size=(4.0, 4.0, 0.02), collision_props=sim_utils.CollisionPropertiesCfg())


def spawn_floor_plate(prim_path, cfg: "FloorPlateCfg", translation=None, orientation=None, **kwargs):
    """``spawn_mesh_cuboid`` with the cuboid's top face at ``cfg.top_z`` above the given translation."""
    x, y, z = translation if translation is not None else (0.0, 0.0, 0.0)
    return sim_utils.spawn_mesh_cuboid(prim_path, cfg, translation=(x, y, z + cfg.top_z - 0.5 * cfg.size[2]),
                                       orientation=orientation, **kwargs)


@configclass
class FloorPlateCfg(sim_utils.MeshCuboidCfg):
    """A flat-ground plate whose top face sits at ``top_z``; the rest of the cuboid extends below the ground plane
    (static-static pairs make no contacts), so a thick plate keeps its top where a thin one would."""

    func: Callable = spawn_floor_plate
    top_z: float = 0.01


def make_floor_dr_plate(half_size_m: float = 8.0, thickness_m: float = 0.2) -> FloorPlateCfg:
    """Flat-ground plate for the floor DR (``mdp.randomize_floor_compliance``): wide enough that free-motion clips stay
    on it, thick enough that a foot sinking into a compliant floor never crosses its mid-plane, top at +0.01 like
    ``_make_flat_plate``. Invisible: the plates of neighbouring envs overlap (each env collides and ray-casts only its
    own)."""
    return FloorPlateCfg(size=(2.0 * half_size_m, 2.0 * half_size_m, thickness_m), top_z=0.01, visible=False,
                         collision_props=sim_utils.CollisionPropertiesCfg())


def _make_box_object_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False, replace_cylinders_with_capsules=True,
            asset_path=PUSHBOX_URDF,
            usd_dir=converted_usd_dir(PUSHBOX_URDF),
            activate_contact_sensors=True,
            rigid_props=_HIDDEN_OBJECT_PROPS,
            articulation_props=_OBJECT_ARTICULATION,
            joint_drive=_OBJECT_DRIVE,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
    )


@configclass
class MotionTrackingWithTerrainSceneCfg(MotionTrackingSceneCfg):
    """Scene with robot + per-env terrain mesh (default: the lab sofa).

    ``MultiAssetSpawnerCfg`` deterministically assigns env_i to ``assets_cfg[i % N]``;
    ``MotionCommandObjectTerrain`` mirrors this to sample only compatible motion clips.

    Registered as ``AssetBaseCfg`` (not ``RigidObjectCfg``): a terrain URDF is an articulation with
    fixed joints to a ``world`` link. Wrapping it as ``RigidObject`` forces
    ``articulation_enabled=False``, silently disabling those joints so its parts scatter on first
    contact. ``AssetBaseCfg`` + ``fix_base=True`` lets PhysX anchor it to ``env_origin``.
    """

    terrain_object: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TerrainMesh",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[_make_terrain_asset(p) for p in TERRAIN_LABSOFA_URDFS],
            random_choice=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


@configclass
class MotionTrackingObjectSceneCfg(MotionTrackingSceneCfg):
    """Flat ground + the push box, spawned hidden (z=-100); ``MotionCommandObjectTerrain`` activates it
    for the envs whose current clip uses it."""

    box: RigidObjectCfg = _make_box_object_cfg()


# SuperScene (distillation / PostRL) terrain mix: each terrain is repeated by its family weight, and env_i
# gets entry i % N. Flat dominates because most clips are robot-only or push-box; the sofa clips still
# get their own envs. The entries are spread evenly through the list (_spread_evenly), so a run with
# fewer than N envs per GPU still gets every family in proportion instead of only the first ones.
FLAT_PLATE_WEIGHT: int = 300


# Sofa (sit) envs: num_envs * W / (300 + W). W = 57 gives ~16% (~654 of 4096 per GPU), the share the deployed policies
# trained with.
LABSOFA_URDF_WEIGHT: int = 57


def _spread_evenly(families: list[list[str]]) -> list[str]:
    """Merge the per-family entry lists so each family's entries sit at evenly spaced list positions."""
    keyed = [((j + 0.5) / len(entries), f, j, e) for f, entries in enumerate(families) for j, e in enumerate(entries)]
    return [e for *_, e in sorted(keyed)]


TERRAIN_SUPERSCENE_ASSETS: list[sim_utils.SpawnerCfg] = _spread_evenly([
    [_make_flat_plate() for _ in range(FLAT_PLATE_WEIGHT)],
    [_make_terrain_asset(p) for p in TERRAIN_LABSOFA_URDFS * LABSOFA_URDF_WEIGHT],
])


@configclass
class MotionTrackingSuperSceneCfg(MotionTrackingWithTerrainSceneCfg):
    """SuperScene: per-env terrain from ``TERRAIN_SUPERSCENE_ASSETS`` (flat plates + sofa) + the hidden push box."""

    terrain_object: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TerrainMesh",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=TERRAIN_SUPERSCENE_ASSETS,
            random_choice=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    box: RigidObjectCfg = _make_box_object_cfg()


@configclass
class MotionTrackingWithObjectRewardsCfg(MotionTrackingRewardsCfg):
    """Rewards with object tracking terms.

    Hand-proximity-gated (ResMimic / MaskedMimic): object tracking reward multiplied by
    ``exp(-(min_hand_dist / 0.3)^2)``, so the policy is only paid once a hand is near the object —
    avoids chasing the object's reference before contact. Default hand bodies = wrist_yaw_links.
    """

    # ``use_sim_object=True``: hand-proximity gate fires on the REAL (physics) object, not the
    # reference pose — object reward is perception-grounded and robust to an imperfect reference.
    object_pos = RewTerm(
        func=mdp.object_global_position_error_exp_gated,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3, "gate_std": 0.3, "use_sim_object": True},
    )
    object_ori = RewTerm(
        func=mdp.object_global_orientation_error_exp_gated,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4, "gate_std": 0.3, "use_sim_object": True},
    )
    # Contact-seeking bonus + drop penalty, masked to active-object envs.
    object_hand_contact = RewTerm(
        func=mdp.object_hand_contact_reward,
        weight=0.3,
        params={"command_name": "motion", "std": 0.15},
    )
    object_drop = RewTerm(
        func=mdp.object_drop_penalty,
        weight=-1.0,
        # use_sim_object=True: gate the drop penalty on the REAL (physics) object — consistent with
        # the pos/ori/contact rewards above.
        params={"command_name": "motion", "height_margin": 0.15, "gate_std": 0.3, "use_sim_object": True},
    )


@configclass
class MotionTrackingWithObjectTerminationsCfg(MotionTrackingTerminationsCfg):
    """Terminations with object tracking thresholds."""

    object_pos = DoneTerm(
        func=mdp.bad_object_pos,
        params={"command_name": "motion", "threshold": 0.25},
    )
    object_ori = DoneTerm(
        func=mdp.bad_object_ori,
        params={"command_name": "motion", "threshold": 0.8},
    )


@configclass
class MotionTrackingWithObjectEventCfg(MotionTrackingEventCfg):
    """Events with push-box domain randomization (startup friction + mass).

    For envs whose clip doesn't use the box, the Graveyard Dispatcher keeps it at z=-100, so the DR is a
    no-op there. Ranges: an empty cardboard box on an indoor floor (friction 0.6-0.8) and 1.0-1.5 kg.
    """

    object_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box", body_names=".*"),
            "static_friction_range": (0.6, 0.8),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )
    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box", body_names=".*"),
            "mass_distribution_params": (0.0, 0.5),
            "operation": "add",
        },
    )


@configclass
class MotionTrackingSceneCommandsCfg(MotionTrackingCommandsCfg):
    """Command config using MotionCommandObjectTerrain for show/hide multiplexing.

    It samples start frames itself (the adaptive_* settings do not apply) and skips the first calibration frames of
    a clip, so the freeze at t=0 only triggers on clips of at most 11 frames.
    """

    motion = mdp.MotionCommandObjectTerrainCfg(**_MOTION_COMMAND_KWARGS)


Cnn_Obs_History_Len = 30


HEIGHT_RES = 0.1


HEIGHT_SIZE = (1.6, 1.0)


# (n_x, n_y) image of the height map; the grid patterns use ordering="yx" so the rays flatten in this (x-major) order.
HEIGHT_GRID = (int(HEIGHT_SIZE[0] / HEIGHT_RES) + 1, int(HEIGHT_SIZE[1] / HEIGHT_RES) + 1)


@configclass
class CnnPrivilegedStateCfg(ObsGroup):
    """Critic: full privileged state, current frame only.

    Unified schema shared by the three PreRL CNN task variants so downstream distillation/PostRL can
    reuse one CNN critic. Exteroception comes from the heightmap only (``CnnHeightCriticCfg``);
    object state is excluded so the schema doesn't grow with object count, adding multi-object
    scenarios needs no retraining, and teacher obs stays within what the student's sensors recover.
    """

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), history_length=0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), history_length=0)
    proj_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), history_length=0)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0), history_length=0)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-100.0, 100.0), history_length=0)
    last_action = ObsTerm(func=mdp.last_action, history_length=0)
    command = ObsTerm(
        func=mdp.tracking_command,
        params={"command_name": "motion", "command_type": "full",
                "flatten_horizon": True, "horizon": Cmd_Horizon},
        history_length=0,
    )
    motion_anchor_ori_b = ObsTerm(
        func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, history_length=0)
    motion_anchor_pos_b = ObsTerm(
        func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, history_length=0)
    body_pos = ObsTerm(
        func=mdp.robot_body_pos_b, params={"command_name": "motion"}, history_length=0)
    body_ori = ObsTerm(
        func=mdp.robot_body_ori_b, params={"command_name": "motion"}, history_length=0)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CnnPrivilegedTemporalCfg(ObsGroup):
    """Critic: full privileged temporal history.

    Unified schema shared by the three PreRL CNN variants; object terms excluded as in
    ``CnnPrivilegedStateCfg`` (exteroception lives in the heightmap obs group).
    """

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0),
                           history_length=Cnn_Obs_History_Len, flatten_history_dim=False)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0),
                           history_length=Cnn_Obs_History_Len, flatten_history_dim=False)
    proj_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0),
                           history_length=Cnn_Obs_History_Len, flatten_history_dim=False)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0),
                        history_length=Cnn_Obs_History_Len, flatten_history_dim=False)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-100.0, 100.0),
                        history_length=Cnn_Obs_History_Len, flatten_history_dim=False)
    last_action = ObsTerm(func=mdp.last_action,clip=(-100.0, 100.0),
                          history_length=Cnn_Obs_History_Len, flatten_history_dim=False)
    motion_anchor_ori_b = ObsTerm(
        func=mdp.motion_anchor_ori_b, params={"command_name": "motion"},
        history_length=Cnn_Obs_History_Len, flatten_history_dim=False)
    motion_anchor_pos_b = ObsTerm(
        func=mdp.motion_anchor_pos_b, params={"command_name": "motion"},
        history_length=Cnn_Obs_History_Len, flatten_history_dim=False)

    command = ObsTerm(
        func=mdp.tracking_command,
        # "partial" = future trajectory only, no keypoint errors; horizon = Cnn_Obs_History_Len.
        params={"command_name": "motion", "command_type": "partial",
                "flatten_horizon": False, "horizon": Cnn_Obs_History_Len},
        clip=(-100.0, 100.0), history_length=0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.concatenate_dim = -1


@configclass
class CnnHeightCriticCfg(ObsGroup):
    """Critic: clean height-map (no noise, no masking), foot-relative."""

    height = ObsTerm(
        func=mdp.height_scan_2d,
        params={"sensor_cfg": SceneEntityCfg("height_scanner"), "grid_size": HEIGHT_GRID,
                "asset_cfg": SceneEntityCfg("robot")},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CnnObservationsCfg:
    """PreRL-CNN observation container (robot-only / push box / lab sofa).

    In PreRL the actor is privileged (same obs as critic), so only the three critic groups are
    exposed. Both actor and critic in ``MotionTrackingScenePPORunnerCfg_PreRL`` map to ``critic_*`` groups,
    so the PreRL teacher's state dict (``cnns.critic_temporal.*`` / ``cnns.critic_height.*``) drops
    cleanly into ``MultiTeacherDistillation`` with ``strict=True``. The noisy student groups
    (``CnnPolicy*``, depth) live in the distillation env.
    """
    critic_state: CnnPrivilegedStateCfg = CnnPrivilegedStateCfg()
    critic_temporal: CnnPrivilegedTemporalCfg = CnnPrivilegedTemporalCfg()
    critic_height: CnnHeightCriticCfg = CnnHeightCriticCfg()


# Scene object entities with their rigid-body link prims (ray-cast targets) — keep in sync with the
# ``_make_*_object_cfg`` prim paths and ``mdp.commands.OBJECT_KEYS_FROM_FILENAME``.
SCENE_OBJECT_LINK_PRIMS: dict[str, str] = {
    "box": "/World/envs/env_.*/Box/box_link",
}


def scene_objects(scene_cfg) -> list[str]:
    """Object entity names this scene actually spawns (present and not ``None``)."""
    return [k for k in SCENE_OBJECT_LINK_PRIMS if getattr(scene_cfg, k, None) is not None]


def scene_raycast_targets(
    objects: list[str], per_env_terrain: bool = True
) -> list[MultiMeshRayCasterCfg.RaycastTargetCfg]:
    """Ray-cast targets of the height scanner and the depth camera: the shared ground, each env's terrain mesh
    (``per_env_terrain``) and the given object entities (see ``scene_objects``).

    The ground and the terrain are static. Objects are teleported on reset, so they track their transforms; their
    targets are the rigid-body LINK prims (not the /Box wrapper), from which the physics-view pose source walks up to
    the link's RigidBodyAPI.
    """
    targets = [MultiMeshRayCasterCfg.RaycastTargetCfg(
        prim_expr="/World/ground", track_mesh_transforms=False, is_shared=True)]
    if per_env_terrain:
        targets.append(MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="/World/envs/env_.*/TerrainMesh", track_mesh_transforms=False, is_shared=False))
    targets += [
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr=SCENE_OBJECT_LINK_PRIMS[k], track_mesh_transforms=True, is_shared=False
        )
        for k in objects
    ]
    return targets


def sensor_update_period(cfg, control_steps: int = 1) -> float:
    """Sensor ``update_period`` for a refresh every ``control_steps`` policy steps.

    Half a physics step below the access period: IsaacLab accumulates per-env sensor time in float32 with a
    1e-6 tolerance, so a period exactly equal to the access period fails the "outdated" test on some accesses
    after ~16 s of episode time and the policy then sees the previous scan.
    """
    return control_steps * cfg.decimation * cfg.sim.dt - 0.5 * cfg.sim.dt


def height_scanner_cfg() -> RayCasterCfg:
    """Height scanner over the shared flat ground (robot-only scenes)."""
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=HEIGHT_RES, size=list(HEIGHT_SIZE), ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


def height_scanner_cfg_per_env_terrain(
    objects: list[str] | None = None, per_env_terrain: bool = True
) -> MultiMeshRayCasterCfg:
    """Height scanner that resolves each env's own terrain mesh and (optionally) its objects.

    Required when ``env_spacing > 0``: ``RayCaster`` loads a single mesh prim (no ``{ENV_REGEX_NS}``
    expansion), so envs at non-zero origins would read env_0's terrain. ``MultiMeshRayCaster`` binds
    one mesh per env via a regex. The shared ``/World/ground`` is always scanned too: a terrain mesh
    covers only a few metres around its env origin, and free-motion clips travel well beyond it, where
    a terrain-only map would be empty (the robot-only teacher trained on the ground everywhere).
    ``per_env_terrain=False`` scans only the ground (flat object scenes).

    ``objects`` ADDS the given object entities (see ``scene_objects``) so the height map perceives
    object top surfaces. Objects are graveyarded at z=-100 for non-object motions, far below the
    surface a downward ray (origin +20 m) hits first, so they never corrupt the map when unused.
    """
    cfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=HEIGHT_RES, size=list(HEIGHT_SIZE), ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=scene_raycast_targets(objects or [], per_env_terrain=per_env_terrain),
    )
    if objects:
        # Pose the objects from per-rank physics views (multi-GPU-safe, see sensors/mgpu_ray_caster.py).
        from .sensors.mgpu_ray_caster import MultiMeshRayCasterMGPU
        cfg.class_type = MultiMeshRayCasterMGPU
    return cfg


def _apply_prerl_cnn_overrides(cfg) -> None:
    """Narrower joint-offset DR, no actuator-gain DR, ee termination at 0.3 m, lighter action-rate penalty.

    Applied to the PreRL teachers and to the SuperScene base, so distillation and PostRL inherit it (PostRL re-adds
    actuator DR in its sim-to-real bundle)."""
    cfg.events.add_joint_default_pos.params["pos_distribution_params"] = (-0.01, 0.01)
    cfg.events.robot_joint_stiffness_and_damping = None
    cfg.terminations.ee_body_pos.params["threshold"] = 0.3
    cfg.rewards.action_rate_l2.weight = -0.03


@configclass
class G1FlatEnvCfg_PreRL_CNN(G1FlatEnvCfg):
    """Robot-only teacher on flat ground.

    Scene/rewards/terminations/events/commands inherited from ``G1FlatEnvCfg``; only obs + height
    scanner change. Actor obs = critic obs (privileged) so the PreRL actor state dict is directly
    reusable as a distillation teacher.
    """

    observations: CnnObservationsCfg = CnnObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner = height_scanner_cfg()
        self.scene.height_scanner.update_period = sensor_update_period(self)
        _apply_prerl_cnn_overrides(self)


@configclass
class G1FlatEnvCfg_PreRL_CNN_Play(G1FlatEnvCfg_PreRL_CNN):
    """Play-mode robot-only teacher: few envs, base DR off, relaxed terminations."""

    def __post_init__(self):
        super().__post_init__()
        apply_play_overrides(self)


def _apply_multi_asset_post_init(cfg, *, per_env_terrain: bool) -> None:
    """Shared ``__post_init__`` body for the multi-asset scene envs (object / terrain, and the
    distillation / PostRL SuperScene).

    All use ``MultiAssetSpawnerCfg`` (so physics replication MUST be off). The height map scans the
    shared ground, each env's own terrain (``per_env_terrain``) and every spawned object, so the
    privileged critic perceives all of them — the same schema the distillation / PostRL critics see.
    """
    cfg.scene.replicate_physics = False
    objects = scene_objects(cfg.scene)
    cfg.scene.height_scanner = height_scanner_cfg_per_env_terrain(objects, per_env_terrain=per_env_terrain)
    cfg.scene.height_scanner.update_period = sensor_update_period(cfg)
    print(f"[SceneEnv] height scanner: per-env terrain={per_env_terrain}, objects={objects}")


@configclass
class G1ObjectEnvCfg_PreRL_CNN(G1FlatEnvCfg):
    """Robot + push box teacher on flat ground (``push_v2_*`` clips)."""

    observations: CnnObservationsCfg = CnnObservationsCfg()
    # Wider env spacing (4 m) keeps the boxes of neighbouring envs visually apart.
    scene: MotionTrackingObjectSceneCfg = MotionTrackingObjectSceneCfg(num_envs=8192, env_spacing=4.0)
    commands: MotionTrackingSceneCommandsCfg = MotionTrackingSceneCommandsCfg()
    rewards: MotionTrackingWithObjectRewardsCfg = MotionTrackingWithObjectRewardsCfg()
    terminations: MotionTrackingWithObjectTerminationsCfg = MotionTrackingWithObjectTerminationsCfg()
    events: MotionTrackingWithObjectEventCfg = MotionTrackingWithObjectEventCfg()

    def __post_init__(self):
        super().__post_init__()
        _require_pushbox(type(self).__name__)
        _apply_multi_asset_post_init(self, per_env_terrain=False)
        _apply_prerl_cnn_overrides(self)


@configclass
class G1TerrainEnvCfg_PreRL_CNN(G1FlatEnvCfg):
    """Robot + lab sofa teacher (``labsofa_*`` sit clips): every env spawns the sofa."""

    observations: CnnObservationsCfg = CnnObservationsCfg()
    # Wider env spacing (5 m) keeps neighbouring sofas apart.
    scene: MotionTrackingWithTerrainSceneCfg = MotionTrackingWithTerrainSceneCfg(
        num_envs=8192, env_spacing=5.0)
    commands: MotionTrackingSceneCommandsCfg = MotionTrackingSceneCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        _require_assets(TERRAIN_LABSOFA_URDFS, "labsofa/labsofa.urdf", type(self).__name__)
        _apply_multi_asset_post_init(self, per_env_terrain=True)
        _apply_prerl_cnn_overrides(self)


@configclass
class G1SceneEnvCfg(G1FlatEnvCfg):
    """SuperScene base for distillation / PostRL: per-env flat plate or sofa + the push box, object
    rewards / terminations / events, object-aware height map, ``MotionCommandObjectTerrain`` dispatch.
    Not registered as a task on its own."""

    observations: CnnObservationsCfg = CnnObservationsCfg()
    # Wider env spacing (5 m) for the sofa and box footprints.
    scene: MotionTrackingSuperSceneCfg = MotionTrackingSuperSceneCfg(num_envs=4096, env_spacing=5.0)
    commands: MotionTrackingSceneCommandsCfg = MotionTrackingSceneCommandsCfg()
    rewards: MotionTrackingWithObjectRewardsCfg = MotionTrackingWithObjectRewardsCfg()
    terminations: MotionTrackingWithObjectTerminationsCfg = MotionTrackingWithObjectTerminationsCfg()
    events: MotionTrackingWithObjectEventCfg = MotionTrackingWithObjectEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self._check_scene_assets()
        _apply_multi_asset_post_init(self, per_env_terrain=True)
        _apply_prerl_cnn_overrides(self)

    def _check_scene_assets(self) -> None:
        _require_assets(TERRAIN_LABSOFA_URDFS, "labsofa/labsofa.urdf", type(self).__name__)
        _require_pushbox(type(self).__name__)


@configclass
class G1ObjectEnvCfg_PreRL_CNN_Play(G1ObjectEnvCfg_PreRL_CNN):
    """Play-mode object teacher: few envs, base DR off, relaxed terminations."""

    def __post_init__(self):
        super().__post_init__()
        apply_play_overrides(self)


@configclass
class G1TerrainEnvCfg_PreRL_CNN_Play(G1TerrainEnvCfg_PreRL_CNN):
    """Play-mode terrain teacher: few envs, base DR off, relaxed terminations."""

    def __post_init__(self):
        super().__post_init__()
        apply_play_overrides(self)


_GROUND_HALF_EXTENT = 250.0


_GROUND_CELL = 10.0


def _generate_flat_ground_usd(path: str) -> bool:
    """Author a minimal flat ground plane (z=0) at ``path`` using ``pxr``. Returns True on success.

    A tessellated flat grid (±``_GROUND_HALF_EXTENT`` m, ``_GROUND_CELL``-m cells) with a static
    triangle-mesh collider and 1.0/1.0 friction baked in; nothing is fetched from the network. The bounded
    triangle size keeps the PhysX GPU broadphase stable on multi-GPU runs (a single km-scale triangle makes the
    device fail with ERROR_DEVICE_LOST). The friction material is best-effort; without it PhysX uses the sim
    default 1.0/1.0 that ``G1FlatEnvCfg`` sets.
    """
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except Exception as exc:  # pragma: no cover - only meaningful inside the Isaac container
        print(f"[LOCAL-GROUND] pxr unavailable ({exc}); leaving remote ground plane in place.")
        return False

    L = _GROUND_HALF_EXTENT
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/GroundPlane")
    stage.SetDefaultPrim(root.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/GroundPlane/CollisionMesh")
    # Tessellated flat grid in z=0: (n+1)^2 vertices, 2*n^2 CCW (+Z) triangles.
    n = max(1, int(round((2.0 * L) / _GROUND_CELL)))   # cells per side
    step = (2.0 * L) / n
    nv = n + 1                                          # vertices per side
    points = [(-L + i * step, -L + j * step, 0.0) for j in range(nv) for i in range(nv)]
    counts, indices = [], []
    for j in range(n):
        for i in range(n):
            v00 = j * nv + i; v10 = j * nv + (i + 1)
            v01 = (j + 1) * nv + i; v11 = (j + 1) * nv + (i + 1)
            counts += [3, 3]
            indices += [v00, v10, v11, v00, v11, v01]   # two CCW (+Z) triangles per cell
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateNormalsAttr([(0.0, 0.0, 1.0)] * len(points))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr([(-L, -L, 0.0), (L, L, 0.0)])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    # Static triangle-mesh collider (no RigidBodyAPI ⇒ static). approximation="none" = exact
    # triangles, valid for a static collider.
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr().Set(UsdPhysics.Tokens.none)

    # Best-effort: bake the 1.0/1.0 friction material so ground friction is explicit rather than relying
    # on the sim default.
    try:
        from pxr import UsdShade

        material = UsdShade.Material.Define(stage, "/GroundPlane/PhysicsMaterial")
        UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        mat_api = UsdPhysics.MaterialAPI(material.GetPrim())
        mat_api.CreateStaticFrictionAttr().Set(1.0)
        mat_api.CreateDynamicFrictionAttr().Set(1.0)
        mat_api.CreateRestitutionAttr().Set(0.0)
        binding_api = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
    except Exception as exc:  # pragma: no cover
        print(f"[LOCAL-GROUND] friction-material bake skipped ({exc}); using sim default material.")

    stage.Export(path)
    return True


def apply_local_ground(cfg) -> None:
    """If the scene uses IsaacLab's remote ``terrain_type="plane"`` ground, swap it for a locally
    generated flat-ground USD (no network). No-op otherwise.

    Safe from any ``__post_init__``: self-targets ``plane`` scenes, leaves generator/usd terrains
    alone, and any failure logs + keeps the original remote terrain.
    """
    terrain = getattr(getattr(cfg, "scene", None), "terrain", None)
    if terrain is None or getattr(terrain, "terrain_type", None) != "plane":
        return  # generator / usd / none terrains don't fetch the remote Grid asset

    # One tiny USD per process (race-free across torchrun ranks; read once at scene creation).
    path = os.path.join(tempfile.gettempdir(), f"halo_flat_ground.{os.getpid()}.usda")
    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
        if not _generate_flat_ground_usd(path):
            return  # generation failed: keep the remote ground

    terrain.terrain_type = "usd"
    terrain.usd_path = path
    terrain.visual_material = None
    print(f"[LOCAL-GROUND] terrain → local USD {path}")
