#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Multi-teacher distillation env cfgs for G1 scene motion tracking.

Distills the privileged PreRL teachers into ONE student via on-policy multi-teacher DAgger. Builds on the
SuperScene env ``G1SceneEnvCfg`` (per-env flat plate or lab sofa + the push box), so the student sees every
motion class: ``MotionCommandObjectTerrain`` dispatches per-env compatible motions and ``MultiTeacherDistillation``
routes each env to the teacher of its current motion via ``TeacherRegistry``.

Two student variants:

  ``G1SceneDistillEnvCfg_Proprio``
      Blind student — noisy proprio (``policy_state``) + motion-reference history (``policy_temporal``).

  ``G1SceneDistillEnvCfg_ProprioDepth``
      Adds one torso-mounted D435 depth image (``student_depth``, 48×84, clip+log1p, depth noise model).

The teacher always consumes the full PreRL privileged obs schema (``critic_state`` + ``critic_temporal`` +
``critic_height``) so the PreRL teacher actor state_dicts load strict.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import math
import os

from . import mdp
from .motion_tracking_scene_env_cfg import (
    Cnn_Obs_History_Len,
    CnnHeightCriticCfg,
    CnnPrivilegedStateCfg,
    CnnPrivilegedTemporalCfg,
    G1SceneEnvCfg,
    apply_play_overrides,
    scene_objects,
    scene_raycast_targets,
    sensor_update_period,
)
from .sensors import RPLDepthCameraCfg


# ═══════════════════════════════════════════════════════════════════════════
# Depth sensor (D435i-class, torso-mounted at torso_link/d435i_link)
# ═══════════════════════════════════════════════════════════════════════════

# (height, width) of the rendered depth image and of the student's CNN input (no resampling); the deployment crops and
# resizes the D435 848×480 stream to it.
DEPTH_HW: tuple[int, int] = (48, 84)

# Encoded z-depth range [m]: the near clip also starts the self-occlusion rays past the camera housing, the far clip is
# what misses, dropouts and invalid stereo pixels map to. Part of the deploy contract: the deploy configs'
# ``student_depth.clip`` (and the C++ depth config's clip_min_m / clip_max_m) must equal it.
DEPTH_CLIP: tuple[float, float] = (0.2, 2.5)
# Camera mount on torso_link: the measured D435i extrinsic, as the deployed depth policies trained and the deploy sims
# render.
DEPTH_CAMERA_POS: tuple[float, float, float] = (0.054058, 0.033836, 0.433928)

# D435 pinhole intrinsics at the render resolution: HFOV = 2·atan(47.78 / (2·24.0)) ≈ 89.74°, VFOV ≈ 59.26°.
_DEPTH_FOCAL_MM = 24.0
_DEPTH_H_APERTURE_MM = 47.78
_DEPTH_V_APERTURE_MM = _DEPTH_H_APERTURE_MM * DEPTH_HW[0] / DEPTH_HW[1]
# Longest ray (image corner) whose z-depth is DEPTH_CLIP[1], plus 1 % margin.
_DEPTH_RAY_MAX_DISTANCE = 1.01 * DEPTH_CLIP[1] * math.sqrt(
    1.0 + (_DEPTH_H_APERTURE_MM / (2 * _DEPTH_FOCAL_MM)) ** 2 + (_DEPTH_V_APERTURE_MM / (2 * _DEPTH_FOCAL_MM)) ** 2
)


def _depth_camera_cfg(objects: list[str] | None = None) -> RPLDepthCameraCfg:
    """Depth camera for the student's perceptive head (D435i, torso-mounted): ``RPLDepthCamera`` (RPL Alg. 1).

    Ray-casts the ground, each env's terrain mesh and the spawned objects (``objects``, see ``scene_objects``; the
    ray caster raises on a target prim that does not exist), plus every robot body link (the cfg's default
    ``body_prim_expr``) for self-occlusion, as on the real robot.

    Mount: parented to ``torso_link`` (IsaacLab merges the fixed d435i links into torso) with the URDF
    torso→d435i transform pre-composed into ``OffsetCfg`` (~50.75° downtilt). Intrinsics (HFOV 89.74°,
    VFOV 59.26° at 84×48) match the D435. The deploy configs (``deploy/g1/configs/g1_v2_depth.yaml``,
    ``deploy_cpp/configs/g1_v2_depth.yaml``) mirror this mount, resolution and clip.
    """
    return RPLDepthCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        update_period=0.0,  # set in __post_init__ to N * decimation * sim.dt
        pattern_cfg=PinholeCameraPatternCfg(
            focal_length=_DEPTH_FOCAL_MM,
            horizontal_aperture=_DEPTH_H_APERTURE_MM,
            height=DEPTH_HW[0],
            width=DEPTH_HW[1],
        ),
        # Z-depth ("distance_to_image_plane"), what the D435 reports. Missed rays get ``max_distance`` (finite).
        data_types=["distance_to_image_plane"],
        # max_distance bounds the RAY LENGTH, but the clip is on z-depth (what the D435 and the deploy encode
        # use): a corner pixel's ray is 1/cos(θ) ≈ 1.52× its z. _DEPTH_RAY_MAX_DISTANCE lets every pixel with
        # z ≤ DEPTH_CLIP[1] hit; anything farther (or a miss) still maps to DEPTH_CLIP[1] after
        # mdp.depth_image's clip. Do NOT go below it.
        max_distance=_DEPTH_RAY_MAX_DISTANCE,
        depth_clipping_behavior="max",
        # Mount offset — the quat encodes the URDF's ~50.75° downtilt at the d435i_link frame.
        offset=RPLDepthCameraCfg.OffsetCfg(
            pos=DEPTH_CAMERA_POS,
            rot=(0.903525, -0.004587, 0.428507, -0.001531),
            convention="world",  # parent is body-frame (+X fwd, +Z up)
        ),
        # The height scanner's targets: ground, each env's terrain, the objects.
        mesh_prim_paths=scene_raycast_targets(objects or []),
        debug_vis=False,
        # Body rays start at the camera's near range, past the D435 housing and torso interior at the lens.
        body_near_clip=float(DEPTH_CLIP[0]),
    )


def _bind_rpl_depth_robot(env, env_ids, sensor_name: str = "depth_camera", robot_name: str = "robot"):
    """Startup event: hand the depth camera the robot articulation, whose per-link poses drive self-occlusion."""
    sensors = env.scene.sensors
    sensor = sensors[sensor_name] if sensor_name in sensors else None
    if sensor is not None and hasattr(sensor, "set_robot_asset"):
        sensor.set_robot_asset(env.scene[robot_name])


def _bind_rpl_depth_objects(env, env_ids, sensor_name: str = "depth_camera",
                            object_tokens: tuple = (("Box", "box"),)):
    """Startup event: hand the depth camera the scene's dynamic objects (``RPLDepthCamera.set_dynamic_object_assets``).

    Maps each ``track_mesh_transforms`` target whose prim_expr contains ``/<token>/`` to the matching RigidObject.
    No-op without a depth camera.
    """
    sensors = env.scene.sensors
    sensor = sensors[sensor_name] if sensor_name in sensors else None
    if sensor is None or not hasattr(sensor, "set_dynamic_object_assets"):
        return
    rigids = env.scene.rigid_objects
    mapping = {}
    tracked = [t for t in sensor.cfg.mesh_prim_paths if getattr(t, "track_mesh_transforms", False)]
    for target_cfg in tracked:
        for token, scene_key in object_tokens:
            if f"/{token}/" in target_cfg.prim_expr and scene_key in rigids:
                mapping[target_cfg.prim_expr] = rigids[scene_key]
                break
    if tracked and not mapping and int(os.environ.get("WORLD_SIZE", "1") or "1") > 1:
        # Without a mapping the objects would be posed through usdrt, which fails under multi-GPU.
        raise RuntimeError(
            "[RPLDepthCamera] no dynamic-object physics-view mapping resolved under multi-GPU "
            "(WORLD_SIZE>1): check the scene spawns the push box the RaycastTargetCfg "
            "prim_exprs expect."
        )
    sensor.set_dynamic_object_assets(mapping)
    print(f"[RPLDepthCamera] dynamic objects posed from physics views: "
          f"{list(mapping) if mapping else 'none (usdrt pose source, single-GPU only)'}")


# ═══════════════════════════════════════════════════════════════════════════
# Student-side observation groups
# ═══════════════════════════════════════════════════════════════════════════
#
# The PreRL envs expose only the privileged ``critic_*`` groups (PreRL actor == critic). The deployable
# STUDENT reads the non-privileged, noisy analog below. Its layout is the deploy contract (``obs`` 434-D,
# ``policy_temporal`` 30×166) mirrored by the ``observation:`` block of the deploy configs.

_REF_CMD_NOISE = 0.01  # motion-reference command obs-noise
_REF_ORI_NOISE = 0.05  # motion-anchor orientation obs-noise

# The actor/student sees a SHORTER motion-command lookahead than the critic (less look-ahead "cheating"
# that hurts sim-to-real). Both ≤ MotionCommandCfg.horizon (=Cmd_Horizon=10).
Actor_Cmd_Horizon = 5


@configclass
class CnnPolicyStateCfg(ObsGroup):
    """Student (actor): noisy partial state, current frame only.

    The deployable-actor analog of ``CnnPrivilegedStateCfg``: drops base_lin_vel / body_pos / body_ori
    (not onboard-observable), adds Unoise, uses the shorter ``Actor_Cmd_Horizon`` command lookahead.
    """

    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),
                           clip=(-100.0, 100.0), history_length=0)
    proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),
                           clip=(-100.0, 100.0), history_length=0)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
                        clip=(-100.0, 100.0), history_length=0)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5),
                        clip=(-100.0, 100.0), history_length=0)
    last_action = ObsTerm(func=mdp.last_action, noise=GaussianNoise(mean=0.0, std=0.01),
                          clip=(-100.0, 100.0), history_length=0)
    command = ObsTerm(
        func=mdp.tracking_command,
        # Actor sees only Actor_Cmd_Horizon=5 future frames (vs critic's 10).
        params={"command_name": "motion", "command_type": "partial",
                "flatten_horizon": True, "horizon": Actor_Cmd_Horizon},
        noise=Unoise(n_min=-_REF_CMD_NOISE, n_max=_REF_CMD_NOISE), clip=(-100.0, 100.0), history_length=0,
    )
    motion_anchor_ori_b = ObsTerm(
        func=mdp.motion_anchor_ori_b, params={"command_name": "motion"},
        noise=Unoise(n_min=-_REF_ORI_NOISE, n_max=_REF_ORI_NOISE), clip=(-100.0, 100.0), history_length=0,
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class CnnPolicyTemporalCfg(ObsGroup):
    """Student (actor): noisy partial temporal history (30-frame Conv1d stream).

    ``Cnn_Obs_History_Len`` (=30) matches the teacher's ``CnnPrivilegedTemporalCfg`` length.
    ``flatten_history_dim=False`` keeps the (B, T, F) layout the temporal Conv1d consumes.
    """

    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),
                           clip=(-100.0, 100.0),
                           history_length=Cnn_Obs_History_Len, flatten_history_dim=False)
    proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),
                           clip=(-100.0, 100.0),
                           history_length=Cnn_Obs_History_Len, flatten_history_dim=False)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
                        clip=(-100.0, 100.0),
                        history_length=Cnn_Obs_History_Len, flatten_history_dim=False)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5),
                        clip=(-100.0, 100.0),
                        history_length=Cnn_Obs_History_Len, flatten_history_dim=False)
    last_action = ObsTerm(func=mdp.last_action, noise=GaussianNoise(mean=0.0, std=0.01),
                          clip=(-100.0, 100.0),
                          history_length=Cnn_Obs_History_Len, flatten_history_dim=False)
    motion_anchor_ori_b = ObsTerm(
        func=mdp.motion_anchor_ori_b, params={"command_name": "motion"},
        noise=Unoise(n_min=-_REF_ORI_NOISE, n_max=_REF_ORI_NOISE),
        history_length=Cnn_Obs_History_Len, flatten_history_dim=False)

    command = ObsTerm(
        func=mdp.tracking_command,
        # partial trajectory over the full 30-frame window (matches the critic temporal layout).
        params={"command_name": "motion", "command_type": "partial",
                "flatten_horizon": False, "horizon": Cnn_Obs_History_Len},
        noise=Unoise(n_min=-_REF_CMD_NOISE, n_max=_REF_CMD_NOISE), clip=(-100.0, 100.0), history_length=0,
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.concatenate_dim = -1


@configclass
class CnnStudentDepthCfg(ObsGroup):
    """Student: single depth image from the torso-mounted D435.

    ``mdp.depth_image`` maps misses to the far clip, applies the depth noise model and log1p, and returns
    ``(B, 1, 48, 84)`` for the student's depth CNN.
    ``delay_range_steps`` keeps its default: 1-3 processed frames of per-episode latency.
    """

    depth = ObsTerm(
        func=mdp.depth_image,
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera"),
            "out_hw": DEPTH_HW,
            "clip": DEPTH_CLIP,
            "log_scale": True,
            # Sensor noise of a stereo depth model (the deployment runs a stereo network on the D435 IR pair):
            # N(0, 5 mm) per pixel and 5 % of pixels dropped to the far clip. Camera extrinsics and latency are
            # randomized separately (randomize_depth_camera_extrinsics, delay_range_steps).
            "noise_std_m": 0.005,
            "dropout_prob": 0.05,
            # Indoor synthetic clutter ("virtual walls"): 0–2 rects inside the clip range (beyond the far clip they
            # would be invisible), size 8–25% of image dims, resampled at reset. Occlude-only (torch.minimum).
            "clutter_cfg": {
                "n_range": (0, 2),
                "depth_range_m": (min(1.5, 0.6 * DEPTH_CLIP[1]), 0.95 * DEPTH_CLIP[1]),
                "size_frac_range": (0.08, 0.25),
            },
            # Random Side Masking: with prob rsm_mask_prob occlude up to rsm_max_cols peripheral columns
            # (random side) → far-plane, so the student relies on the central region. At W=84, 12 cols
            # ≈ 14% per side.
            "rsm_mask_prob": 0.1,
            "rsm_max_cols": 12,
            # 25 Hz depth: a new frame every 2nd control step (deploy: observation.student_depth.obs_decim).
            "obs_decim": 2,
        },
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


# ═══════════════════════════════════════════════════════════════════════════
# Observation containers
# ═══════════════════════════════════════════════════════════════════════════

@configclass
class CnnSceneDistillObservationsCfg_Proprio:
    """Proprio-only student + privileged teacher.

    Groups (matching the runner ``obs_groups`` routing):
      student → policy_state, policy_temporal
      teacher → critic_state, critic_temporal, critic_height  (identical to the PreRL teacher's obs
                schema → PreRL actor state_dict loads strict).
    """

    policy_state: CnnPolicyStateCfg = CnnPolicyStateCfg()
    policy_temporal: CnnPolicyTemporalCfg = CnnPolicyTemporalCfg()
    critic_state: CnnPrivilegedStateCfg = CnnPrivilegedStateCfg()
    critic_temporal: CnnPrivilegedTemporalCfg = CnnPrivilegedTemporalCfg()
    critic_height: CnnHeightCriticCfg = CnnHeightCriticCfg()


@configclass
class CnnSceneDistillObservationsCfg_ProprioDepth(CnnSceneDistillObservationsCfg_Proprio):
    """Proprio + depth-image student + privileged teacher."""

    student_depth: CnnStudentDepthCfg = CnnStudentDepthCfg()


# ═══════════════════════════════════════════════════════════════════════════
# Environment configs
# ═══════════════════════════════════════════════════════════════════════════

@configclass
class G1SceneDistillEnvCfg_Proprio(G1SceneEnvCfg):
    """Multi-teacher distillation env — proprio-only (blind) student.

    Builds on ``G1SceneEnvCfg`` (SuperScene: per-env flat plate or sofa + the push box, per-env motion dispatch
    via ``MotionCommandObjectTerrain``, object rewards/terminations/events, object-aware height scanner). The
    observation container adds the student groups, and terminations are relaxed for BC (the student
    drifts mid-training; whole-body 3D reference terminations would kill most of each clip's timeline).
    """

    observations: CnnSceneDistillObservationsCfg_Proprio = CnnSceneDistillObservationsCfg_Proprio()

    def __post_init__(self):
        super().__post_init__()

        # ── Distillation-specific termination relaxation ──────────────────────────
        # Mid-training the student drifts horizontally even when joint trajectories are roughly right, so
        # whole-body 3D reference terminations would end most episodes early and most of each clip would
        # never enter the training signal. Keep Z-drift + lenient-orientation terminations (same pattern as
        # apply_play_overrides); drop object_pos / object_ori entirely (PostRL inherits this set and only
        # retunes the anchor_ori / ee_body_pos thresholds).
        for term in ("anchor_pos", "anchor_ori", "ee_body_pos", "object_pos", "object_ori"):
            if hasattr(self.terminations, term):
                setattr(self.terminations, term, None)

        # Split position termination: TIGHT Z-only @ 0.3 m (catch falls fast) + LOOSE 3D @ 0.8 m (catch
        # teleport blow-ups, tolerate recoverable XY drift).
        self.terminations.anchor_pos = DoneTerm(
            func=mdp.bad_anchor_pos_z_only, params={"command_name": "motion", "threshold": 0.3})
        self.terminations.anchor_pos_3d = DoneTerm(
            func=mdp.bad_anchor_pos, params={"command_name": "motion", "threshold": 0.8})
        self.terminations.anchor_ori = DoneTerm(
            func=mdp.bad_anchor_ori,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 1.0},
        )
        # Feet/wrist Z-drift cap relaxed for BC so aggressive phases (kicks, jumps) don't reset
        # mid-trajectory; still catches a fallen robot.
        self.terminations.ee_body_pos = DoneTerm(
            func=mdp.bad_motion_body_pos_z_only,
            params={
                "command_name": "motion",
                "threshold": 0.5,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            },
        )

        # Joint-velocity safety cutoff at 60 rad/s (PreRL: 30) so transient spikes in aggressive phases do not end
        # episodes; physics blow-ups still exceed it.
        self.terminations.joint_vel_out_of_limit.params["threshold"] = 60.0

        # BC is supervised (reward gradients never reach the student), so the power-safety penalty only
        # clutters logs. PostRL re-enables it.
        self.rewards.power_safety.weight = 0.0


@configclass
class G1SceneDistillEnvCfg_ProprioDepth(G1SceneDistillEnvCfg_Proprio):
    """Multi-teacher distillation env — proprio + depth student."""

    observations: CnnSceneDistillObservationsCfg_ProprioDepth = CnnSceneDistillObservationsCfg_ProprioDepth()

    def __post_init__(self):
        super().__post_init__()
        objects = scene_objects(self.scene)
        self.scene.depth_camera = _depth_camera_cfg(objects)
        # Depth at 25 Hz (every 2nd policy step; ``CnnStudentDepthCfg`` holds the frame in between), matching the
        # deploy configs' depth obs_decim 2. The policy loop still runs at 50 Hz on proprio.
        self.scene.depth_camera.update_period = sensor_update_period(self, control_steps=2)

        # Extrinsic DR — per-episode ±2.5 cm / ±2.5° perturbation around the calibrated mount.
        self.events.randomize_depth_camera = EventTerm(
            func=mdp.randomize_depth_camera_extrinsics,
            mode="reset",
            params={"sensor_cfg": SceneEntityCfg("depth_camera")},
        )
        # RPL binds (multi-GPU-safe pose sources): startup events, once per env at init, per-rank.
        self.events.bind_rpl_depth_robot = EventTerm(
            func=_bind_rpl_depth_robot,
            mode="startup",
            params={"sensor_name": "depth_camera", "robot_name": "robot"},
        )
        self.events.bind_rpl_depth_objects = EventTerm(
            func=_bind_rpl_depth_objects,
            mode="startup",
            params={"sensor_name": "depth_camera"},
        )


# ═══════════════════════════════════════════════════════════════════════════
# Play variants — evaluation (few envs, base DR off, relaxed terminations)
# ═══════════════════════════════════════════════════════════════════════════

@configclass
class G1SceneDistillEnvCfg_Proprio_Play(G1SceneDistillEnvCfg_Proprio):
    """Play-mode proprio-only (blind) distillation env."""

    def __post_init__(self):
        super().__post_init__()
        apply_play_overrides(self)


@configclass
class G1SceneDistillEnvCfg_ProprioDepth_Play(G1SceneDistillEnvCfg_ProprioDepth):
    """Play-mode proprio + depth distillation env."""

    def __post_init__(self):
        super().__post_init__()
        apply_play_overrides(self)
