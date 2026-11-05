#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a HALO tracker checkpoint and export its policy (JIT + ONNX) to ``<checkpoint_dir>/exported/``."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Play and export an RL agent trained with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the rollout.")
parser.add_argument(
    "--video_length", type=int, default=None, help="Length of the recorded video in steps (default: 30 s)."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--robot_id", type=int, default=None, help="Robot to follow with the camera (1-based).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--motion_data_folder",
    type=str,
    nargs="+",
    default=None,
    help="One or more folders of NPZ motion files, globbed by the motion command.",
)
parser.add_argument(
    "--motion_data",
    nargs="+",
    type=str,
    default=None,
    help="Explicit .npz motion files (ignored when --motion_data_folder is given).",
)
parser.add_argument(
    "--teacher_registry_yaml",
    type=str,
    default=None,
    help="Teacher registry YAML. Required for the Distillation and PostRL tasks: the algorithm rebuilds the "
         "per-motion teachers / critics from it before the checkpoint loads.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--export_only", action="store_true", default=False,
    help="Export the policy (JIT + ONNX) and exit without stepping the sim (no rendering needed).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

cli_args.check_rsl_rl_version()

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import halo_tracker.tasks  # noqa: F401

DEFAULT_VIDEO_SECONDS = 30.0


def _follow_robot(env, robot_index: int, step: int, total_steps: int) -> None:
    """Video camera: close behind the robot for the first third of the video, then a slow pull-back to a wide shot."""
    robot = env.unwrapped.scene.articulations["robot"]
    robot_index = min(robot_index, robot.data.root_pos_w.shape[0] - 1)
    robot_pos = robot.data.root_pos_w[robot_index].cpu().numpy()
    start = robot_pos + np.array([1.0, 1.0, 1.0])
    first_phase = total_steps // 3
    progress = 0.0 if step <= first_phase else min((step - first_phase) / max(total_steps - first_phase, 1), 1.0)
    eye = start + progress * (np.array([1.0, -45.0, 6.0]) - start)
    controller = env.unwrapped.viewport_camera_controller
    controller.set_view_env_index(robot_index)
    controller.update_view_location(eye, robot_pos)


def main():
    """Play with RSL-RL agent."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    cli_args.apply_motion_data_args(env_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        import carb

        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/pathtracing/spp", 64)
        settings.set("/rtx/post/denoiser/enabled", False)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    dt = env.unwrapped.step_dt
    video_steps = int(DEFAULT_VIDEO_SECONDS / dt) if args_cli.video_length is None else args_cli.video_length
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": video_steps,
            "disable_logger": True,
        }
        print(f"[INFO] Recording a {video_steps}-step video (~{video_steps * dt:.1f} s).")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # The registry-based algorithms rebuild their per-motion teachers / critics before the checkpoint loads.
    cli_args.apply_teacher_registry_arg(agent_cfg, args_cli)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, map_location=agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Export the deployable policy (the actor, or the distilled student). The observation normalizer is baked in
    # (an identity here: all HALO models train with obs_normalization=False).
    export_model_dir = os.path.join(log_dir, "exported")
    os.makedirs(export_model_dir, exist_ok=True)
    runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
    runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    print(f"[INFO] Exported policy (JIT + ONNX) to: {export_model_dir}")

    if args_cli.export_only:
        env.close()
        return

    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        if args_cli.video and args_cli.robot_id is not None:
            _follow_robot(env, args_cli.robot_id - 1, timestep, video_steps)
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == video_steps:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
