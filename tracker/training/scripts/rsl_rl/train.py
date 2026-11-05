#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a HALO tracker stage (PreRL / Distillation / PostRL) with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--motion_data_folder",
    type=str,
    nargs="+",
    default=None,
    help="One or more folders of NPZ motion files, globbed by the motion command (sets "
         "``env_cfg.commands.motion.motion_folder``).",
)
parser.add_argument(
    "--motion_data",
    nargs="+",
    type=str,
    default=None,
    help="Explicit .npz motion files (ignored when --motion_data_folder is given).",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--load_checkpoint_path",
    type=str,
    default=None,
    help="Path to a checkpoint to warm-start from (takes precedence over --resume / --load_run / --checkpoint; the "
         "iteration count restarts at 0). PostRL: the distillation checkpoint (its student initializes the actor).",
)
parser.add_argument(
    "--teacher_registry_yaml",
    type=str,
    default=None,
    help="Teacher registry YAML (task_type -> PreRL teacher checkpoint + motion routing rules). Required by the "
         "Distillation and PostRL tasks.",
)
parser.add_argument(
    "--resume_optim",
    type=lambda s: str(s).lower() in ("true", "1", "yes", "y", "t"),
    default=True,
    help="On a --load_checkpoint_path warm start, also load the optimizer state (default) or start a fresh "
         "optimizer (False). A PostRL warm start from a distillation checkpoint always starts a fresh optimizer.",
)
parser.add_argument(
    "--verbose_isaac", action="store_true", default=False,
    help="Keep the default IsaacLab / Omniverse / URDF-importer warnings (muted by default).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if not args_cli.verbose_isaac:
    cli_args.mute_isaaclab_warnings()

"""Check the RSL-RL version."""

cli_args.check_rsl_rl_version()

"""Rest everything follows."""

import gymnasium as gym
import pickle
import torch
import yaml
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import halo_tracker.tasks  # noqa: F401
from halo_tracker.rsl_rl_ext.utils import install_nan_sanitizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def dump_pickle(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def dump_yaml(filename, data, sort_keys=False):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        yaml.dump(data, f, sort_keys=sort_keys)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    cli_args.apply_motion_data_args(env_cfg, args_cli)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # Keep the learner on the env's device (a single-GPU run on --device cuda:K would otherwise learn on cuda:0).
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # Only rank 0 writes checkpoints and configs under DDP.
    is_rank_zero = (not args_cli.distributed) or (int(os.environ.get("RANK", "0")) == 0)

    # specify directory for logging experiments
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Resolve which checkpoint to load: an explicit warm-start path, or --resume within this experiment.
    if args_cli.load_checkpoint_path == "":
        raise ValueError(
            "--load_checkpoint_path was passed as an empty string (an unset shell variable?). Pass the path of a "
            "model_*.pt checkpoint."
        )
    is_warm_start = args_cli.load_checkpoint_path is not None
    resume_path = None
    if is_warm_start:
        resume_path = os.path.abspath(args_cli.load_checkpoint_path)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"--load_checkpoint_path not found: {resume_path}")
    elif agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Multi-teacher distillation routes teacher POLICIES, multi-critic PostRL routes teacher CRITICS: both load the
    # per-motion teachers from the registry.
    cli_args.apply_teacher_registry_arg(agent_cfg, args_cli)

    # create runner from rsl-rl; a NaN env output resets that env instead of ending the run
    install_nan_sanitizer()
    runner_class = DistillationRunner if cli_args.is_multi_teacher_distillation(agent_cfg) else OnPolicyRunner
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # load the checkpoint
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        load_cfg = None
        if is_warm_start and cli_args.is_multi_teacher_distillation(agent_cfg):
            # Keep the teachers just loaded from the registry; the checkpoint may hold older ones.
            load_cfg = {"student": True, "optimizer": args_cli.resume_optim, "teachers": False, "iteration": True}
            print("[INFO]: distillation warm start → loading the student only, teachers come from the registry")
        elif is_warm_start and not args_cli.resume_optim:
            peek = torch.load(resume_path, weights_only=False, map_location="cpu")
            if "actor_state_dict" in peek:
                # "critic": PPO, "critics": MultiCriticPPO
                load_cfg = {"actor": True, "critic": True, "critics": True, "optimizer": False, "iteration": True}
                print("[INFO]: --resume_optim=False → loading actor + critic(s) only, fresh optimizer")
            del peek
        runner.load(resume_path, load_cfg=load_cfg, map_location=agent_cfg.device)
        # A warm start loads from a DIFFERENT run: restart the iteration count. --resume continues it.
        if is_warm_start:
            runner.current_learning_iteration = 0

    # dump the configuration into log-directory (rank 0 only: each DDP rank derives its own timestamped log_dir).
    if is_rank_zero:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()

    # Tear down the NCCL process group on DDP exit (suppresses the trailing warning).
    if args_cli.distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # run the main function
    main()
    if args_cli.distributed:
        # Checkpoints and logs are written and the env is closed by now. Skip Kit's plugin teardown in torchrun
        # workers: PhysX can segfault while releasing its CUDA context there and fail a run that already finished.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    # close sim app
    simulation_app.close()
