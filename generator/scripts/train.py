#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Main training script for generator.

Usage (run from generator/):
    python scripts/train.py --config configs/qwen3_5_2b_dit_vision_finetune.yaml
    python scripts/train.py --config configs/qwen3_5_2b_dit_vision_finetune.yaml --device cuda:0
"""

import argparse
import sys
from pathlib import Path
import os

# Add generator root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
import random
import numpy as np

from configs.config import Config, load_config
from data.dataset import create_data_loaders
from models.motion_generator import MotionGenerator, MotionGeneratorConfig
from training.trainer import Trainer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_distributed():
    """Initialize distributed training if launched via torchrun.

    Returns True if distributed mode is active.
    """
    if "RANK" not in os.environ:
        return False
    # Generous collective timeout (default is 10 min): rank 0 does long
    # rank-0-only work at epoch boundaries (eval_openloop ~10 min, plus the
    # cold-start per-node chunk download) while other ranks wait at the next
    # collective. The default 600s NCCL watchdog aborts the job in that window
    # on multi-node; 60 min absorbs it.
    from datetime import timedelta
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True


def cleanup_distributed():
    """Destroy the distributed process group if active."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_model_config(config: Config) -> MotionGeneratorConfig:
    """Create model config from training config (shared with eval_openloop)."""
    from models.motion_generator import model_config_from
    return model_config_from(config)


def create_training_config(config: Config, args) -> dict:
    """Create training config dict from Config object."""
    return {
        "num_epochs": config.training.num_epochs,
        "accumulation_steps": config.training.accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "vlm_lr": config.training.vlm_lr,
        "weight_decay": config.training.weight_decay,
        "betas": config.training.betas,
        "max_grad_norm": config.training.max_grad_norm,
        "checkpoint_dir": config.training.checkpoint_dir,
        "checkpoint_every": config.training.checkpoint_every,
        "eval_every": config.training.eval_every,
        "inpaint_prev_chunk": config.model.inpaint_prev_chunk,
        "prev_chunk_frames": config.model.prev_chunk_frames,
        "config_path": args.config,
        "eval_episode_idx": 0,
        "use_wandb": args.wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": config.experiment_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Train generator")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="generator",
        help="W&B project name",
    )
    args = parser.parse_args()

    # Distributed training setup (auto-detected from torchrun env vars)
    is_distributed = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # is_main = the single global rank 0 (so multi-node logs once, not per node).
    is_main = not is_distributed or dist.get_rank() == 0

    # Load configuration
    config = load_config(args.config)

    # Set device: use LOCAL_RANK when distributed, else respect --device or auto-detect
    if is_distributed:
        config.device = f"cuda:{local_rank}"
    elif args.device is not None:
        config.device = args.device
    elif config.device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            config.device = "mps"
            if is_main:
                print("CUDA not available, using MPS (Apple Silicon)")
        else:
            config.device = "cpu"
            if is_main:
                print("CUDA not available, using CPU")

    if is_main:
        print(f"Using device: {config.device}")
        if is_distributed:
            print(f"Distributed training: {dist.get_world_size()} GPUs")

    # Set seed (offset per rank for data augmentation diversity;
    # DDP broadcasts model weights from rank 0 regardless)
    set_seed(config.seed + (dist.get_rank() if is_distributed else 0))
    if config.training.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if is_main:
        print(f"Random seed: {config.seed}")
        print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")

        # Print config
        print("\n" + "=" * 60)
        print("Configuration:")
        print("=" * 60)
        print(config)
        print("=" * 60 + "\n")

    # Create data loaders
    if is_main:
        print("Loading dataset...")
    train_loader, val_loader = create_data_loaders(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        train_split=config.data.train_split,
        image_size=config.data.image_size,
        trajectory_length=config.data.trajectory_length,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        normalize_motion=config.data.normalize_motion,
        cache_frames=config.data.cache_frames,
        distributed=is_distributed,
        streaming_chunk_size=config.data.streaming_chunk_size,
        max_episodes=config.data.max_episodes,
        prev_chunk_frames=config.model.prev_chunk_frames,
        cold_start_augment=config.data.cold_start_augment,
        cache_dir=config.data.cache_dir,
    )
    if is_main:
        print(f"Train samples: {len(train_loader.dataset)}")
        if val_loader is not None:
            print(f"Val samples: {len(val_loader.dataset)}")
        else:
            print("Val: skipped (streaming mode)")

    # Create model
    if is_main:
        print("\nCreating model...")
    model_config = create_model_config(config)
    model = MotionGenerator(model_config)

    # Pass normalizer from dataset to model for denormalization in predict()
    if hasattr(train_loader, 'underlying_dataset'):
        normalizer = train_loader.underlying_dataset.get_normalizer()
    elif hasattr(train_loader.dataset, 'dataset'):
        normalizer = train_loader.dataset.dataset.get_normalizer()
    else:
        normalizer = getattr(train_loader.dataset, 'get_normalizer', lambda: None)()
    if normalizer is not None:
        model.normalizer = normalizer
        if is_main:
            print("Motion normalizer attached to model")

    if is_main:
        print(f"Model parameters: {model.get_num_params():,}")

    # Create training config dict
    training_config = create_training_config(config, args)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=config.device,
    )

    # Train
    if is_main:
        print("\nStarting training...")
    try:
        trainer.train()
    finally:
        cleanup_distributed()

    if is_main:
        print("\nTraining complete!")


if __name__ == "__main__":
    main()
