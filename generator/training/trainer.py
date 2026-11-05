#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Training loop for motion generator.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from contextlib import nullcontext
from pathlib import Path
import logging
import math
import os
from typing import Dict, Optional, Any
from tqdm import tqdm
import json
import subprocess
import sys
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from .flow_matching import FlowMatchingScheduler, FlowMatchingLoss


class Trainer:
    """
    Trainer for motion generation model.

    Handles:
    - Training loop with gradient accumulation
    - Validation
    - Checkpointing
    - Logging (console + file)
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        """
        Initialize Trainer.

        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration dict
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Distributed training state
        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        self.is_main = (self.rank == 0)

        # Wrap model with DDP
        if self.is_distributed:
            local_rank = int(os.environ["LOCAL_RANK"])
            # broadcast_buffers=False is value-neutral: the only registered buffer
            # is the DiT's constant sinusoidal positional encoding, identical on
            # every rank, so DDP's per-forward buffer broadcast is pure overhead.
            self.model = DDP(
                self.model, device_ids=[local_rank], broadcast_buffers=False,
            )

        # Unwrapped model reference (for checkpoints, param counts, etc.)
        self.raw_model = self.model.module if self.is_distributed else self.model

        # Flow matching (the DiT predicts a velocity field)
        self.flow_scheduler = FlowMatchingScheduler()
        self.flow_loss_fn = FlowMatchingLoss()

        # prev_chunk inpainting: hard-fix the first prev_chunk_frames of the
        # trajectory to the clean target prefix during the flow-matching step and
        # exclude them from the velocity loss (the model only predicts the rest).
        self.inpaint_prev_chunk = config.get("inpaint_prev_chunk", False)
        self.prev_chunk_frames = config.get("prev_chunk_frames", 15)

        # Optimizer (only trainable params, avoids wasting memory on frozen VLM).
        # An unfrozen VLM vision tower gets its OWN group at a much smaller LR:
        # it is pretrained, and these configs run learning_rate 1e-3, which would
        # wreck it in the first few steps.
        base_lr = config.get("learning_rate", 1e-4)
        vlm_lr = config.get("vlm_lr", 1e-5)
        raw = model.module if hasattr(model, "module") else model
        vision_params, vision_ids = [], set()
        enc = getattr(raw, "vlm_encoder", None)
        if enc is not None and getattr(enc, "vision_trainable", False):
            vision_params = enc.trainable_vlm_parameters()
            vision_ids = {id(p) for p in vision_params}
        other = [p for p in model.parameters()
                 if p.requires_grad and id(p) not in vision_ids]
        groups = [{"params": other, "lr": base_lr}]
        if vision_params:
            groups.append({"params": vision_params, "lr": vlm_lr})
            if self.is_main:
                # print(), not self.logger: _setup_logging() runs later in
                # __init__, so self.logger does not exist yet here.
                print(f"[trainer] optimizer groups: "
                      f"{sum(p.numel() for p in other):,} params @ lr {base_lr:g}, "
                      f"vision tower {sum(p.numel() for p in vision_params):,} "
                      f"@ lr {vlm_lr:g}", flush=True)
        self.optimizer = AdamW(
            groups,
            lr=base_lr,
            weight_decay=config.get("weight_decay", 0.01),
            betas=config.get("betas", (0.9, 0.999)),
        )

        # Learning rate: cosine decay over the epochs, stepped once per epoch
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("num_epochs", 100),
            eta_min=config.get("learning_rate", 1e-4) * 0.01,
        )

        # Gradient accumulation
        self.accumulation_steps = config.get("accumulation_steps", 1)

        # Logging
        self._setup_logging(config)
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Checkpoint directory ($HALO_OUTPUT_DIR overrides the config, e.g. to
        # point at a cluster job's persistent artifact directory)
        self.checkpoint_dir = Path(
            os.environ.get("HALO_OUTPUT_DIR") or config.get("checkpoint_dir", "./checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer (rank 0 only)
        self.writer = None
        if self.is_main and HAS_TENSORBOARD:
            tb_log_dir = self.checkpoint_dir / "tensorboard"
            self.writer = SummaryWriter(log_dir=str(tb_log_dir))

        # Weights & Biases (rank 0 only)
        self.wandb_run = None
        if self.is_main and config.get("use_wandb", False) and HAS_WANDB:
            wandb_project = config.get("wandb_project", "generator")
            wandb_name = config.get("wandb_run_name", None)
            self.wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_name,
                config=config,
            )
            # Plot epoch-level families against "epoch" rather than the raw
            # global step, while all logs still share one monotone step.
            wandb.define_metric("epoch")
            for fam in ("train/*", "val/*", "eval/*"):
                wandb.define_metric(fam, step_metric="epoch")

        # Periodic open-loop evaluation
        self.eval_every = config.get("eval_every", 0)
        self.config_path = config.get("config_path", None)
        self.eval_episode_idx = config.get("eval_episode_idx", 0)

        # Save config (rank 0 only)
        if self.is_main:
            self._save_config()

    def _setup_logging(self, config: Dict):
        """Setup logging to console and file (rank 0 only for file/info)."""
        self.logger = logging.getLogger(f"Trainer-rank{self.rank}")
        self.logger.handlers = []

        if self.is_main:
            self.logger.setLevel(logging.INFO)

            log_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
            log_dir.mkdir(parents=True, exist_ok=True)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)

            # File handler
            file_handler = logging.FileHandler(log_dir / "train.log")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(console_format)
            self.logger.addHandler(file_handler)
        else:
            # Non-main ranks: suppress info logging
            self.logger.setLevel(logging.WARNING)

    def _save_config(self):
        """Save training configuration."""
        config_path = self.checkpoint_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)

    def _batch_to_device(self, batch):
        """-> (rgb, language, trajectory), tensors on self.device."""
        return (batch["rgb"].to(self.device), batch["language"],
                batch["trajectory"].to(self.device))

    def _apply_prev_chunk_inpainting(self, trajectory, noisy_traj, t):
        """Prepare the flow-matching step for prev_chunk hard-inpainting.

        When inpaint_prev_chunk is set, the first prev_chunk_frames are the known
        clean prefix, so we (1) copy them clean into the model input, (2) mark
        them t=1 per-token — in this scheduler's convention noisy = t*clean +
        (1-t)*noise, so t=1 is clean, letting AdaLN modulate them as clean and
        making the count of t=1 tokens signal the inference delay (training-time
        RTC, Fig. 2) — and (3) exclude them from the loss via the returned mask.
        When it is off, returns the inputs unchanged (scalar timestep, no mask).

        Returns:
            (noisy_traj, model_t, flow_mask)
        """
        if not self.inpaint_prev_chunk:
            return noisy_traj, t, None
        P = self.prev_chunk_frames
        B, T = trajectory.shape[0], trajectory.shape[1]
        noisy_traj = noisy_traj.clone()
        noisy_traj[:, :P] = trajectory[:, :P]
        flow_mask = torch.ones(B, T, device=self.device)
        flow_mask[:, :P] = 0.0
        model_t = t.view(-1, 1).expand(-1, T).clone()
        model_t[:, :P] = 1.0
        return noisy_traj, model_t, flow_mask

    def train(self):
        """Run full training loop."""
        num_epochs = self.config.get("num_epochs", 100)
        checkpoint_every = self.config.get("checkpoint_every", 10)

        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Model parameters: {self.raw_model.get_num_params():,}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Update DistributedSampler epoch for proper shuffling
            if self.is_distributed:
                if hasattr(self.train_loader, "set_epoch"):
                    self.train_loader.set_epoch(epoch)
                else:
                    self.train_loader.sampler.set_epoch(epoch)
                if self.val_loader is not None:
                    if hasattr(self.val_loader, "set_epoch"):
                        self.val_loader.set_epoch(epoch)
                    elif hasattr(self.val_loader, "sampler") and hasattr(self.val_loader.sampler, "set_epoch"):
                        self.val_loader.sampler.set_epoch(epoch)

            # Train one epoch
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._validate(epoch)

            self.scheduler.step()

            # Logging
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            log_msg = (
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['loss']:.4f}, "
            )
            if val_metrics:
                log_msg += f"val_loss={val_metrics['loss']:.4f}, "
            log_msg += f"lr={lr:.2e}, time={epoch_time:.1f}s"
            self.logger.info(log_msg)

            # Log epoch-level metrics to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("train/loss", train_metrics["loss"], epoch)
                self.writer.add_scalar("train/lr", lr, epoch)
                if val_metrics:
                    self.writer.add_scalar("val/loss", val_metrics["loss"], epoch)

            # Log epoch-level metrics to W&B
            if self.wandb_run is not None:
                wandb_log = {"train/loss": train_metrics["loss"], "train/lr": lr, "epoch": epoch}
                for k, v in train_metrics.items():
                    wandb_log[f"train/{k}"] = v
                if val_metrics:
                    for k, v in val_metrics.items():
                        wandb_log[f"val/{k}"] = v
                # step=self.global_step, NOT step=epoch: the per-step perf/* logs
                # advance W&B's step counter, and W&B silently drops logs to an
                # earlier step. "epoch" is carried as a field and registered as
                # the x-axis via define_metric, so plots are still per-epoch.
                wandb.log(wandb_log, step=self.global_step)

            # Checkpoint based on validation loss (or train loss if no val)
            current_loss = val_metrics.get("loss", train_metrics["loss"])
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                if self.is_main:
                    self._save_checkpoint(epoch, is_best=True)
                    self.logger.info(f"  New best model! loss={current_loss:.4f}")

            if (epoch + 1) % checkpoint_every == 0:
                if self.is_main:
                    self._save_checkpoint(epoch)

            # Periodic open-loop evaluation (rank 0 only). All other ranks wait
            # at a barrier; the PG timeout (60 min, set in setup_distributed)
            # tolerates the eval duration. We do NOT use a file sentinel: it
            # lived under checkpoint_dir, which is node-local on multi-node, so a
            # sentinel written by rank 0 is invisible to other nodes' ranks and
            # deadlocks the following barrier.
            if self.eval_every > 0 and (epoch + 1) % self.eval_every == 0:
                if self.is_main:
                    self._run_eval_openloop(epoch)
                if self.is_distributed:
                    dist.barrier()

        total_time = time.time() - start_time
        self.logger.info(f"Training complete! Total time: {total_time/3600:.2f} hours")

        # Save final checkpoint
        if self.is_main:
            self._save_checkpoint(num_epochs - 1, is_final=True)

        # Final open-loop evaluation
        if self.eval_every > 0:
            if self.is_main:
                self._run_eval_openloop(num_epochs - 1, is_final=True)
            if self.is_distributed:
                dist.barrier()

        if self.writer is not None:
            self.writer.close()

        if self.wandb_run is not None:
            wandb.finish()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        # Non-finite batches seen in THIS epoch (the abort threshold is a fraction
        # of this epoch's batches, not of the whole run).
        epoch_nonfinite = 0

        loader = self.train_loader
        if self.is_main:
            loader = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

        self.optimizer.zero_grad()

        # Throughput instrumentation: split wall time into dataloader wait (time
        # blocked waiting for the next batch) vs everything else (forward,
        # backward, optimizer step, DDP sync). Averaged over the whole window of
        # throughput_log_every steps, so the less frequent all-reduce + optimizer
        # steps are included. Low overhead: cuda.synchronize only on logging
        # steps, rank 0 only.
        log_every = int(self.config.get("throughput_log_every", 20))
        _is_cuda = str(self.device).startswith("cuda")
        _t_prev = time.perf_counter()
        _win_t0, _win_wait, _win_n = _t_prev, 0.0, 0

        for batch_idx, batch in enumerate(loader):
            _data_wait = time.perf_counter() - _t_prev
            _win_wait += _data_wait
            rgb, language, trajectory = self._batch_to_device(batch)

            # Skip DDP gradient sync on intermediate accumulation steps
            is_accumulation_step = (batch_idx + 1) % self.accumulation_steps != 0
            no_sync = self.model.no_sync if (self.is_distributed and is_accumulation_step) else nullcontext

            with no_sync():
                # Forward pass: flow matching
                noise = torch.randn_like(trajectory)
                t = self.flow_scheduler.sample_timestep(trajectory.shape[0], self.device)
                noisy_traj = self.flow_scheduler.add_noise(trajectory, noise, t)
                velocity_target = self.flow_scheduler.get_velocity_target(trajectory, noise)

                # prev_chunk inpainting (no-op when disabled): clean prefix
                # context + per-token t=1 on the prefix + loss mask.
                noisy_traj, model_t, flow_mask = self._apply_prev_chunk_inpainting(
                    trajectory, noisy_traj, t
                )

                outputs = self.model(rgb, language, noisy_trajectory=noisy_traj, timestep=model_t)
                losses = self.flow_loss_fn(outputs["velocity"], velocity_target, mask=flow_mask)
                loss = losses["loss"] / self.accumulation_steps
                loss.backward()

            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                max_grad_norm = self.config.get("max_grad_norm", 1.0)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm,
                    error_if_nonfinite=False,
                )

                # NaN/Inf guard. Gate the optimizer step on a FINITE GLOBAL grad
                # norm. total_norm is computed from the post-all-reduce gradients,
                # so it is identical on every rank — a nan on any single rank makes
                # it nan everywhere and all ranks skip in lockstep. A per-rank skip
                # would desync the all-reduce and hang instead.
                #
                # Skipping protects the weights AND Adam's moment buffers: one nan
                # reaching the moments corrupts every later step even on finite
                # grads.
                grads_finite = bool(torch.isfinite(total_norm))
                if not grads_finite:
                    self._nonfinite_skips = getattr(self, "_nonfinite_skips", 0) + 1
                    if self.is_main and self._nonfinite_skips <= 50:
                        print(f"[nan-guard] non-finite grad norm at epoch {epoch} "
                              f"batch {batch_idx}; skipping optimizer step "
                              f"({self._nonfinite_skips} so far)", flush=True)

                if grads_finite:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # Accumulate metrics — EXCLUDING non-finite batches, so a degenerate
            # regime shows up as a rising nonfinite count next to a still-readable
            # mean instead of poisoning the epoch mean to a bare `nan`.
            _loss_val = losses["loss"].item()
            if math.isfinite(_loss_val):
                total_loss += _loss_val
                num_batches += 1
            else:
                self._nonfinite_batches = getattr(self, "_nonfinite_batches", 0) + 1
                epoch_nonfinite += 1
                if self._nonfinite_batches <= 5 or self._nonfinite_batches % 500 == 0:
                    print(f"[nonfinite] step {self.global_step} epoch {epoch}: loss is "
                          f"{_loss_val} (count={self._nonfinite_batches}); optimizer "
                          f"step was skipped", flush=True)
            self.global_step += 1

            # Log per-step metrics to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("train/loss_step", losses["loss"].item(), self.global_step)

            # Update progress bar
            if self.is_main and hasattr(loader, "set_postfix"):
                loader.set_postfix({"loss": f"{losses['loss'].item():.4f}"})

            # Throughput logging (rank 0, every log_every steps). Sync only here
            # so the compute-vs-data split is accurate without per-step overhead.
            _win_n += 1
            if self.is_main and log_every > 0 and _win_n >= log_every:
                if _is_cuda:
                    torch.cuda.synchronize(self.device)
                _step_time = (time.perf_counter() - _win_t0) / _win_n
                _data_wait = _win_wait / _win_n
                _compute = max(_step_time - _data_wait, 0.0)
                _bs = trajectory.shape[0]
                _sps = (_bs * self.world_size) / max(_step_time, 1e-9)
                _msg = (
                    f"[throughput] step {self.global_step} "
                    f"it/s={1.0/max(_step_time,1e-9):.2f} samples/s={_sps:.1f} "
                    f"data_wait={_data_wait*1e3:.1f}ms compute={_compute*1e3:.1f}ms "
                    f"data_frac={_data_wait/max(_step_time,1e-9):.2f}"
                )
                self.logger.info(_msg)
                if self.writer is not None:
                    self.writer.add_scalar("perf/samples_per_s", _sps, self.global_step)
                    self.writer.add_scalar("perf/it_per_s", 1.0/max(_step_time,1e-9), self.global_step)
                    self.writer.add_scalar("perf/data_wait_ms", _data_wait*1e3, self.global_step)
                    self.writer.add_scalar("perf/compute_ms", _compute*1e3, self.global_step)
                if self.wandb_run is not None:
                    wandb.log({
                        "perf/samples_per_s": _sps,
                        "perf/it_per_s": 1.0/max(_step_time,1e-9),
                        "perf/data_wait_ms": _data_wait*1e3,
                        "perf/compute_ms": _compute*1e3,
                        "perf/data_frac": _data_wait/max(_step_time,1e-9),
                    }, step=self.global_step)

            _t_prev = time.perf_counter()
            if log_every > 0 and _win_n >= log_every:
                _win_t0, _win_wait, _win_n = _t_prev, 0.0, 0

        # num_batches counts only FINITE batches now, so it can be 0 if an epoch
        # diverged completely — guard the division rather than raising
        # ZeroDivisionError on top of an already-bad run.
        _n = max(1, num_batches)
        return {
            "loss": total_loss / _n,
            "nonfinite_batches": epoch_nonfinite,
            "finite_batches": num_batches,
        }

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            rgb, language, trajectory = self._batch_to_device(batch)

            noise = torch.randn_like(trajectory)
            t = self.flow_scheduler.sample_timestep(trajectory.shape[0], self.device)
            noisy_traj = self.flow_scheduler.add_noise(trajectory, noise, t)
            velocity_target = self.flow_scheduler.get_velocity_target(trajectory, noise)
            noisy_traj, model_t, flow_mask = self._apply_prev_chunk_inpainting(
                trajectory, noisy_traj, t
            )
            outputs = self.model(rgb, language, noisy_trajectory=noisy_traj, timestep=model_t)
            losses = self.flow_loss_fn(outputs["velocity"], velocity_target, mask=flow_mask)

            total_loss += losses["loss"].item()
            num_batches += 1

        # Synchronize validation metrics across all ranks
        if self.is_distributed:
            metrics_tensor = torch.tensor(
                [total_loss, num_batches], device=self.device, dtype=torch.float64,
            )
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            total_loss, num_batches = metrics_tensor.tolist()

        return {"loss": total_loss / num_batches}

    def _save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        is_final: bool = False,
    ):
        """Save model checkpoint (rank 0 only)."""
        if not self.is_main:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.raw_model.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # Save epoch checkpoint
        if not is_best and not is_final:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        # Save final model
        if is_final:
            final_path = self.checkpoint_dir / "final_model.pt"
            torch.save(checkpoint, final_path)

    def _run_eval_openloop(self, epoch: int, is_final: bool = False):
        """Run eval_openloop.py as a subprocess on the current best checkpoint."""
        checkpoint_path = self.checkpoint_dir / "best_model.pt"
        if not checkpoint_path.exists():
            self.logger.warning("No best_model.pt found, skipping eval_openloop")
            return

        if self.config_path is None:
            self.logger.warning("No config_path set, skipping eval_openloop")
            return

        tag = "final" if is_final else f"epoch_{epoch}"
        output_dir = self.checkpoint_dir / f"eval_{tag}"

        cmd = [
            sys.executable, "scripts/eval_openloop.py",
            "--checkpoint", str(checkpoint_path),
            "--config", self.config_path,
            "--output_dir", str(output_dir),
            "--episode_idx", str(self.eval_episode_idx),
            # In-training eval must finish under the 30-min subprocess cap, which
            # evaluating every start frame of a long episode can exceed.
            # Subsample to a representative set of windows (metrics + gif);
            # standalone eval (no flag) still covers all frames.
            "--max_start_frames", str(self.config.get("eval_max_start_frames", 48)),
        ]

        self.logger.info(f"Running eval_openloop ({tag})...")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800,
            )
            if result.returncode != 0:
                self.logger.error(f"eval_openloop failed:\n{result.stderr[-500:]}")
            else:
                self.logger.info(f"eval_openloop ({tag}) completed successfully")
        except subprocess.TimeoutExpired:
            self.logger.error(f"eval_openloop ({tag}) timed out after 30 min")
        except Exception as e:
            self.logger.error(f"eval_openloop ({tag}) error: {e}")

        # Log eval metrics to W&B
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists() and self.wandb_run is not None:
            with open(metrics_path) as f:
                metrics_data = json.load(f)
            wandb_log = {"epoch": epoch}
            for k, v in metrics_data.get("avg_metrics", {}).items():
                wandb_log[f"eval/{k}"] = v
            wandb.log(wandb_log, step=self.global_step)
