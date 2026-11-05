# generator

Vision-language motion generator for humanoid robots. Predicts 5-second (150-frame) whole-body motion plans from an RGB image and a natural language command.

## Requirements

Linux with an NVIDIA GPU and CUDA 12.8 wheels, `conda`, and `sudo` (or root) for `apt-get` (EGL and ffmpeg for the
evaluation videos); a local CUDA toolkit for the optional `causal-conv1d` build. On macOS `setup.sh` installs the
default PyTorch without the CUDA-only extras. Run every command from `generator/`.

## Setup

```bash
bash setup.sh                 # conda env halo_generator (Python 3.10)
conda activate halo_generator
```

## Weights and data

- Backbone: [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B), downloaded from Hugging Face on the first
  run into `$HF_HOME` (default `~/.cache/huggingface`).
- No trained generator checkpoint and no dataset ship with the repository: train on data you generate with
  [`../datagen/`](../datagen/README.md).

## Training data

In `configs/qwen3_5_2b_dit_vision_frozen.yaml`, set `data_dir` to the dataset datagen writes (the stitched
`$HALO_S3_ROOT/halo_sit_n50k`, with `data/`, `videos/` and `meta/`): an `s3://` URI, or a local directory (then also
set `streaming_chunk_size: 0`). For `s3://` the loader uses the standard AWS credential chain (`AWS_ENDPOINT_URL` for
other S3-compatible stores) and copies files under `data.cache_dir` (default `data/cache`); with streaming it holds
`streaming_chunk_size` episodes at a time and deletes them as training goes. Environment knobs: `LOADER_S3_WORKERS`
(parallel downloads, default 64), `PERSIST_CHUNKS=1` (keep chunks on disk across epochs), `PREDOWNLOAD_ALL=1`
(fetch every chunk up front; needs `PERSIST_CHUNKS=1`).

## Training

```bash
python scripts/train.py --config configs/qwen3_5_2b_dit_vision_frozen.yaml
```

The configs are tuned for 64 GPUs at 80 samples per GPU (global batch 5120, `num_workers: 24`). On fewer GPUs, raise
`training.accumulation_steps` to keep the global batch, or lower `data.batch_size` for smaller GPUs.

## Multi-GPU training

```bash
# one node, 8 GPUs
torchrun --nproc_per_node 8 scripts/train.py --config configs/qwen3_5_2b_dit_vision_frozen.yaml

# 8 nodes x 8 GPUs: run on every node with its NODE_RANK (0-7) and the rank-0 node's address
torchrun --nnodes 8 --node_rank $NODE_RANK --nproc_per_node 8 \
    --master_addr $MASTER_ADDR --master_port 29500 \
    scripts/train.py --config configs/qwen3_5_2b_dit_vision_frozen.yaml
```

Outputs go to `training.checkpoint_dir` (`$HALO_OUTPUT_DIR` overrides it): `best_model.pt`, `final_model.pt`,
`checkpoint_epoch_N.pt`, `config.json`, `train.log` and TensorBoard logs in `tensorboard/` (if `tensorboard` is
installed). `--wandb` logs to Weights & Biases (`pip install wandb`). Every `eval_every` epochs rank 0 also runs the
open-loop evaluation below on episode 0, so keep episode 0 (datagen shard 0) in the dataset.

## Open-loop evaluation

```bash
python scripts/eval_openloop.py --checkpoint checkpoints_qwen2b_dit_vision_frozen/best_model.pt \
    --config configs/qwen3_5_2b_dit_vision_frozen.yaml --episode_idx 1004 --max_start_frames 48
```

Replays the dataset episode (ids are datagen's `shard * 1000 + i`; `--data_dir` overrides the config's) and predicts
a plan from evenly spaced start frames. It writes `eval_output/` (`--output_dir`): `metrics.json` (MPJPE, joint /
pelvis errors, heading error, turn-sign accuracy) and a GIF / MP4 of predicted vs. ground-truth motion rendered with
MuJoCo (`MUJOCO_GL=egl` is set in the env; needs `ffmpeg`).

## Configs

| Config | Qwen3.5-2B |
|---|---|
| `qwen3_5_2b_dit_vision_frozen.yaml` | frozen; only the projections and the DiT train |
| `qwen3_5_2b_dit_vision_finetune.yaml` | the whole VLM fine-tunes at `vlm_lr` 1e-5 |
| `qwen3_5_2b_dit_vision_finetune_vdrop.yaml` | as finetune, plus a black RGB frame for 30 % of training samples (`vision_dropout`) so the plan also follows the instruction alone |
