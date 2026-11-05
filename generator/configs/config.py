#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Configuration management using dataclasses and YAML.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    """Data configuration."""

    # The dataset: a local directory (read in place), or an s3:// URI (copied
    # under cache_dir, whole or streamed chunk by chunk). See data/remote.py.
    data_dir: str = "data/halo_sit_n50k"

    # Local copy root for an s3:// data_dir; when streaming, files are deleted
    # from here as training goes. Unused for a local data_dir.
    cache_dir: str = "data/cache"

    # Image settings
    image_size: Tuple[int, int] = (224, 224)

    # Trajectory settings
    trajectory_length: int = 30  # 1 second at 30fps

    # Split
    train_split: float = 0.9

    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Preprocessing
    normalize_motion: bool = False
    cache_frames: bool = False

    # Episodes per streamed chunk (in RAM at a time); s3:// data_dir only.
    # 0 = no streaming: an s3:// data_dir is downloaded whole, then read locally.
    streaming_chunk_size: int = 0

    # Subset the dataset to the first N episodes (after metadata load).
    # 0 = use all episodes. Intended for quick smoke tests on a fraction
    # of a large dataset.
    max_episodes: int = 0

    # Cold-start augmentation for prev_chunk inpainting: at each episode
    # start, add prev_chunk_frames extra samples that blend a static hold of
    # the first frame with the future plan (see MotionDataset). Only meaningful
    # when the model inpaints the prev_chunk prefix.
    cold_start_augment: bool = False


@dataclass
class ModelConfig:
    """Model configuration."""

    # Vision-language backbone
    vlm_backbone: str = "qwen3.5-2b"  # "qwen3.5-2b" or "qwen3.5-4b"
    vlm_frozen: bool = True
    vlm_dtype: str = "bfloat16"  # "bfloat16", "float16", "float32"
    # Unfreeze the vision tower (LM stays frozen).
    vlm_vision_trainable: bool = False
    # Also unfreeze the LM (needs vlm_vision_trainable). ~2B trainable params.
    vlm_lm_trainable: bool = False
    # Qwen image preprocessing on the GPU + cached prompt tokens, instead of
    # PIL + HF processor on the CPU in the training process every step.
    vlm_gpu_preprocess: bool = False
    # Per-sample probability of a black RGB frame before the VLM during training
    # (text-only conditioning on those samples). Live VLM only.
    vision_dropout: float = 0.0

    # Conditioning tokens: every image token plus every instruction token. The
    # DiT cross-attends to them as an unordered set (positional encoding is
    # applied to the action tokens only), so the image layout — i.e. left vs
    # right — is invisible without cond_pos_embed.
    cond_pos_embed: bool = False
    max_cond_tokens: int = 16

    # DiT motion decoder
    hidden_dim: int = 512
    num_decoder_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    num_inference_steps: int = 4  # denoising steps for DiT inference

    # Output
    trajectory_length: int = 30
    output_dim: int = 36  # 29 joints + 3 pos + 4 quat

    # prev_chunk inpainting: condition on the first prev_chunk_frames of the
    # target plan by hard-inpainting them during flow-matching denoising (a
    # training-time replacement for the inference-time predict_rtc guidance).
    # The model only predicts the remaining frames; the first prev_chunk_frames
    # of every generated plan exactly equal the supplied prev_chunk.
    inpaint_prev_chunk: bool = False
    prev_chunk_frames: int = 15  # 0.5s at 30fps


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Epochs and batches
    num_epochs: int = 100
    accumulation_steps: int = 1

    # Optimizer
    learning_rate: float = 1e-4
    # LR for an unfrozen VLM vision tower. It is PRETRAINED, so the main
    # learning_rate (1e-3 in these configs) would destroy it; give it its own,
    # much smaller group. Ignored unless model.vlm_vision_trainable.
    vlm_lr: float = 1e-5
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)

    # Gradient clipping
    max_grad_norm: float = 1.0

    # TF32 tensor-core matmuls/convs for fp32 weights (e.g. the fp32 fine-tuned
    # Qwen vision tower). Off by default so existing configs keep their numerics.
    allow_tf32: bool = False

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = 10

    # Periodic open-loop evaluation (0 = disabled)
    eval_every: int = 10


@dataclass
class Config:
    """Full configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment settings
    experiment_name: str = "halo_baseline"
    seed: int = 42
    device: str = "cuda"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load config from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            Config instance
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        # Handle nested configs
        data_config = DataConfig(**data.get("data", {}))
        model_config = ModelConfig(**data.get("model", {}))
        training_config = TrainingConfig(**data.get("training", {}))

        # Handle tuple conversion for image_size and betas
        if "image_size" in data.get("data", {}):
            data_config.image_size = tuple(data["data"]["image_size"])
        if "betas" in data.get("training", {}):
            training_config.betas = tuple(data["training"]["betas"])

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            experiment_name=data.get("experiment_name", "halo_baseline"),
            seed=data.get("seed", 42),
            device=data.get("device", "cuda"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "device": self.device,
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
        }

    def __str__(self) -> str:
        """String representation."""
        return yaml.dump(self.to_dict(), default_flow_style=False)


def load_config(path: Optional[str] = None) -> Config:
    """
    Load configuration from file or return defaults.

    Args:
        path: Optional path to config file

    Returns:
        Config instance
    """
    if path is not None and Path(path).exists():
        return Config.from_yaml(path)
    return Config()
