#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
MotionGenerator: Vision-Language to Motion Trajectory model.

A Qwen3.5 vision-language encoder turns an RGB frame and a language command into
conditioning tokens; a DiT motion decoder (flow matching) generates the motion
trajectory from them.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass

from .motion_decoder import DiTMotionDecoder
from .vision_language_encoder import VLMEncoder, get_vlm_encoder_dim


@dataclass
class MotionGeneratorConfig:
    """Configuration for MotionGenerator model."""

    # Vision-language backbone
    vlm_backbone: str = "qwen3.5-2b"  # "qwen3.5-2b" or "qwen3.5-4b"
    vlm_frozen: bool = True
    vlm_dtype: str = "bfloat16"  # "bfloat16", "float16", "float32"
    # Unfreeze the VISION TOWER (all 24 layers); the language model stays frozen
    # unless vlm_lm_trainable is also set.
    vlm_vision_trainable: bool = False
    # Additionally unfreeze the language model (requires vlm_vision_trainable).
    # ~2B trainable params: 32 GB/GPU of weights+grads+Adam state, an 8 GB
    # all-reduce per step, and ~24 GB per checkpoint.
    vlm_lm_trainable: bool = False
    # Preprocess Qwen images on the GPU and cache the per-task prompt tokens
    # (models/vision_language_encoder.py:_prepare_fast). The default CPU path stalls the GPU
    # every step (.cpu(), PIL, HF processor, .tolist()).
    vlm_gpu_preprocess: bool = False
    # Probability, per training sample, of replacing the RGB frame with a black
    # image before the VLM, so both the image tokens AND the text tokens (which
    # attend to the image inside the LM) carry no scene: the DiT must act from
    # the instruction alone. Counters policies that ignore the text prompt.
    # Training forward only (never predict/eval).
    vision_dropout: float = 0.0

    # Conditioning tokens: every image token (the VLM's merged grid) plus every
    # instruction token, with a key-padding mask over the text. The DiT
    # cross-attends to them as an UNORDERED set: it adds positional encoding to
    # the action tokens only (motion_decoder.py), and vlm_image_proj shares
    # weights across image tokens. So without cond_pos_embed, "target on the
    # left" and "on the right" yield the same multiset of vectors and an
    # identical cross-attention output.
    cond_pos_embed: bool = False
    max_cond_tokens: int = 16

    # Decoder
    hidden_dim: int = 512
    num_decoder_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    # Output
    trajectory_length: int = 30  # 1 second at 30fps
    output_dim: int = 36  # 29 joints + 3 pos + 4 quat

    num_inference_steps: int = 4  # denoising steps for DiT inference

    # prev_chunk inpainting (training-time replacement for predict_rtc): hard-fix
    # the first prev_chunk_frames of the trajectory to the supplied prev_chunk
    # during flow-matching denoising; the model predicts only the remainder.
    inpaint_prev_chunk: bool = False
    prev_chunk_frames: int = 15


def model_config_from(config) -> "MotionGeneratorConfig":
    """Build a MotionGeneratorConfig from a full Config's `model` section.

    Copies EVERY MotionGeneratorConfig field that the model section defines, by
    name, so training and evaluation always build the same model.
    """
    import dataclasses
    kw = {f.name: getattr(config.model, f.name)
          for f in dataclasses.fields(MotionGeneratorConfig)
          if hasattr(config.model, f.name)}
    return MotionGeneratorConfig(**kw)


class MotionGenerator(nn.Module):
    """
    Main model for generating motion trajectories from vision and language.

    Architecture:
        RGB frame + language -> VLMEncoder -> image / text features
                             -> projections -> conditioning tokens [B, N, hidden]
        noise -> DiTMotionDecoder (flow matching, cross-attends to the tokens)
              -> trajectory [B, T, 36]

    Input:
        - rgb: [B, 3, H, W] RGB frame
        - language: List[str] of length B

    Output:
        - trajectory: [B, T, 36] motion trajectory (29 joints + 3 pos + 4 quat)
    """

    def __init__(self, config: MotionGeneratorConfig):
        super().__init__()
        self.config = config
        self.inpaint_prev_chunk = config.inpaint_prev_chunk
        self.prev_chunk_frames = config.prev_chunk_frames
        self.normalizer = None

        vlm_dim = get_vlm_encoder_dim(config.vlm_backbone)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.vlm_encoder = VLMEncoder(
            model_name=config.vlm_backbone,
            frozen=config.vlm_frozen,
            dtype=dtype_map[config.vlm_dtype],
            vision_trainable=config.vlm_vision_trainable,
            lm_trainable=config.vlm_lm_trainable,
            gpu_preprocess=config.vlm_gpu_preprocess,
        )

        # Separate projections for image and text features
        self.vlm_image_proj = nn.Sequential(
            nn.Linear(vlm_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.vlm_text_proj = nn.Sequential(
            nn.Linear(vlm_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        # Learned positions over the conditioning tokens (see config docstring).
        self.cond_pos_embed = None
        if config.cond_pos_embed:
            self.cond_pos_embed = nn.Parameter(
                torch.zeros(1, config.max_cond_tokens, config.hidden_dim))
            nn.init.normal_(self.cond_pos_embed, std=0.02)

        # Motion decoder (DiT, flow matching)
        self.dit_decoder = DiTMotionDecoder(
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            trajectory_length=config.trajectory_length,
            dropout=config.dropout,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize the VLM projection weights."""
        for module in (self.vlm_image_proj, self.vlm_text_proj):
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _vlm_tokens(self, image_feat, text_feat, text_mask):
        """VLM features -> (tokens [B, N + T, hidden], key-padding mask or None).

        image_feat [B, N, D] (every image token), text_feat [B, T, D] (every
        instruction token) and text_mask [B, T] (True = pad), as returned by
        self.vlm_encoder.
        """
        if image_feat.dim() != 3 or text_feat.dim() != 3:
            raise RuntimeError(
                f"expected per-token VLM features [B, N, D] / [B, T, D], got "
                f"{tuple(image_feat.shape)} / {tuple(text_feat.shape)}")
        # Linear/LayerNorm broadcast over leading token axes.
        image_token = self.vlm_image_proj(image_feat)
        text_token = self.vlm_text_proj(text_feat)
        tokens = torch.cat([image_token, text_token], dim=1)
        mask = None
        if text_mask is not None:
            mask = torch.cat([
                torch.zeros(image_token.shape[:2], dtype=torch.bool, device=tokens.device),
                text_mask.to(device=tokens.device, dtype=torch.bool)], dim=1)
        return tokens, mask

    def _encode_conditioning(self, conditioning: torch.Tensor) -> torch.Tensor:
        """Add learned token positions to the DiT memory (no-op unless cond_pos_embed)."""
        if self.cond_pos_embed is not None:
            n = conditioning.shape[1]
            if n > self.cond_pos_embed.shape[1]:
                raise RuntimeError(
                    f"{n} conditioning tokens exceeds max_cond_tokens="
                    f"{self.cond_pos_embed.shape[1]}")
            conditioning = conditioning + self.cond_pos_embed[:, :n]
        return conditioning

    def forward(
        self,
        rgb: torch.Tensor,
        language: List[str],
        return_features: bool = False,
        noisy_trajectory: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            rgb: [B, 3, H, W] input RGB frame
            language: List of B language commands
            return_features: If True, also return intermediate features
            noisy_trajectory: [B, T, 36] noisy trajectory (DiT training only)
            timestep: [B] flow matching timestep (DiT training only)

        Returns:
            Dict with 'velocity' [B, T, 36], the flow-matching velocity
        """
        p = self.config.vision_dropout
        if self.training and p > 0:
            drop = torch.rand(rgb.shape[0], device=rgb.device) < p
            rgb = rgb.masked_fill(drop[:, None, None, None], 0.0)
        image_feat, text_feat, text_mask = self.vlm_encoder(rgb, language)
        conditioning, cond_mask = self._vlm_tokens(image_feat, text_feat, text_mask)

        conditioning = self._encode_conditioning(conditioning)

        velocity = self.dit_decoder(noisy_trajectory, conditioning, timestep,
                                    conditioning_mask=cond_mask)
        outputs = {"velocity": velocity}

        if return_features:
            outputs["image_features"] = image_feat
            outputs["text_features"] = text_feat

        return outputs

    def predict(
        self,
        rgb: torch.Tensor,
        language: List[str],
        prev_chunk: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Simple prediction interface.

        Args:
            rgb: [B, 3, H, W] or [3, H, W] input RGB frame
            language: List[str] or str language command
            prev_chunk: Optional [P, output_dim] or [B, P, output_dim] known
                prefix in NORMALIZED space (P = prev_chunk_frames). When the
                model was trained with inpaint_prev_chunk, the first P frames of
                the generated plan are hard-fixed to this prefix during
                denoising (training-time replacement for predict_rtc).

        Returns:
            [B, T, 36] or [T, 36] motion trajectory (denormalized)
        """
        # Handle single sample
        single_sample = False
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
            single_sample = True
        if isinstance(language, str):
            language = [language]

        B = rgb.shape[0]

        # Prepare the inpainting prefix (normalized space, [B, P, D])
        do_inpaint = self.inpaint_prev_chunk and prev_chunk is not None
        if do_inpaint:
            P = self.prev_chunk_frames
            prefix = prev_chunk.to(rgb.device)
            if prefix.dim() == 2:
                prefix = prefix.unsqueeze(0).expand(B, -1, -1)

        with torch.no_grad():
            conditioning, cond_mask = self._vlm_tokens(*self.vlm_encoder(rgb, language))
            conditioning = self._encode_conditioning(conditioning)

            # Euler integration from noise
            trajectory = torch.randn(
                B, self.config.trajectory_length, self.config.output_dim,
                device=rgb.device,
            )
            if do_inpaint:
                # Seed the known prefix with the clean prev_chunk so the
                # very first denoising step already sees it as context.
                trajectory[:, :P] = prefix
            num_steps = self.config.num_inference_steps
            dt = 1.0 / num_steps

            for k in range(num_steps):
                t = torch.full((B,), k / num_steps, device=rgb.device)
                if do_inpaint:
                    # Per-token timestep: prefix frames are clean (t=1), the
                    # rest denoise at the current step (matches training).
                    t = t.view(B, 1).expand(-1, self.config.trajectory_length).clone()
                    t[:, :P] = 1.0
                velocity = self.dit_decoder(trajectory, conditioning, t,
                                            conditioning_mask=cond_mask)
                trajectory = trajectory + dt * velocity
                if do_inpaint:
                    # Hard-fix the prefix every step: the first P frames stay
                    # exactly equal to prev_chunk; only the rest is denoised.
                    trajectory[:, :P] = prefix

        if single_sample:
            trajectory = trajectory.squeeze(0)

        if self.normalizer is not None:
            trajectory = self.normalizer.denormalize(trajectory)

        return trajectory

    def predict_rtc(
        self,
        rgb: torch.Tensor,
        language: List[str],
        prev_chunk: Optional[torch.Tensor] = None,
        inference_delay: int = 4,
        execution_horizon: Optional[int] = None,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Predict with Real-Time Chunking (inpainting guidance).

        Generates a new action chunk that is consistent with the tail of the
        previous chunk (the actions already committed during inference delay).

        Args:
            rgb: [B, 3, H, W] or [3, H, W]
            language: List[str] or str
            prev_chunk: [T, output_dim] previous chunk in normalized space.
                        If None, falls back to standard predict().
            inference_delay: d — number of steps already executed from prev_chunk
            execution_horizon: s — how many fresh actions to use (default: T//2)
            beta: max guidance weight

        Returns:
            [T, output_dim] or [B, T, output_dim] trajectory (denormalized)
        """
        if prev_chunk is None:
            return self.predict(rgb, language)

        # Handle single sample
        single_sample = False
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
            single_sample = True
        if isinstance(language, str):
            language = [language]

        B = rgb.shape[0]
        T = self.config.trajectory_length
        D = self.config.output_dim
        d = inference_delay
        s = execution_horizon if execution_horizon is not None else T // 2
        device = rgb.device

        # Build target: the overlapping actions from prev_chunk
        # prev_chunk[d:] are the actions that overlap with the new chunk's prefix
        target = prev_chunk.to(device)
        if target.dim() == 2:
            target = target.unsqueeze(0).expand(B, -1, -1)

        # Normalize target if normalizer exists (guidance operates in normalized space)
        if self.normalizer is not None:
            target = self.normalizer.normalize(target)

        # Build soft mask W: [T]
        W = torch.zeros(T, device=device)
        for i in range(T):
            if i < d:
                W[i] = 1.0
            elif i < T - s:
                c_i = (T - s - i) / (T - s - d + 1)
                W[i] = c_i * (torch.exp(torch.tensor(c_i)) - 1).item() / (np.e - 1)
            # else: W[i] = 0 (free)
        W = W.view(1, T, 1).expand(B, T, D)

        with torch.no_grad():
            # Build conditioning (same as predict)
            conditioning, cond_mask = self._vlm_tokens(*self.vlm_encoder(rgb, language))
            conditioning = self._encode_conditioning(conditioning)

        # Euler integration with IIGDM guidance
        trajectory = torch.randn(B, T, D, device=device)
        num_steps = self.config.num_inference_steps
        dt = 1.0 / num_steps

        for k in range(num_steps):
            tau = k / num_steps
            t = torch.full((B,), tau, device=device)

            # Need gradients for guidance correction
            trajectory.requires_grad_(True)
            velocity = self.dit_decoder(trajectory, conditioning, t,
                                        conditioning_mask=cond_mask)

            # Estimate final clean chunk
            A_hat = trajectory + (1 - tau) * velocity

            # Weighted error
            weighted_error = (target - A_hat) * W

            # Vector-Jacobian product: correction = weighted_error^T @ dA_hat/d(trajectory)
            correction = torch.autograd.grad(
                outputs=A_hat,
                inputs=trajectory,
                grad_outputs=weighted_error,
                retain_graph=False,
            )[0]

            trajectory = trajectory.detach()

            # Guidance scale with clipping
            r_tau_sq = (1 - tau) ** 2 / (tau ** 2 + (1 - tau) ** 2 + 1e-8)
            raw_scale = (1 - tau) / (tau * r_tau_sq + 1e-8)
            guidance_scale = min(beta, raw_scale)

            # Guided Euler step
            v_guided = velocity.detach() + guidance_scale * correction
            trajectory = trajectory + dt * v_guided

        trajectory = trajectory.detach()
        if single_sample:
            trajectory = trajectory.squeeze(0)

        if self.normalizer is not None:
            trajectory = self.normalizer.denormalize(trajectory)

        return trajectory

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if self.normalizer is not None:
            for k, v in self.normalizer.state_dict().items():
                sd[f"_normalizer.{k}"] = torch.tensor(v) if isinstance(v, np.ndarray) else v
        return sd

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        normalizer_keys = [k for k in state_dict if k.startswith("_normalizer.")]
        if normalizer_keys:
            from data.transforms import MotionNormalizer
            norm_sd = {}
            for k in normalizer_keys:
                v = state_dict.pop(k)
                clean_k = k.removeprefix("_normalizer.")
                norm_sd[clean_k] = v.cpu().numpy() if isinstance(v, torch.Tensor) else v
            self.normalizer = MotionNormalizer()
            self.normalizer.load_state_dict(norm_sd)
        return super().load_state_dict(state_dict, strict=strict, **kwargs)
