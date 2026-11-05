#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""MLX (Apple-Silicon) inpaint Euler loop for the DiT motion decoder.

The torch ``DiTMotionDecoder`` remains the single source of truth for the weights; this module
re-expresses its forward in MLX so the per-step kernel-dispatch overhead that dominates at batch
size 1 on MPS is collapsed into one compiled, lazily-evaluated graph.

  * fp32 everywhere; ``mx.compile`` + ``mx.fast.scaled_dot_product_attention``.
  * torch<->MLX crossing stages through CPU numpy (no zero-copy dlpack between torch-MPS and
    MLX-GPU in mlx 0.31.x); it happens only twice per predict() and the tensors are KB-sized.
  * MLX is lazy: the whole Euler loop builds one graph, materialized by a single ``mx.eval``.
  * Inpaint is exact: the timestep is per token (prefix t=1, rest t=k/N) and AdaLN is a pure
    function of t, so per-token modulation = ``where(token < P, mod(1.0), mod(k/N))``.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as mxnn
import numpy as np
import torch


# A linear layer's parameters, stored in torch's [out, in] layout so that the
# forward matches ``F.linear`` exactly: y = x @ W.T + b.
LinearWB = Tuple["mx.array", "mx.array"]


def _to_mx(t: torch.Tensor) -> "mx.array":
    """torch tensor -> MLX fp32 array (bit-exact, staged through CPU numpy)."""
    return mx.array(t.detach().to("cpu", torch.float32).contiguous().numpy())


def _to_torch(a: "mx.array", device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """MLX array -> torch tensor on ``device``. ``np.array(a)`` forces mx.eval."""
    return torch.from_numpy(np.array(a, copy=True)).to(device=device, dtype=dtype)


def _param(sd: Dict[str, torch.Tensor], key: str, expected_shape: Tuple[int, ...],
           consumed: set) -> "mx.array":
    """Fetch one parameter, assert its shape, mark it consumed. Fails loudly."""
    if key not in sd:
        raise KeyError(f"[MLX weight transfer] missing parameter: {key!r}")
    t = sd[key]
    if tuple(t.shape) != expected_shape:
        raise ValueError(
            f"[MLX weight transfer] shape mismatch for {key!r}: "
            f"got {tuple(t.shape)}, expected {expected_shape}")
    consumed.add(key)
    return _to_mx(t)


class MLXInpaintDiTDecoder:
    """MLX re-implementation of ``DiTMotionDecoder``'s forward, run as the inpaint Euler loop.

    The constructor performs an explicit, fully-audited weight transfer from the
    torch module's ``state_dict`` and fails loudly if any expected parameter is
    missing/mis-shaped or if any parameter is left unmapped.
    """

    def __init__(self, torch_decoder: torch.nn.Module) -> None:
        if not mx.metal.is_available():
            raise RuntimeError("MLX Metal is unavailable.")
        # Apple-GPU fp32 matmul is ~1e-3 less precise than torch-MPS (≈0.1°/joint).
        self._stream = mx.gpu

        # THREADING: MLX-GPU state (arrays, compiled graphs, streams) is bound to
        # the thread that first creates/uses it; touching it from another thread
        # raises "There is no Stream(gpu, 0) in current thread". So all MLX work —
        # construction AND every call — runs on one private single-threaded executor.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-dit")
        # Build MLX state ON the executor thread (.result() re-raises on caller).
        self._executor.submit(self._build_mlx_state, torch_decoder).result()

    def _build_mlx_state(self, torch_decoder: torch.nn.Module) -> None:
        """Create all MLX arrays + the compiled step ON the executor thread."""
        # ---- read architecture from the torch module (no guessing) ----------
        self.hidden_dim = int(torch_decoder.hidden_dim)
        self.output_dim = int(torch_decoder.output_dim)
        self.trajectory_length = int(torch_decoder.trajectory_length)
        self.num_layers = len(torch_decoder.blocks)
        # num_heads is not stored on the module; read it from an attention layer.
        self.num_heads = int(torch_decoder.blocks[0].self_attn.num_heads)
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must divide num_heads"
        self.head_dim = self.hidden_dim // self.num_heads
        self.eps = 1e-5  # torch LayerNorm default, elementwise_affine=False

        self._W = self._transfer_weights(torch_decoder)

        # Timestep sinusoidal frequencies (frequency_dim=256 -> half=128), computed
        # in fp32 to mirror TimestepEncoder.forward.
        half = 128
        freqs = np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / half)
        self._freqs = mx.array(freqs.astype(np.float32))

        # AdaLN (modulated LayerNorm) is the most-invoked op (3L+1 per step); it is fused
        # into ONE hand-written Metal kernel, which beats the compiler's elementwise+reduce.
        self._metal_adaln = self._build_metal_adaln()

        # Cache of precomputed per-step modulations, keyed by (num_steps, B).
        # The denoising schedule is fixed, so these are reused across generations.
        self._mod_cache: Dict = {}
        self._tok_step = mx.compile(self._build_tok_step())
        self._mod_one: dict = {}

    def _build_metal_adaln(self) -> Callable:
        """Fused modulated-LayerNorm as a single Metal kernel.

        One launch: y = LN(x) * (1 + scale) + shift, with mean/var reduced over the
        hidden dim H in threadgroup memory. scale/shift are [B, H] (timestep modulation,
        broadcast over T); x is [B, T, H], fp32.
        """
        H, T, eps = self.hidden_dim, self.trajectory_length, self.eps
        TGS = 256  # threadgroup size; power-of-two for the tree reduction
        source = f"""
            uint row = threadgroup_position_in_grid.x;
            uint lane = thread_position_in_threadgroup.x;
            uint nt = threads_per_threadgroup.x;
            const uint Hh = {H};
            const uint Tt = {T};
            uint b = row / Tt;                                  // batch index
            const device float* xr = x + (uint)row * Hh;
            device float* yr = out + (uint)row * Hh;
            const device float* sc = scale + (uint)b * Hh;
            const device float* sh = shift + (uint)b * Hh;
            threadgroup float ts[{TGS}];
            threadgroup float tq[{TGS}];
            float ssum = 0.0f, sq = 0.0f;
            for (uint h = lane; h < Hh; h += nt) {{ float v = xr[h]; ssum += v; sq += v * v; }}
            ts[lane] = ssum; tq[lane] = sq;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint off = nt >> 1; off > 0; off >>= 1) {{
                if (lane < off) {{ ts[lane] += ts[lane + off]; tq[lane] += tq[lane + off]; }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            float mean = ts[0] / (float)Hh;
            float var = tq[0] / (float)Hh - mean * mean;        // biased (ddof=0), matches torch
            float rstd = rsqrt(var + {eps}f);
            for (uint h = lane; h < Hh; h += nt) {{
                float xn = (xr[h] - mean) * rstd;
                yr[h] = xn * (1.0f + sc[h]) + sh[h];
            }}
        """
        kernel = mx.fast.metal_kernel(
            name="dit_adaln_fused",
            input_names=["x", "scale", "shift"],
            output_names=["out"],
            source=source,
        )

        def _forward(x: "mx.array", scale: "mx.array", shift: "mx.array") -> "mx.array":
            if x.dtype != mx.float32:
                raise TypeError(f"metal AdaLN expects fp32, got {x.dtype}")
            rows = x.shape[0] * x.shape[1]
            return kernel(inputs=[x, scale, shift],
                          grid=(TGS * rows, 1, 1), threadgroup=(TGS, 1, 1),
                          output_shapes=[x.shape], output_dtypes=[x.dtype])[0]

        return _forward

    def _run(self, fn: Callable, *args):
        """Dispatch an MLX computation onto the private executor thread."""
        return self._executor.submit(fn, *args).result()

    # ------------------------------------------------------------------ #
    # Weight transfer (highest-risk surface — explicit and asserted)
    # ------------------------------------------------------------------ #
    def _transfer_weights(self, dec: torch.nn.Module) -> Dict:
        sd: Dict[str, torch.Tensor] = dec.state_dict()
        consumed: set = set()
        H = self.hidden_dim

        def lin(prefix: str, out_f: int, in_f: int) -> LinearWB:
            return (_param(sd, f"{prefix}.weight", (out_f, in_f), consumed),
                    _param(sd, f"{prefix}.bias", (out_f,), consumed))

        # AdaLN modulation projections (norm1/2/3 per block + final_norm) all take
        # the SAME timestep embedding, so the step can run them as a single fused
        # matmul (one launch instead of 3*num_layers+1). Collect them in call order.
        adaln_w: List["mx.array"] = []
        adaln_b: List["mx.array"] = []

        def add_norm(prefix: str) -> int:
            w, b = lin(prefix, 2 * H, H)
            idx = len(adaln_w)
            adaln_w.append(w)
            adaln_b.append(b)
            return idx

        W: Dict = {}
        W["action_in"] = lin("action_in_proj", H, self.output_dim)
        # Positional encoding buffer: transfer verbatim (bit-exact), slice to T.
        pe = _param(sd, "temporal_pe.pe", tuple(sd["temporal_pe.pe"].shape), consumed)
        W["pe"] = pe[: self.trajectory_length]
        W["ts0"] = lin("timestep_encoder.mlp.0", H, 256)
        W["ts1"] = lin("timestep_encoder.mlp.2", H, H)

        blocks: List[Dict] = []
        for i in range(self.num_layers):
            p = f"blocks.{i}"
            # MHA in_proj is a single [3H, H] matrix. We keep it FUSED (one matmul
            # for Q,K,V) rather than pre-splitting — fewer kernel launches at B=1,
            # and bit-identical to torch's F.linear-then-chunk.
            sa_w = _param(sd, f"{p}.self_attn.in_proj_weight", (3 * H, H), consumed)
            sa_b = _param(sd, f"{p}.self_attn.in_proj_bias", (3 * H,), consumed)
            ca_w = _param(sd, f"{p}.cross_attn.in_proj_weight", (3 * H, H), consumed)
            ca_b = _param(sd, f"{p}.cross_attn.in_proj_bias", (3 * H,), consumed)
            blocks.append({
                "n1": add_norm(f"{p}.norm1.proj"),
                "sa_qkv": (sa_w, sa_b),                       # [3H,H] fused Q|K|V
                "sa_o": lin(f"{p}.self_attn.out_proj", H, H),
                "n2": add_norm(f"{p}.norm2.proj"),
                # Cross-attn: Q is from the action tokens, K|V from conditioning —
                # different inputs, so Q stays separate but K|V are fused (one matmul).
                "ca_q": (ca_w[0:H], ca_b[0:H]),
                "ca_kv": (ca_w[H:3 * H], ca_b[H:3 * H]),     # [2H,H] fused K|V
                "ca_o": lin(f"{p}.cross_attn.out_proj", H, H),
                "n3": add_norm(f"{p}.norm3.proj"),
                "ff0": lin(f"{p}.ffn.0", 4 * H, H),
                "ff3": lin(f"{p}.ffn.3", H, 4 * H),
            })
        W["blocks"] = blocks
        W["fn"] = add_norm("final_norm.proj")
        W["action_out"] = lin("action_out_proj", self.output_dim, H)

        # Fuse every AdaLN projection into one [(3L+1)*2H, H] weight. At step time
        # a single matmul produces all modulation params; each AdaLN slices its
        # row block by the index stored above. Math is identical to per-norm matmuls.
        W["adaln_w"] = mx.concatenate(adaln_w, axis=0)   # [(3L+1)*2H, H]
        W["adaln_b"] = mx.concatenate(adaln_b, axis=0)   # [(3L+1)*2H]

        # Fail loudly on any unmapped parameter so the two impls cannot drift.
        unmapped = set(sd.keys()) - consumed
        if unmapped:
            raise ValueError(
                f"[MLX weight transfer] {len(unmapped)} unmapped parameter(s): "
                f"{sorted(unmapped)}")
        return W

    # ------------------------------------------------------------------ #
    # Forward step (mirrors DiTMotionDecoder.forward / DiTBlock.forward)
    # ------------------------------------------------------------------ #
    def _build_tok_step(self) -> Callable:
        """One denoising step; `mod` is (mod_t1, mod_tk, is_prefix_mask)."""
        W = self._W
        H, NH, HD = self.hidden_dim, self.num_heads, self.head_dim
        attn_scale = 1.0 / math.sqrt(HD)  # torch MHA scales q by 1/sqrt(head_dim)
        mod_width = 2 * H  # each AdaLN consumes a [.., 2H] (scale|shift) slice

        def fc(wb: LinearWB, x: "mx.array") -> "mx.array":
            return x @ wb[0].T + wb[1]

        metal_adaln = self._metal_adaln

        def adaln(x, mods, idx):
            # Run AdaLN for both timesteps and select per token (avoids a [T, (3L+1)*2H] modulation).
            m1, mk, is_pre = mods
            base = idx * mod_width
            return mx.where(is_pre,
                            metal_adaln(x, m1[:, base:base + H], m1[:, base + H:base + mod_width]),
                            metal_adaln(x, mk[:, base:base + H], mk[:, base + H:base + mod_width]))

        def heads(t: "mx.array", B: int, L: int) -> "mx.array":
            return t.reshape(B, L, NH, HD).transpose(0, 2, 1, 3)

        def step(traj, ck_all, cv_all, mod):
            # ck_all/cv_all [L,B,NH,N,HD]: cross-attn K/V, precomputed per call (cond only).
            B, T = traj.shape[0], traj.shape[1]
            x = fc(W["action_in"], traj) + W["pe"][:T]
            for j, bw in enumerate(W["blocks"]):
                nrm = adaln(x, mod, bw["n1"])
                q, k, v = mx.split(fc(bw["sa_qkv"], nrm), 3, axis=-1)
                o = mx.fast.scaled_dot_product_attention(
                    heads(q, B, T), heads(k, B, T), heads(v, B, T), scale=attn_scale)
                x = x + fc(bw["sa_o"], o.transpose(0, 2, 1, 3).reshape(B, T, H))
                nrm = adaln(x, mod, bw["n2"])
                co = mx.fast.scaled_dot_product_attention(
                    heads(fc(bw["ca_q"], nrm), B, T), ck_all[j], cv_all[j], scale=attn_scale)
                x = x + fc(bw["ca_o"], co.transpose(0, 2, 1, 3).reshape(B, T, H))
                nrm = adaln(x, mod, bw["n3"])
                # exact GELU, matching torch nn.GELU default
                x = x + fc(bw["ff3"], mxnn.gelu(fc(bw["ff0"], nrm)))
            x = adaln(x, mod, W["fn"])
            return fc(W["action_out"], x)                       # velocity [B,T,D]

        return step

    # ------------------------------------------------------------------ #
    # Loop-invariant precompute (schedule-only `mod`; cond K/V)
    # ------------------------------------------------------------------ #
    def _mod_from_tau(self, tau_value: float, B: int) -> "mx.array":
        """Timestep embedding + fused AdaLN projection for one tau. Depends only
        on the schedule (and frozen weights), never on the input — so it is a
        per-(tau, model) constant."""
        W = self._W
        tau = mx.full((B,), tau_value, dtype=mx.float32)
        a = tau[:, None] * self._freqs
        emb = mx.concatenate([mx.cos(a), mx.sin(a)], axis=-1)
        t_emb = mxnn.silu(emb @ W["ts0"][0].T + W["ts0"][1])
        t_emb = t_emb @ W["ts1"][0].T + W["ts1"][1]
        return t_emb @ W["adaln_w"].T + W["adaln_b"]            # [B, (3L+1)*2H]

    def _precompute_mods(self, num_steps: int, B: int) -> "mx.array":
        """All per-step modulations, stacked [num_steps, B, (3L+1)*2H], cached
        across generations (the denoising schedule is fixed)."""
        key = (num_steps, B)
        cached = self._mod_cache.get(key)
        if cached is not None:
            return cached
        mods = mx.stack([self._mod_from_tau(k / num_steps, B)
                         for k in range(num_steps)], axis=0)
        mx.eval(mods)  # materialize once; reused on every subsequent call
        self._mod_cache[key] = mods
        return mods

    def _precompute_cond_kv(self, cond: "mx.array") -> Tuple["mx.array", "mx.array"]:
        """Per-layer cross-attn K/V in head form, [L, B, NH, N, HD]. Constant
        across Euler steps (depends only on the conditioning), so computed once
        per predict() call instead of every step."""
        W = self._W
        B, N = cond.shape[0], cond.shape[1]
        NH, HD = self.num_heads, self.head_dim
        cks, cvs = [], []
        for bw in W["blocks"]:
            kv = cond @ bw["ca_kv"][0].T + bw["ca_kv"][1]       # [B,N,2H]
            ck, cv = mx.split(kv, 2, axis=-1)
            cks.append(ck.reshape(B, N, NH, HD).transpose(0, 2, 1, 3))
            cvs.append(cv.reshape(B, N, NH, HD).transpose(0, 2, 1, 3))
        return mx.stack(cks, axis=0), mx.stack(cvs, axis=0)

    # ------------------------------------------------------------------ #
    # Public: inpaint Euler loop
    # ------------------------------------------------------------------ #
    def inpaint_euler(self, traj0: torch.Tensor, conditioning: torch.Tensor,
                      num_steps: int, prefix: torch.Tensor, P: int) -> torch.Tensor:
        """Inpaint Euler loop: prefix frames clean (t=1), hard-fixed every step.

        `traj0` must already have `traj0[:, :P] = prefix` (torch-seeded, as in the torch path).
        Executes on the private MLX thread regardless of the calling thread."""
        return self._run(self._inpaint_impl, traj0, conditioning, num_steps, prefix, P)

    def _inpaint_impl(self, traj0, conditioning, num_steps, prefix, P):
        device, dtype = traj0.device, traj0.dtype
        traj = _to_mx(traj0)
        cond = _to_mx(conditioning)
        pre = _to_mx(prefix)
        B, T = traj.shape[0], traj.shape[1]
        dt = 1.0 / num_steps
        with mx.stream(self._stream):
            ck_all, cv_all = self._precompute_cond_kv(cond)
            mods = self._precompute_mods(num_steps, B)              # [N, B, M]
            if B not in self._mod_one:
                m1 = self._mod_from_tau(1.0, B); mx.eval(m1); self._mod_one[B] = m1
            m1 = self._mod_one[B]
            is_pre = (mx.arange(T) < P)[None, :, None]              # [1, T, 1]
            for k in range(num_steps):
                v = self._tok_step(traj, ck_all, cv_all, (m1, mods[k], is_pre))
                traj = traj + dt * v
                traj = mx.concatenate([pre, traj[:, P:]], axis=1)   # hard-fix prefix
            mx.eval(traj)  # single materialization point for the entire loop
        return _to_torch(traj, device, dtype)
