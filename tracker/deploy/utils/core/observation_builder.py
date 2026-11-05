#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Config-driven observation builder for G1 deploy controllers.

The observation is declared in YAML (the ``observation:`` block of the robot
config) as an ordered list of *terms* grouped into *groups* (one group per ONNX
input). It mirrors the observation groups of the training env; the C++ deployment
parses the same YAML block.

Two term kinds:
  * STATE terms   — per-frame robot signals (``STATE_TERM_DIMS``). The controller
                    computes the current-frame value and passes it in ``frame_values``;
                    the builder buffers it for stack groups.
  * REFERENCE terms — command signals sampled from the loaded motion over a set of
                    time indices (tracking_command_partial); never buffered.

Group layouts:
  * ``flat``  — concatenate each term for the current frame into a 1-D vector.
                A reference term may set ``horizon: K``: the current reference frame
                and the next K-1 frames, in the order t+K-1, ..., t+1, t (current
                frame last), flattened ``term_major``.
  * ``stack`` — (history, feature): state terms come from their history buffer,
                reference terms are sampled over the full ``time_indices`` window.
  * ``image`` — one depth frame shaped ``(1, 1, H, W)``. The raw metric frame is handed
                in via ``set_depth_from(...)`` and encoded ``clip([min,max]) -> log1p``.
                Missing frame => far-plane fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .history_buffer import HistoryBuffer

_GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def quat_rotate_inverse_batch(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vectors by the inverse of quaternions (wxyz). (N,4),(N,3) -> (N,3)."""
    w = quat_wxyz[:, 0]
    qx = -quat_wxyz[:, 1]
    qy = -quat_wxyz[:, 2]
    qz = -quat_wxyz[:, 3]
    uv_x = qy * vec[:, 2] - qz * vec[:, 1]
    uv_y = qz * vec[:, 0] - qx * vec[:, 2]
    uv_z = qx * vec[:, 1] - qy * vec[:, 0]
    uuv_x = qy * uv_z - qz * uv_y
    uuv_y = qz * uv_x - qx * uv_z
    uuv_z = qx * uv_y - qy * uv_x
    return np.stack([
        vec[:, 0] + 2.0 * (w * uv_x + uuv_x),
        vec[:, 1] + 2.0 * (w * uv_y + uuv_y),
        vec[:, 2] + 2.0 * (w * uv_z + uuv_z),
    ], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Reference-term registry: name -> (fn(motion_loader, indices) -> (T, D), subfields)
# `subfields` gives the internal (name, dim) layout so `term_major` flattening
# can reproduce the training action_manager order.
# --------------------------------------------------------------------------- #
@dataclass
class RefTerm:
    fn: Callable
    subfields: list[tuple[str, int]]


def _at(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Sample ``arr`` at ``indices``. The clip guards a concurrent motion swap: the loader's arrays can
    shrink under us mid-tick, and an out-of-range index would raise on the control path."""
    return arr[np.clip(indices, 0, arr.shape[0] - 1)]


def _tracking_command_partial(ml, indices: np.ndarray) -> np.ndarray:
    """Per-frame [joint_pos(29), joint_vel(29), gravity_b(3), lin_vel_b(3),
    ang_vel_b(3)] = 67, body-frame fields via quat_rotate_inverse."""
    jp = _at(ml.motion_dof_poss, indices)
    jv = _at(ml.motion_dof_vels, indices)
    rots = _at(ml.motion_base_rots, indices)
    lv_w = _at(ml.motion_base_lin_vels, indices)
    av_w = _at(ml.motion_base_ang_vels, indices)
    n = len(rots)
    gravity_b = quat_rotate_inverse_batch(rots, np.broadcast_to(_GRAVITY_WORLD, (n, 3)).copy())
    lv_b = quat_rotate_inverse_batch(rots, lv_w)
    av_b = quat_rotate_inverse_batch(rots, av_w)
    return np.concatenate([jp, jv, gravity_b, lv_b, av_b], axis=-1).astype(np.float32)


REF_TERMS: dict[str, RefTerm] = {
    "tracking_command_partial": RefTerm(
        fn=_tracking_command_partial,
        subfields=[("joint_pos", 29), ("joint_vel", 29),
                   ("root_gravity_b", 3), ("root_lin_vel_b", 3), ("root_ang_vel_b", 3)],
    ),
}

# State terms the controller provides each tick, with their per-frame size (same set as the C++
# state_term_dim). A term name in neither table is rejected when the spec is parsed.
STATE_TERM_DIMS: dict[str, int] = {
    "base_ang_vel": 3,
    "proj_gravity": 3,
    "joint_pos_rel": 29,
    "joint_vel_rel": 29,
    "last_action": 29,
    "motion_anchor_ori_b": 6,
}


# --------------------------------------------------------------------------- #
# Spec dataclasses (parsed from the YAML `observation:` block)
# --------------------------------------------------------------------------- #
@dataclass
class TermCfg:
    name: str
    horizon: Optional[int] = None          # reference terms in flat groups

    @property
    def is_reference(self) -> bool:
        return self.name in REF_TERMS


@dataclass
class GroupCfg:
    name: str
    layout: str                            # flat | stack | image
    terms: list[TermCfg]                   # [] for image groups
    history: Optional[int] = None          # stack groups
    # image groups: one depth frame (1, 1, height, width), fed in via ObservationBuilder.set_depth_from()
    height: Optional[int] = None
    width: Optional[int] = None
    clip: Optional[tuple] = None           # (min, max) meters — encode clip range
    encode: Optional[str] = None           # "log1p" (clip FIRST, then log1p)
    obs_decim: int = 1                     # control steps each frame is held for in sim (see depth_hold_steps)


@dataclass
class ObservationSpec:
    history_length: int
    groups: dict[str, GroupCfg] = field(default_factory=dict)
    # Robot attitude for motion_anchor_ori_b and the reference rebase: the odometry attitude. "odom" is the
    # only value; the key is required so the config states it, as in the C++ ObsSpec.
    anchor_mode: str = "odom"

    @staticmethod
    def _term(t) -> TermCfg:
        if isinstance(t, str):
            term = TermCfg(name=t)
        else:
            flatten = t.get("flatten", "term_major")
            if flatten != "term_major":
                raise ValueError(f"term {t['name']!r}: flatten must be term_major, got {flatten!r}")
            term = TermCfg(name=t["name"], horizon=t.get("horizon"))
        if term.name not in STATE_TERM_DIMS and term.name not in REF_TERMS:
            raise ValueError(f"unknown observation term {term.name!r}; known: "
                             f"{sorted(STATE_TERM_DIMS) + sorted(REF_TERMS)}")
        return term

    @staticmethod
    def _image_obs_decim(gname: str, g: dict) -> int:
        """Validate `obs_decim` (default 1); invalid values are rejected rather than clamped."""
        raw = g.get("obs_decim")
        if raw is None:
            return 1
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"image group {gname!r}: obs_decim must be a positive int (control steps to "
                             f"hold each frame), got {raw!r}")
        if raw < 1:
            raise ValueError(f"image group {gname!r}: obs_decim must be >= 1, got {raw}. 1 = a fresh "
                             f"frame every control step; 2 = 25 Hz, the rate the tracker trains depth at.")
        return raw

    @staticmethod
    def _image_clip(gname: str, g: dict) -> tuple[float, float]:
        """Validate and return an image group's ``(near, far)`` clip, in metres.

        `clip` is mandatory, with no default: it must equal the clip the policy trained with. The value
        itself is checked against the training clip by variant_support.check_depth_clip().
        """
        if "clip" not in g:
            raise ValueError(
                f"image group {gname!r} must set an explicit `clip: [min, max]` (meters), equal "
                f"to the clip its checkpoint trained with (0.2/2.5 for the tracker's training "
                f"DEPTH_CLIP). There is deliberately no default.")
        clip = g["clip"]
        try:
            near, far = float(clip[0]), float(clip[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(
                f"image group {gname!r}: clip must be [min, max] in meters, got {clip!r}") from exc
        if not (np.isfinite(near) and np.isfinite(far) and 0.0 < near < far):
            raise ValueError(
                f"image group {gname!r}: clip must satisfy 0 < near < far (finite), got [{near}, {far}]. "
                f"A non-positive near makes log1p meaningless and a far <= near collapses the "
                f"encoded range to zero.")
        return near, far

    @staticmethod
    def _image_geometry(gname: str, g: dict) -> tuple[int, int]:
        """``(height, width)`` of an image group."""
        if "height" not in g or "width" not in g:
            raise ValueError(f"image group {gname!r} needs height and width")
        return int(g["height"]), int(g["width"])

    @staticmethod
    def _image_encode(gname: str, g: dict) -> str:
        """Only ``log1p`` (the training encode) is accepted; omitting the key means log1p."""
        encode = g.get("encode", "log1p")
        if encode != "log1p":
            raise ValueError(f"image group {gname!r}: only encode: log1p is supported, got {encode!r}")
        return encode

    @classmethod
    def from_dict(cls, d: dict) -> "ObservationSpec":
        if "history_length" not in d:
            raise ValueError("observation.history_length is required")
        hl = int(d["history_length"])
        anchor = d.get("anchor_mode")
        if anchor != "odom":
            raise ValueError(f"observation.anchor_mode is required and must be 'odom', got {anchor!r}")
        groups: dict[str, GroupCfg] = {}
        for gname, g in d["groups"].items():
            layout = g.get("layout")
            if layout == "image":
                height, width = cls._image_geometry(gname, g)
                groups[gname] = GroupCfg(
                    name=gname, layout="image", terms=[],
                    height=height, width=width,
                    clip=cls._image_clip(gname, g),
                    encode=cls._image_encode(gname, g),
                    obs_decim=cls._image_obs_decim(gname, g),
                )
            elif layout in ("flat", "stack"):
                groups[gname] = GroupCfg(
                    name=gname,
                    layout=layout,
                    history=g.get("history", hl),
                    terms=[cls._term(t) for t in g["terms"]],
                )
            else:
                raise ValueError(f"group {gname!r}: `layout` is required (flat|stack|image), got {layout!r}")
        return cls(history_length=hl, groups=groups, anchor_mode=anchor)


def depth_hold_steps(spec: ObservationSpec, is_simulation: bool) -> int:
    """Control steps each depth frame is held for (1 = a fresh frame every control step).

    Sim: the image group's ``obs_decim`` (2 => 25 Hz, the rate the tracker trains depth at); the renderer is
    synchronous at the control rate and would otherwise feed depth faster than in training. Hardware: always 1;
    the camera is asynchronous and slower than the control loop, so the policy takes the freshest frame and a
    hold would only add latency. 1 without an image group.
    """
    g = next((g for g in spec.groups.values() if g.layout == "image"), None)
    return g.obs_decim if (is_simulation and g is not None) else 1


def _flatten_reference(cmd: np.ndarray, subfields: list[tuple[str, int]]) -> np.ndarray:
    """Flatten a (T, D) reference block term_major: [sub0 over T | sub1 over T | ...]."""
    parts, off = [], 0
    for _, dim in subfields:
        parts.append(cmd[:, off:off + dim].reshape(-1))
        off += dim
    return np.concatenate(parts).astype(np.float32)


class ObservationBuilder:
    """Assembles ONNX feed tensors from an ObservationSpec.

    History buffers for the state terms of stack groups are allocated on the first
    build from the ``frame_values`` shapes. Call ``update_and_build`` once per control tick.
    ``depth_hold_steps`` is the value of ``depth_hold_steps(spec, is_simulation)``.
    """

    def __init__(self, spec: ObservationSpec, depth_hold_steps: int = 1):
        self.spec = spec
        self._buffers: dict[str, HistoryBuffer] = {}
        self._depth: Optional[np.ndarray] = None          # metric frame fed to the image group
        self._image_group = next((g for g in spec.groups.values() if g.layout == "image"), None)
        self._depth_step = 0                              # set_depth_from calls, for the frame hold
        self.depth_hold_steps = max(1, int(depth_hold_steps))
        self._buffered_terms = self._collect_buffered_terms()

    def _collect_buffered_terms(self) -> dict[str, int]:
        """term_name -> history length required (max across stack groups)."""
        need: dict[str, int] = {}
        for g in self.spec.groups.values():
            if g.layout != "stack":
                continue
            for t in g.terms:
                if not t.is_reference:
                    need[t.name] = max(need.get(t.name, 0), g.history)
        return need

    def set_depth_from(self, get_frame) -> bool:
        """Hand the image group a metric-depth frame (H, W float32 meters) from ``get_frame()``.

        Called once per policy tick before ``update_and_build``. With ``depth_hold_steps > 1`` the
        previous frame is kept on the held steps and ``get_frame`` is not called (in sim it is a
        render). The frame is stored raw; the clip + log1p encode happens in ``_build_image``.
        Returns True when a fresh frame was taken.
        """
        if self.depth_hold_steps > 1:
            held = self._depth_step % self.depth_hold_steps != 0 and self._depth is not None
            self._depth_step += 1
            if held:
                return False
        self._depth = np.asarray(get_frame(), dtype=np.float32)
        return True

    def get_depth(self) -> Optional[np.ndarray]:
        """The metric frame currently fed to the image group (the held frame), or None."""
        return self._depth

    def has_image_group(self) -> bool:
        """True if any group uses the ``image`` layout (i.e. depth is required)."""
        return self._image_group is not None

    def update_and_build(self, frame_values: dict[str, np.ndarray],
                         time_indices: np.ndarray, motion_loader) -> dict[str, np.ndarray]:
        for name, hlen in self._buffered_terms.items():
            v = np.asarray(frame_values[name], dtype=np.float32)
            if name not in self._buffers:
                self._buffers[name] = HistoryBuffer(hlen, v.shape[-1])
            self._buffers[name].add(v)

        out: dict[str, np.ndarray] = {}
        for gname, g in self.spec.groups.items():
            if g.layout == "flat":
                out[gname] = self._build_flat(g, frame_values, time_indices, motion_loader)
            elif g.layout == "stack":
                out[gname] = self._build_stack(g, time_indices, motion_loader)
            else:
                out[gname] = self._build_image(g)
        return out

    def _build_flat(self, g, frame_values, time_indices, motion_loader) -> np.ndarray:
        parts = []
        for t in g.terms:
            if t.is_reference:
                ref = REF_TERMS[t.name]
                k = t.horizon or 1
                cmd = ref.fn(motion_loader, time_indices[-k:])   # (k, D): frames t+k-1 .. t
                parts.append(_flatten_reference(cmd, ref.subfields))
            else:
                parts.append(np.asarray(frame_values[t.name], dtype=np.float32).ravel())
        return np.concatenate(parts).reshape(1, -1).astype(np.float32)

    def _build_stack(self, g, time_indices, motion_loader) -> np.ndarray:
        parts = []
        for t in g.terms:
            if t.is_reference:
                parts.append(REF_TERMS[t.name].fn(motion_loader, time_indices))  # (h, D)
            else:
                parts.append(self._buffers[t.name].get_history())
        feat = np.concatenate(parts, axis=-1)               # (h, F)
        return feat.reshape(1, g.history, feat.shape[-1]).astype(np.float32)

    def _build_image(self, g) -> np.ndarray:
        """Build the (1, 1, H, W) depth tensor: clip([min, max] m) then log1p, as in training.

        A missing frame yields the far plane (clip max), so the group is always emitted;
        non-finite and non-positive pixels map to far (training's inf/nan -> far).
        """
        clip_min, clip_max = g.clip
        if self._depth is None:
            img = np.full((g.height, g.width), float(clip_max), dtype=np.float32)
        else:
            img = np.where(np.isfinite(self._depth) & (self._depth > 1e-6), self._depth, float(clip_max))
        img = np.log1p(np.clip(img, float(clip_min), float(clip_max)))
        return img.astype(np.float32).reshape(1, 1, g.height, g.width)
