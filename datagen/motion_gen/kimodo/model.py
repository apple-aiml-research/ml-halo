#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Load the Kimodo G1 model once, plus the helpers every generator shares."""

import os

import torch

os.environ.setdefault("TEXT_ENCODER_MODE", "local")

from kimodo.model import load_model  # noqa: E402
from kimodo.skeleton import G1Skeleton34  # noqa: E402
from kimodo.exports.mujoco import MujocoQposConverter  # noqa: E402
from motion_gen.kimodo.walk import _g1_standing_keyframe  # noqa: E402

_SKEL_CONV = None


def skel_conv_singleton():
    """CPU skeleton for the converter, separate from the constraint skeleton so that"""
    global _SKEL_CONV
    if _SKEL_CONV is None:
        _SKEL_CONV = G1Skeleton34()
    return _SKEL_CONV


def load_model_and_skel(dev=None):
    """Load Kimodo and build the skeletons / converter / standing pose once."""
    dev = dev or ("mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = load_model("Kimodo-G1-RP-v1", device=dev)
    skel = G1Skeleton34()  # for constraints (moved to device)
    converter = MujocoQposConverter(skel_conv_singleton())  # CPU seat-pose converter
    standing = _g1_standing_keyframe(skel, model=model, device=dev)
    return model, skel, converter, standing, dev


class CachedTextEncoder:
    """Wraps Kimodo's text_encoder to encode each distinct prompt once and broadcast"""

    def __init__(self, enc):
        self._enc = enc
        self._cache = {}

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        uniq = set(texts)
        if len(uniq) == 1:
            t = texts[0]
            if t not in self._cache:
                self._cache[t] = self._enc([t])  # (feat[1,1,D] tensor, len list[1])
            f, l = self._cache[t]
            B = len(texts)
            f_b = f.repeat(B, *([1] * (f.dim() - 1)))
            l_b = (list(l) * B) if isinstance(l, list) else l.repeat(B)
            return f_b, l_b
        return self._enc(texts)  # mixed batch -> uncached

    def __getattr__(self, name):
        return getattr(self._enc, name)
