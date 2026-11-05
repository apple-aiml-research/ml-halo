#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Checker identity / version — stamped on every MotionQualityReport so results"""

from __future__ import annotations

import hashlib
from dataclasses import fields, is_dataclass

CHECKER_VERSION = "0.8.0"


def config_digest(cfg) -> str:
    if cfg is None or not is_dataclass(cfg):
        return "none"
    parts = []
    for f in sorted(fields(cfg), key=lambda x: x.name):
        v = getattr(cfg, f.name)
        if isinstance(v, dict):
            v = sorted(v.items())
        parts.append(f"{f.name}={v!r}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]


def checks_digest(checkers) -> str:
    sig = ";".join(f"{c.name}:{c.severity}" for c in (checkers or []))
    return hashlib.sha1(sig.encode()).hexdigest()[:10]


def checker_id(cfg, checkers) -> str:
    return f"mqc-{CHECKER_VERSION}+cfg.{config_digest(cfg)}+chk.{checks_digest(checkers)}"
