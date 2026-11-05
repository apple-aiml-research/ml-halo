#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Unitree G1 (29 DOF) motion-tracking deployment.
"""

# Make sibling modules in this directory importable as top-level (e.g.
# `from g1_controller import G1Controller` inside deploy_real.py), the same way
# g1_controller.py puts tracker/deploy on sys.path for `from utils ...`. The
# scripts then work both when run directly and when imported as
# `tracker.deploy.g1.*`.
import sys as _sys
from pathlib import Path as _Path
_g1_dir = str(_Path(__file__).resolve().parent)
if _g1_dir not in _sys.path:
    _sys.path.insert(0, _g1_dir)

from . import g1_controller as _g1_controller
# deploy_real.py (and the other scripts) import the sibling top-level as `g1_controller`. Alias it to this
# package's module, so the package and the scripts share one G1Controller class instead of loading the file
# twice (two classes: G1RealController would not subclass the G1Controller exported here).
_sys.modules.setdefault("g1_controller", _g1_controller)
from .g1_controller import G1Controller  # noqa: E402
from .deploy_real import G1RealController  # noqa: E402

__all__ = ["G1Controller", "G1RealController"]