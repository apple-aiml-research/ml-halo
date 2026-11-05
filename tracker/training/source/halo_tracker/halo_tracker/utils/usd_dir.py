#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Output directory for Isaac Lab's URDF -> USD conversions."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


def converted_usd_dir(asset_path: str) -> str:
    """Per-process, per-asset directory for the converted USD of ``asset_path``.

    Isaac Lab's default is ``/tmp/IsaacLab/usd_<second>_<random>``, and the env seeds ``random`` before it builds the
    scene, so trainings launched in the same second with the same seed convert into the SAME directory and corrupt
    each other's USD (the robot then comes up with no rigid bodies). Keying the directory on the process id and the
    asset path keeps concurrent runs apart. The directories stay behind after a run (a few MB each, like Isaac Lab's
    own); delete ``<tmp>/IsaacLab`` to reclaim them.
    """
    tag = hashlib.md5(os.path.abspath(asset_path).encode()).hexdigest()[:8]
    return os.path.join(tempfile.gettempdir(), "IsaacLab", f"halo_{os.getpid()}", f"{Path(asset_path).stem}_{tag}")
