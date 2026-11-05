#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Backend implementations for robot deployment.

MuJoCo simulation and the Unitree G1 (unitree_sdk2py) backends. The Unitree backend
module imports without unitree_sdk2py; UnitreeBackend then raises at construction.
"""

from .mujoco_backend import SimulatorBackend
from .unitree_backend import UnitreeBackend

__all__ = [
    "SimulatorBackend",
    "UnitreeBackend",
]
