#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Math utilities and data types for robot deployment.
"""

from .types import (
    RobotState,
    MotorCommand,
    ControllerConfig,
    build_index_map,
)
from .math_utils import (
    quat_mul,
    quat_difference,
    axis_angle_from_quat,
    quat_slerp,
    get_gravity_orientation,
    quat_to_rotation_matrix,
    quat_to_euler,
    pd_control,
    so3_derivative,
)

__all__ = [
    "ControllerConfig",
    "MotorCommand",
    "RobotState",
    "axis_angle_from_quat",
    "build_index_map",
    "get_gravity_orientation",
    "pd_control",
    "quat_difference",
    "quat_mul",
    "quat_slerp",
    "quat_to_euler",
    "quat_to_rotation_matrix",
    "so3_derivative",
]
