#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local sensors for the G1 motion-tracking tasks.

Ray-cast (warp) perception that also runs under multi-GPU training:

  * :class:`MultiMeshRayCasterMGPU`: the privileged height scanner of the object scenes, IsaacLab's
    ``MultiMeshRayCaster`` with the dynamic push box posed from per-rank PhysX views.
  * :class:`RPLDepthCamera`: the student's depth camera (RPL Algorithm 1), the scene targets plus the robot's own
    body-link meshes (self-occlusion).
"""

from .mgpu_ray_caster import MultiMeshRayCasterMGPU
from .rpl_depth_camera import RPLDepthCamera, RPLDepthCameraCfg

__all__ = [
    "MultiMeshRayCasterMGPU",
    "RPLDepthCamera",
    "RPLDepthCameraCfg",
]
