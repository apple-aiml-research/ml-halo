#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Builds on unitree_rl_gym (Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD., BSD-3-Clause).

"""
Unitree remote controller and low-level command helpers.

- RemoteController: parses the ``wireless_remote`` bytes of the G1 LowState.
- init_cmd_hg / create_damping_cmd: LowCmd initialization helpers.
"""

import struct


class KeyMap:
    """Standard key mapping for Unitree remote controllers."""
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    """Unitree remote controller state, parsed from LowState ``wireless_remote`` bytes."""

    def __init__(self):
        """Initialize remote controller state."""
        self.button = [0] * 16

    def set(self, data: bytes):
        """Set controller state from Unitree wireless_remote data.

        Args:
            data: Raw bytes from low_state.wireless_remote
        """
        # Parse button states (16 buttons packed into 2 bytes)
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i


# ============= Motor Control Helpers =============

class MotorMode:
    """Ankle/waist motor control modes of the Unitree HG LowCmd (``mode_pr``)."""
    PR = 0  # Series Control for Pitch/Roll Joints (the mode this deployment uses)


def _set_cmd_gains(cmd, kp: float = 0, kd: float = 0):
    """Set all motor commands to specified gains with zero targets.

    Args:
        cmd: Unitree LowCmd message
        kp: Position gain for all motors
        kd: Velocity gain for all motors
    """
    for i in range(len(cmd.motor_cmd)):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].dq = 0
        cmd.motor_cmd[i].kp = kp
        cmd.motor_cmd[i].kd = kd
        cmd.motor_cmd[i].tau = 0


def create_damping_cmd(cmd):
    """Set command to damping mode for safe shutdown.

    Args:
        cmd: Unitree LowCmd message (LowCmdHG)
    """
    _set_cmd_gains(cmd, kp=0, kd=8)


def init_cmd_hg(cmd, mode_machine: int, mode_pr: int):
    """Initialize HG (G1 humanoid) command structure.

    Args:
        cmd: Unitree HG LowCmd message
        mode_machine: Machine mode
        mode_pr: Ankle/waist motor mode (MotorMode.PR)
    """
    cmd.mode_machine = mode_machine
    cmd.mode_pr = mode_pr
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].mode = 1
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].dq = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0
