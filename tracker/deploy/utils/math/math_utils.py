#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Math Utilities for Robot Deployment.

Numpy-based math functions for quaternion operations, rotations, and control.
"""

import numpy as np


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions together.

    Args:
        q1: First quaternion in (w, x, y, z). Shape is (..., 4).
        q2: Second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        Product quaternion in (w, x, y, z). Shape is (..., 4).
    """
    if q1.shape != q2.shape:
        raise ValueError(f"Quaternion shape mismatch: {q1.shape} != {q2.shape}")

    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)

    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.stack([w, x, y, z], axis=-1).reshape(shape)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute conjugate of a quaternion.

    Args:
        q: Quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        Conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate([q[:, 0:1], -q[:, 1:]], axis=-1).reshape(shape)


def quat_difference(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compute relative rotation from q1 to q2.

    Returns q_rel such that q2 = q1 * q_rel.

    Args:
        q1: First quaternion in (w, x, y, z).
        q2: Second quaternion in (w, x, y, z).

    Returns:
        Relative quaternion representing rotation from q1 to q2.
    """
    q1_conj = quat_conjugate(q1)
    return quat_mul(q1_conj, q2)


def axis_angle_from_quat(quat: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Convert quaternion to axis-angle representation.

    Args:
        quat: Quaternion in (w, x, y, z). Shape is (..., 4).
        eps: Tolerance for Taylor approximation.

    Returns:
        Axis-angle vector. Shape is (..., 3).
    """
    # Ensure positive real part
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))

    mag = np.linalg.norm(quat[..., 1:], axis=-1)
    half_angle = np.arctan2(mag, quat[..., 0])
    angle = 2.0 * half_angle

    # Use safe_angle to avoid 0/0 in the division; np.where evaluates both
    # branches eagerly, so the divide would produce NaN even for masked elements.
    safe_angle = np.where(np.abs(angle) > eps, angle, 1.0)
    sin_half_over_angle = np.where(
        np.abs(angle) > eps,
        np.sin(half_angle) / safe_angle,
        0.5 - angle * angle / 48
    )

    return quat[..., 1:4] / sin_half_over_angle[..., np.newaxis]


def quat_slerp(q1: np.ndarray, q2: np.ndarray, tau: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions.

    Args:
        q1: First quaternion (w, x, y, z).
        q2: Second quaternion (w, x, y, z).
        tau: Interpolation coefficient [0, 1].

    Returns:
        Interpolated quaternion.
    """
    # Renormalize inputs. The slerp formula below assumes UNIT quaternions
    # (it uses dot == cos(angle)); feeding it slightly non-unit quats makes
    # the output norm shrink further than its inputs, which compounds across
    # repeated interpolation (e.g. the RTC trim→resend loop) into a runaway
    # collapse toward a zero-norm quaternion. Normalizing here stops that.
    n1 = np.linalg.norm(q1)
    n2 = np.linalg.norm(q2)
    if n1 < 1e-8 or n2 < 1e-8:
        # Degenerate input — nothing meaningful to interpolate; return the
        # valid operand (or identity if both are degenerate).
        if n1 >= 1e-8:
            return (q1 / n1).astype(q1.dtype)
        if n2 >= 1e-8:
            return (q2 / n2).astype(q2.dtype)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=q1.dtype)
    q1 = (q1 / n1).astype(q1.dtype)
    q2 = (q2 / n2).astype(q2.dtype)

    if tau == 0.0:
        return q1.copy()
    elif tau == 1.0:
        return q2.copy()

    d = np.dot(q1, q2)
    if abs(abs(d) - 1.0) < np.finfo(q1.dtype).eps * 4.0:
        return q1.copy()

    q2_adj = q2.copy()
    if d < 0.0:
        d = -d
        q2_adj *= -1.0

    angle = np.arccos(np.clip(d, -1, 1))
    if abs(angle) < np.finfo(q1.dtype).eps * 4.0:
        return q1.copy()

    isin = 1.0 / np.sin(angle)
    q1_weighted = q1 * np.sin((1.0 - tau) * angle) * isin
    q2_weighted = q2_adj * np.sin(tau * angle) * isin

    return q1_weighted + q2_weighted


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion to gravity vector in body frame.

    Args:
        quaternion: Quaternion in (w, x, y, z) format.

    Returns:
        Gravity vector [gx, gy, gz] in body frame.
    """
    qw, qx, qy, qz = quaternion
    gravity = np.zeros(3, dtype=np.float32)
    gravity[0] = 2 * (-qz * qx + qw * qy)
    gravity[1] = -2 * (qz * qy + qw * qx)
    gravity[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity


def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix.

    Args:
        quat: Quaternion in (w, x, y, z) format.

    Returns:
        3x3 rotation matrix.
    """
    qw, qx, qy, qz = quat
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ], dtype=np.float32)


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat: Quaternion in (w, x, y, z) format.

    Returns:
        Euler angles [roll, pitch, yaw] in radians.
    """
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray
) -> np.ndarray:
    """PD control law for calculating torques.

    Args:
        target_q: Target joint positions.
        q: Current joint positions.
        kp: Proportional gains.
        target_dq: Target joint velocities.
        dq: Current joint velocities.
        kd: Derivative gains.

    Returns:
        Control torques.
    """
    return (target_q - q) * kp + (target_dq - dq) * kd


def so3_derivative(rotations: np.ndarray, dt: float) -> np.ndarray:
    """Compute derivative of SO3 rotation sequence.

    Central difference ``axis_angle(q[i+1] * q[i-1]^-1) / (2 dt)`` on interior frames, as the training NPZ
    converter (``data/scripts/extract_sequences.py``) computes the reference root angular velocity, and one-sided
    differences on the first / last frame (the ``np.gradient`` edge rule the linear velocities use).

    Args:
        rotations: Quaternion sequence (N, 4) in wxyz format.
        dt: Time step between frames.

    Returns:
        Angular velocities (N, 3).
    """
    n = rotations.shape[0]
    omega = np.zeros((n, 3), dtype=np.float32)
    if n < 2:
        return omega

    def _rate(q_a: np.ndarray, q_b: np.ndarray, span: float) -> np.ndarray:
        return axis_angle_from_quat(quat_mul(q_b, quat_conjugate(q_a))) / span

    omega[0] = _rate(rotations[0:1], rotations[1:2], dt)[0]
    omega[-1] = _rate(rotations[-2:-1], rotations[-1:], dt)[0]
    if n > 2:
        omega[1:-1] = _rate(rotations[:-2], rotations[2:], 2.0 * dt)
    return omega