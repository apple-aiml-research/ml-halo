#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Abstract base controller for the G1 deployment.

Shared by simulation (MuJoCo) and real-robot deployment. Handles the common
initialization from the config (joint index maps, policy loading, gains and
torque limits), torque-aware action clipping, the non-finite guards and the
control-loop timing.
"""

import time
from abc import ABC, abstractmethod
from enum import IntEnum
import numpy as np

from ..math import ControllerConfig, MotorCommand, build_index_map
from .policy_runner import PolicyRunner


class ControlState(IntEnum):
    """Real-robot operating states."""
    INIT = 0
    DEFAULT_POS = 2
    CONTROL = 10


def all_finite(*values) -> bool:
    """True if every entry of every array (or of every array in a dict) is finite."""
    arrays = [a for v in values for a in (v.values() if isinstance(v, dict) else (v,))]
    return all(bool(np.all(np.isfinite(np.asarray(a)))) for a in arrays)


class BaseController(ABC):
    """Abstract base controller.

    Provides:
    - Config-based initialization (joint maps, gains, torque limits)
    - Policy loading and inference
    - Torque-aware action clipping
    - Control loop structure, timing and statistics

    Subclasses implement observation building (``get_observation``).

    Non-finite guards (same rules as the C++ control loop): a non-finite observation
    or policy action is replaced by a zero action, which holds the default pose
    (torque-clipped) until the policy output is finite again; a non-finite motor
    target latches damping (``latch_damping``) and stops the run.
    """

    # Safety state. Assigned on the instance only when it changes, so the class holds the initial values.
    fault = None               # first latched safety fault, or None
    _nonfinite_active = False  # a non-finite observation/action is being replaced this tick

    def __init__(self, config: ControllerConfig, policy_path: str, backend=None):
        """Initialize base controller.

        Args:
            config: Controller configuration
            policy_path: Path to ONNX policy file
            backend: Robot backend (SimulatorBackend or UnitreeBackend)
        """
        self.config = config
        self.backend = backend

        # Build joint index maps (training <-> deployment)
        self.deploy_to_train_map = build_index_map(
            config.deployment_joint_order,
            config.training_joint_order
        )
        self.train_to_deploy_map = build_index_map(
            config.training_joint_order,
            config.deployment_joint_order
        )

        self.policy = PolicyRunner(
            policy_path,
            clip_action=config.clip_action,
            clip_observation=config.clip_observation,
            use_gpu=config.use_gpu,
        )

        # Pre-allocate action arrays
        self.actions = np.zeros(config.num_joints, dtype=np.float32)
        self.last_actions = np.zeros(config.num_joints, dtype=np.float32)

        self._setup_gains_and_limits()

        # Control state
        self.counter = 0          # Policy step counter (runs at control_dt)
        self._loop_count = 0      # Physics step counter (runs at simulation_dt)
        self._start_time = 0.0
        self.control_times = []
        self._last_policy_time = 0.0
        self.policy_intervals = []  # Wall-clock intervals between policy steps

        # Common state flags
        self.robot_running = True
        self.recording_duration = getattr(config, 'recording_duration', 300.0)

        # Cached joint state for torque-aware clipping (updated in get_observation)
        self._cached_joint_pos = np.zeros(config.num_joints, dtype=np.float32)
        self._cached_joint_vel = np.zeros(config.num_joints, dtype=np.float32)

        # All joints are position-controlled.
        self._motor_modes = np.zeros(config.num_joints, dtype=np.int32)

        print(f"[BaseController] Initialized: {config.robot_name}")
        print(f"  Joints: {config.num_joints}")
        print(f"  Control freq: {1.0 / config.control_dt:.0f} Hz")

    def _setup_gains_and_limits(self):
        """Setup gains and torque limits from config."""
        config = self.config

        # PD gains
        self.joint_kps = config.joint_kps.copy()
        self.joint_kds = config.joint_kds.copy()

        # Action scale
        self.action_scale = config.action_scale.copy()

        # Torque limits
        self.max_torque = config.max_torque.copy()

    def compute_motor_targets(
        self,
        actions: np.ndarray,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
    ) -> tuple:
        """Compute position targets with torque-aware clipping.

        Each action is clipped so that the PD torque at the current state,
        ``kp * (target - q) - kd * v``, stays within ``[-max_torque, max_torque]``.

        Args:
            actions: Action vector in training order
            current_pos: Current joint positions in deployment order
            current_vel: Current joint velocities in deployment order

        Returns:
            Tuple of (target_positions, target_velocities, clipped_actions_deploy)
        """
        actions_deploy = actions[self.train_to_deploy_map]

        target_positions = np.zeros(self.config.num_joints, dtype=np.float32)
        target_velocities = np.zeros(self.config.num_joints, dtype=np.float32)
        clipped_actions = actions_deploy.copy()

        for i in range(self.config.num_joints):
            if self.joint_kps[i] > 0:
                damp = self.joint_kds[i] * current_vel[i]
                pos_range_min = (-self.max_torque[i] + damp) / self.joint_kps[i]
                pos_range_max = (self.max_torque[i] + damp) / self.joint_kps[i]

                pos_min = current_pos[i] + pos_range_min
                pos_max = current_pos[i] + pos_range_max

                if self.action_scale[i] != 0:
                    action_min = (pos_min - self.config.default_angles[i]) / self.action_scale[i]
                    action_max = (pos_max - self.config.default_angles[i]) / self.action_scale[i]
                    clipped_actions[i] = np.clip(actions_deploy[i], action_min, action_max)

            target_positions[i] = clipped_actions[i] * self.action_scale[i] + self.config.default_angles[i]

        return target_positions, target_velocities, clipped_actions

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Build the policy input from the current robot state.

        Must also refresh ``_cached_joint_pos`` / ``_cached_joint_vel``, which the
        torque-aware clip in ``apply_action`` uses.

        Returns:
            Observation for policy inference (array, or dict of named inputs)
        """
        pass

    def infer_action(self, observation) -> np.ndarray:
        """Run policy inference.

        Args:
            observation: Policy input (array, or dict of named inputs)

        Returns:
            Action vector in training order
        """
        return self.policy.infer(observation)

    def _guarded_infer(self, observation) -> np.ndarray:
        """``infer_action`` with the non-finite guard: a zero action replaces a non-finite input or output.

        A zero action is the default pose after the torque-aware clip (the C++ loop does the same; a NaN
        would otherwise pass the clip as a finite limit, i.e. full torque). Logged once when it starts
        and once when the policy output is finite again.
        """
        actions = self.infer_action(observation) if all_finite(observation) else None
        if actions is not None and not all_finite(actions):
            actions = None
        if actions is None:
            if not self._nonfinite_active:
                print("[SAFETY] non-finite observation or policy action; holding the default pose "
                      "until it is finite again")
            self._nonfinite_active = True
            return np.zeros(self.config.num_joints, dtype=np.float32)
        if self._nonfinite_active:
            print("[SAFETY] policy input and output finite again; resuming")
            self._nonfinite_active = False
        return actions

    def _policy_step(self) -> np.ndarray:
        """Observation -> guarded inference -> counter advance. Returns the action (training order)."""
        observation = self.get_observation()
        actions = self._guarded_infer(observation)
        self.counter += 1
        return actions

    def latch_damping(self, fault: str) -> None:
        """Latch a safety fault: command damping (if the backend supports it) and stop the run.

        Sticky like the C++ SafetyMonitor latch: nothing clears it, and the caller's loop exits on
        ``robot_running == False``. Logged once, on the first fault.
        """
        if self.fault is None:
            print(f"[SAFETY] {fault}: damping and stopping")
            self.fault = fault
        self.robot_running = False
        send_damping = getattr(self.backend, "send_damping", None)
        if send_damping is not None:
            send_damping()

    def apply_action(self, actions: np.ndarray):
        """Apply actions with torque-aware clipping and send them to the backend.

        Stores the clipped actions (training order) as ``last_actions``; this is
        the ``last_action`` observation term of the next policy step. A non-finite
        motor target is never sent: it latches damping instead.

        Args:
            actions: Action vector in training order
        """
        if self.fault is not None:
            return
        target_positions, target_velocities, clipped_actions = self.compute_motor_targets(
            actions, self._cached_joint_pos, self._cached_joint_vel,
        )
        if not all_finite(target_positions, target_velocities):
            self.latch_damping("non-finite motor target")
            return
        self.last_actions = clipped_actions[self.deploy_to_train_map]
        cmd = MotorCommand(
            positions=target_positions,
            velocities=target_velocities,
            kp=self.joint_kps,
            kd=self.joint_kds,
            torques=np.zeros(self.config.num_joints, dtype=np.float32),
            modes=self._motor_modes,
        )
        self.backend.send_command(cmd)

    def run_sim_step(self) -> bool:
        """Execute one physics step at simulation_dt rate.

        Runs at simulation_dt (e.g., 200 Hz). Policy inference happens
        every `decimation` steps. The same actions are re-applied on every
        physics step in between; the torque-aware clip uses the joint state
        cached by the last observation.

        Returns:
            True to continue, False to stop
        """
        t0 = time.time()

        # Run policy at decimated rate (e.g., every 4th physics step = 50 Hz)
        if self._loop_count % self.config.decimation == 0:
            # Track wall-clock interval between policy steps
            now = time.time()
            if self._last_policy_time > 0:
                self.policy_intervals.append(now - self._last_policy_time)
                if len(self.policy_intervals) > 100:
                    self.policy_intervals = self.policy_intervals[-100:]
            self._last_policy_time = now

            # Observation (reads fresh state) -> policy -> counter advance
            self.actions = self._policy_step()

        # Apply actions every physics step (re-applies same actions between
        # policy steps, which recomputes torque-aware clipping with the
        # state cached from the last observation call)
        self.apply_action(self.actions)

        # Update physics step counter
        self._loop_count += 1

        # Timing - maintain simulation_dt rate
        elapsed = time.time() - t0
        if self._loop_count % self.config.decimation == 0:
            self.control_times.append(elapsed)
        # Pace to an absolute deadline, not a per-step delta. `time.sleep(dt - elapsed)`
        # oversleeps (on macOS by ~1.2 ms for a 5 ms request), and paying that on every
        # physics step runs the simulation at ~0.8x real time regardless of machine speed.
        # With a running deadline, overshooting one sleep shortens the next, so the average
        # rate converges on simulation_dt. Falling more than one step behind resets the
        # deadline instead of accumulating a debt the loop would burn CPU to repay.
        _dt = self.config.simulation_dt
        _next = getattr(self, '_next_step_deadline', 0.0)
        _now = time.time()
        _next = (_now if _next <= 0.0 else _next) + _dt
        _slack = _next - time.time()
        if _slack > 0:
            time.sleep(_slack)
        elif _slack < -_dt:
            _next = time.time()
        self._next_step_deadline = _next

        return self._should_continue()

    def run_real_step(self) -> bool:
        """Execute one control step at control_dt rate for real robots.

        Unlike run_sim_step() which runs at simulation_dt with decimation
        (for MuJoCo physics stepping), this runs directly at control_dt.
        Each call computes a fresh observation, infers an action, and
        sends it — no decimation loop needed since there is no physics
        simulation to step through.

        Returns:
            True to continue, False to stop
        """
        t0 = time.time()

        # Track wall-clock interval between policy steps
        now = time.time()
        if self._last_policy_time > 0:
            self.policy_intervals.append(now - self._last_policy_time)
            if len(self.policy_intervals) > 100:
                self.policy_intervals = self.policy_intervals[-100:]
        self._last_policy_time = now

        # Observation (reads fresh state) -> policy -> counter advance, then send
        self.actions = self._policy_step()
        self.apply_action(self.actions)

        self._loop_count += self.config.decimation  # Advance by decimation steps for _should_continue

        # Timing - maintain control_dt rate
        elapsed = time.time() - t0
        self.control_times.append(elapsed)
        sleep_time = max(0.0, self.config.control_dt - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
        if self.counter % 50 == 0:
            print(f"  [step {self.counter:5d}]  compute={elapsed*1000:.1f}ms  sleep={sleep_time*1000:.1f}ms  target={self.config.control_dt*1000:.1f}ms")

        return self._should_continue()

    def _should_continue(self) -> bool:
        """Check if controller should continue running.

        Checks recording duration, robot_running flag, and viewer state (MuJoCo).
        Subclasses can override for custom logic.
        """
        passed_time = self._loop_count * self.config.simulation_dt
        if passed_time >= self.recording_duration:
            return False
        if not self.robot_running:
            return False
        if hasattr(self, 'backend') and getattr(self.backend, 'viewer', None) is not None:
            return self.backend.is_viewer_running()
        return True

    def cleanup(self):
        """Cleanup on exit."""
        print(f"[{self.__class__.__name__}] Cleaning up...")
        self.backend.cleanup()

    def run_sim(self):
        """Main simulation control loop.

        Calls run_sim_step() in a loop until completion or interruption.
        The real robot runs G1RealController.run_real instead.
        """
        print("[Controller] Starting control loop")
        self._start_time = time.time()

        try:
            while self.run_sim_step():
                pass
        except KeyboardInterrupt:
            print("\n[Controller] Interrupted by user")
        except Exception as e:
            print(f"\n[Controller] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
            self.print_stats()

    def print_stats(self):
        """Print performance statistics."""
        if self.control_times:
            times_ms = np.array(self.control_times) * 1000
            total_time = time.time() - self._start_time
            print("\n[Controller] Statistics:")
            print(f"  Steps: {self._loop_count} physics, {self.counter} policy, Time: {total_time:.1f}s")
            print(f"  Freq: {self._loop_count / total_time:.1f} Hz physics, {self.counter / total_time:.1f} Hz policy")
            print(f"  Control time: {np.mean(times_ms):.2f}ms avg, {np.max(times_ms):.2f}ms max")