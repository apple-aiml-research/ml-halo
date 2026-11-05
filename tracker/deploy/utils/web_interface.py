#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
Flask web interface for the G1 deployment.

- WebControllerBase: dashboard, REST endpoints (status, stop, damping) and robot
  telemetry (read from the MuJoCo data by default; the real-robot script overrides
  the state monitor).
- MotionTrackingWebInterface: motion selection from a folder of CSV clips,
  stand-up, restart and playback FPS (used by deploy_mujoco.py --web and
  play_real_web_interface.py).

The server has no authentication and binds 127.0.0.1 by default (``host``); bind
another address only on a trusted network.
"""

import time
import threading
import traceback
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .math import quat_to_rotation_matrix, quat_to_euler

from flask import Flask, jsonify, request, render_template_string


# =============================================================================
# BASE DASHBOARD HTML
# =============================================================================

BASE_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ robot_name }} Control</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 15px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #eee;
            min-height: 100vh;
        }
        .main-layout {
            display: grid;
            grid-template-columns: 2fr 3fr;
            gap: 15px;
            align-items: start;
        }
        .left-column, .right-column {
            min-width: 0;
        }
        @media (max-width: 900px) {
            .main-layout { grid-template-columns: 1fr; }
        }
        .header {
            text-align: center;
            padding: 15px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 15px;
        }
        .header h1 {
            margin: 0;
            font-size: 1.8em;
            background: linear-gradient(90deg, #00d9ff, #ff00ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .panel {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .panel-title {
            font-size: 0.85em;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }

        /* Robot State Grid */
        .state-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        @media (max-width: 500px) { .state-grid { grid-template-columns: 1fr; } }
        .state-box {
            background: rgba(0,0,0,0.2);
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .state-box.wide { grid-column: span 2; }
        .state-box h4 {
            margin: 0 0 8px 0;
            font-size: 0.7em;
            color: #888;
            text-transform: uppercase;
        }
        .state-row {
            display: flex;
            justify-content: space-between;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            padding: 2px 0;
        }
        .state-row span { color: #00d9ff; font-weight: bold; }
        .joint-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.75em;
        }
        .joint-item {
            background: rgba(0,0,0,0.2);
            padding: 4px;
            border-radius: 4px;
            text-align: center;
        }
        .joint-item .joint-name { color: #888; font-size: 0.8em; }
        .joint-item .joint-val { color: #00d9ff; font-weight: bold; }

        /* Controls */
        .btn {
            background: #0f4c75;
            color: white;
            border: none;
            padding: 12px 24px;
            margin: 5px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.2s;
        }
        .btn:hover { background: #1b6ca8; transform: translateY(-1px); }
        .btn-danger { background: #c0392b; }
        .btn-danger:hover { background: #e74c3c; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #2ecc71; }
        .control-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .control-panel {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .slider-group { margin: 15px 0; }
        .slider-group label {
            display: block;
            margin-bottom: 5px;
            color: #888;
            font-size: 0.85em;
            font-weight: bold;
        }
        input[type="range"] {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            -webkit-appearance: none;
            outline: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: linear-gradient(135deg, #00d9ff, #0099cc);
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,217,255,0.4);
        }
        .value-display {
            text-align: right;
            font-family: 'Courier New', monospace;
            color: #00d9ff;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ robot_name }} Control Panel</h1>
    </div>

    <div class="main-layout">
    <!-- Left Column: Robot State -->
    <div class="left-column">

    <!-- Robot State Telemetry -->
    <div class="panel">
        <div class="panel-title">Robot State</div>
        <div class="state-grid">
            <div class="state-box">
                <h4>Linear Vel (m/s)</h4>
                <div class="state-row">X: <span id="lin-x">0.00</span></div>
                <div class="state-row">Y: <span id="lin-y">0.00</span></div>
                <div class="state-row">Z: <span id="lin-z">0.00</span></div>
            </div>
            <div class="state-box">
                <h4>Angular Vel (rad/s)</h4>
                <div class="state-row">X: <span id="ang-x">0.00</span></div>
                <div class="state-row">Y: <span id="ang-y">0.00</span></div>
                <div class="state-row">Z: <span id="ang-z">0.00</span></div>
            </div>
            <div class="state-box">
                <h4>Orientation (deg)</h4>
                <div class="state-row">Roll: <span id="or-roll">0.0</span></div>
                <div class="state-row">Pitch: <span id="or-pitch">0.0</span></div>
                <div class="state-row">Yaw: <span id="or-yaw">0.0</span></div>
            </div>
            <div class="state-box">
                <h4>Position (m)</h4>
                <div class="state-row">X: <span id="pos-x">0.00</span></div>
                <div class="state-row">Y: <span id="pos-y">0.00</span></div>
                <div class="state-row">Z: <span id="pos-z">0.00</span></div>
            </div>
            <div class="state-box wide">
                <h4>Joint Positions (rad)</h4>
                <div class="joint-grid" id="joint-pos-grid"></div>
            </div>
        </div>
    </div>

    <!-- Emergency Controls -->
    <div class="panel">
        <div class="control-buttons">
            <button class="btn btn-danger" onclick="emergencyStop()">EMERGENCY STOP</button>
            <button class="btn" onclick="enterDamping()">Damping Mode</button>
            <button class="btn" onclick="findRobot()" style="background:#2980b9;">Find Robot</button>
        </div>
    </div>

    </div><!-- end left-column -->

    <!-- Right Column: Controls -->
    <div class="right-column">
    {{ extra_content | safe }}
    </div><!-- end right-column -->

    </div><!-- end main-layout -->

    <script>
        const jointNames = {{ joint_names_json | safe }};
        const numJoints = {{ num_joints }};

        // Build joint grids dynamically
        function buildJointGrids() {
            const posGrid = document.getElementById('joint-pos-grid');
            let posHtml = '';
            for (let i = 0; i < numJoints; i++) {
                const name = jointNames[i] || ('joint_' + i);
                posHtml += '<div class="joint-item"><span class="joint-name">' + name +
                           '</span><br><span class="joint-val" id="jp-' + i + '">0.00</span></div>';
            }
            posGrid.innerHTML = posHtml;
        }
        buildJointGrids();

        function updateStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    const rs = data.robot_state;
                    if (rs) {
                        if (rs.base_lin_vel) {
                            document.getElementById('lin-x').textContent = rs.base_lin_vel[0].toFixed(2);
                            document.getElementById('lin-y').textContent = rs.base_lin_vel[1].toFixed(2);
                            document.getElementById('lin-z').textContent = rs.base_lin_vel[2].toFixed(2);
                        }
                        if (rs.base_ang_vel) {
                            document.getElementById('ang-x').textContent = rs.base_ang_vel[0].toFixed(2);
                            document.getElementById('ang-y').textContent = rs.base_ang_vel[1].toFixed(2);
                            document.getElementById('ang-z').textContent = rs.base_ang_vel[2].toFixed(2);
                        }
                        if (rs.orientation) {
                            document.getElementById('or-roll').textContent = rs.orientation.roll.toFixed(1);
                            document.getElementById('or-pitch').textContent = rs.orientation.pitch.toFixed(1);
                            document.getElementById('or-yaw').textContent = rs.orientation.yaw.toFixed(1);
                        }
                        if (rs.position) {
                            document.getElementById('pos-x').textContent = rs.position[0].toFixed(2);
                            document.getElementById('pos-y').textContent = rs.position[1].toFixed(2);
                            document.getElementById('pos-z').textContent = rs.position[2].toFixed(2);
                        }
                        if (rs.joint_pos) {
                            for (let i = 0; i < rs.joint_pos.length; i++) {
                                const el = document.getElementById('jp-' + i);
                                if (el) el.textContent = rs.joint_pos[i].toFixed(2);
                            }
                        }
                    }
                })
                .catch(e => {});
        }

        function emergencyStop() {
            fetch('/stop', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.success) updateStatus();
                });
        }

        function findRobot() {
            fetch('/camera/find_robot', {method: 'POST'})
                .then(r => r.json());
        }

        function enterDamping() {
            fetch('/damping', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.success) updateStatus();
                });
        }

        // Poll at 10Hz
        setInterval(updateStatus, 100);
        updateStatus();
    </script>
</body>
</html>
"""


# =============================================================================
# WEB CONTROLLER BASE
# =============================================================================

class WebControllerBase(ABC):
    """Abstract base class for web control interfaces using Flask.

    Subclasses must implement:
        - get_robot_state(): Returns dict with robot status
        - send_command(): Handles incoming commands
        - get_extra_dashboard_content(): Returns HTML for custom controls

    Common endpoints provided:
        GET  /           -> HTML dashboard
        GET  /status     -> JSON robot state
        POST /stop       -> Emergency stop
        POST /damping    -> Enter damping mode
    """

    def __init__(self, controller, robot_name: str = "Robot", port: int = 8080, host: str = "127.0.0.1"):
        """Initialize web interface.

        Args:
            controller: Robot controller instance
            robot_name: Display name for dashboard
            port: HTTP port number
            host: Address to bind (default loopback only; the server has no authentication)
        """
        self.controller = controller
        self.robot_name = robot_name
        self.port = port
        self.host = host

        # Robot state telemetry (updated by background thread)
        self._robot_state = {
            'base_lin_vel': [0.0, 0.0, 0.0],
            'base_ang_vel': [0.0, 0.0, 0.0],
            'orientation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            'position': [0.0, 0.0, 0.0],
            'joint_pos': [0.0] * getattr(controller.config, 'num_joints', 0),
            'joint_vel': [0.0] * getattr(controller.config, 'num_joints', 0),
        }

        # Create Flask app
        self.app = Flask(__name__)
        self._setup_routes()

        # Thread for running server
        self._server_thread: Optional[threading.Thread] = None
        self._state_thread: Optional[threading.Thread] = None
        self._running = False

    def get_joint_names(self) -> List[str]:
        """Get joint names from controller config.

        Returns:
            List of joint name strings.
        """
        if hasattr(self.controller, 'config') and hasattr(self.controller.config, 'deployment_joint_order'):
            return list(self.controller.config.deployment_joint_order)
        num = getattr(self.controller.config, 'num_joints', 0)
        return [f'joint_{i}' for i in range(num)]

    def _setup_routes(self):
        """Setup Flask routes."""
        import json

        joint_names = self.get_joint_names()
        num_joints = len(joint_names)
        joint_names_json = json.dumps(joint_names)

        @self.app.route('/')
        def dashboard():
            extra = self.get_extra_dashboard_content()
            return render_template_string(
                BASE_DASHBOARD_HTML,
                robot_name=self.robot_name,
                extra_content=extra,
                joint_names_json=joint_names_json,
                num_joints=num_joints,
            )

        @self.app.route('/status')
        def status():
            state = self.get_robot_state()
            state['robot_state'] = self._robot_state
            return jsonify(state)

        @self.app.route('/stop', methods=['POST'])
        def stop():
            result = self.send_command('stop')
            return jsonify(result)

        @self.app.route('/damping', methods=['POST'])
        def damping():
            result = self.send_command('damping')
            return jsonify(result)

        @self.app.route('/camera/find_robot', methods=['POST'])
        def camera_find_robot():
            backend = getattr(self.controller, 'backend', None)
            if backend is None:
                return jsonify({'success': False, 'message': 'No backend'})
            viewer = getattr(backend, 'viewer', None)
            d = getattr(backend, 'd', None)
            if viewer is None or d is None:
                return jsonify({'success': False, 'message': 'No viewer or data'})
            robot_pos = d.qpos[0:3]
            viewer.cam.lookat[0] = robot_pos[0]
            viewer.cam.lookat[1] = robot_pos[1]
            return jsonify({'success': True})

    # =========================================================================
    # STATE MONITOR
    # =========================================================================

    def _state_monitor_loop(self):
        """Background thread: read MuJoCo data at 20 Hz and update _robot_state."""
        num_joints = getattr(self.controller.config, 'num_joints', 0)

        while self._running:
            try:
                backend = getattr(self.controller, 'backend', None)
                if backend is None:
                    time.sleep(0.1)
                    continue

                d = getattr(backend, 'd', None)
                if d is None or d.time <= 0.0:
                    time.sleep(0.05)
                    continue

                # Position
                self._robot_state['position'] = d.qpos[0:3].copy().tolist()

                # Orientation: quaternion (wxyz) -> euler
                quat = d.qpos[3:7].copy()
                euler = quat_to_euler(quat)
                self._robot_state['orientation'] = {
                    'roll': float(np.degrees(euler[0])),
                    'pitch': float(np.degrees(euler[1])),
                    'yaw': float(np.degrees(euler[2])),
                }

                # Body-frame velocities via rotation matrix
                R = quat_to_rotation_matrix(quat)
                Rt = R.T

                world_lin_vel = d.qvel[0:3].copy()
                world_ang_vel = d.qvel[3:6].copy()

                self._robot_state['base_lin_vel'] = (Rt @ world_lin_vel).tolist()
                self._robot_state['base_ang_vel'] = (Rt @ world_ang_vel).tolist()

                # Joint states
                self._robot_state['joint_pos'] = d.qpos[7:7 + num_joints].copy().tolist()
                self._robot_state['joint_vel'] = d.qvel[6:6 + num_joints].copy().tolist()

                time.sleep(0.05)  # 20Hz
            except Exception:
                time.sleep(0.1)

    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================

    @abstractmethod
    def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state as dictionary.

        Returns:
            Dict with status info like:
            {
                "mode": "control",
                "state": "RUNNING",
                "counter": 1234,
            }
        """
        pass

    @abstractmethod
    def send_command(self, cmd_type: str, **kwargs) -> Dict[str, Any]:
        """Handle incoming command.

        Args:
            cmd_type: Command type ('stop', 'damping', ...)
            **kwargs: Command-specific parameters

        Returns:
            Dict with result: {"success": True/False, "message": "..."}
        """
        pass

    def get_extra_dashboard_content(self) -> str:
        """Get extra HTML content for dashboard.

        Override in subclass to add robot-specific controls.

        Returns:
            HTML string to insert into dashboard
        """
        return ""

    # =========================================================================
    # SERVER CONTROL
    # =========================================================================

    def start(self, threaded: bool = True):
        """Start the web server.

        Args:
            threaded: Run in background thread (default True)
        """
        self._running = True

        # Start state monitor thread
        self._state_thread = threading.Thread(
            target=self._state_monitor_loop,
            daemon=True
        )
        self._state_thread.start()

        if threaded:
            self._server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self._server_thread.start()
            print(f"[Web] Server started on http://{self.host}:{self.port}")
        else:
            self._run_server()

    def _run_server(self):
        """Run Flask server (internal)."""
        # Disable Flask's default logging
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        self.app.run(
            host=self.host,
            port=self.port,
            threaded=True,
            use_reloader=False
        )

    def stop(self):
        """Stop the web server."""
        self._running = False
        print("[Web] Server stopped")


# =============================================================================
# MOTION TRACKING WEB INTERFACE (G1)
# =============================================================================

class MotionTrackingWebInterface(WebControllerBase):
    """Web interface for G1 motion tracking.

    Adds motion discovery (``data_dir/*.csv``), a motion button grid, playback
    controls, and auto-standup on motion completion. CSV clips are sent to the
    controller in deployment order.
    """

    def __init__(self, controller, robot_name: str = "G1", port: int = 8080,
                 data_dir=None, initial_motion=None,
                 default_motion_fps: int = 30, host: str = "127.0.0.1"):
        self.data_dir = data_dir
        self._motion_names = self._discover_motions()
        self._current_motion = initial_motion
        self._motion_fps = default_motion_fps
        super().__init__(controller, robot_name, port, host=host)
        self._add_motion_routes()

    def _discover_motions(self):
        """Scan data dir for available CSV motion files."""
        if not self.data_dir:
            return []
        csvs = sorted(Path(self.data_dir).glob('*.csv'))
        return [p.stem for p in csvs]

    def _motion_path(self, name) -> Optional[Path]:
        """``data_dir/<name>.csv`` if it is an existing file inside data_dir, else None.

        The name comes from the client; resolving it and requiring the result to stay in data_dir
        refuses path traversal (``../``, absolute paths, symlinks out of the directory).
        """
        if not self.data_dir or not isinstance(name, str) or not name:
            return None
        try:
            root = Path(self.data_dir).resolve()
            path = (root / f"{name}.csv").resolve()
            if path.parent != root or not path.is_file():
                return None
        except (OSError, ValueError):
            return None
        return path

    def _send_motion_file(self, path: Path) -> Dict[str, Any]:
        """Load a CSV clip and send it to the controller. Errors are logged here, not returned to the client."""
        try:
            qpos_data = np.loadtxt(path, delimiter=",", ndmin=2)
            if not self.controller.send_ref_motion(qpos_data, input_fps=self._motion_fps):
                return {'success': False, 'message': 'Motion rejected by the controller'}
        except Exception:
            print(f"[Web] Loading motion {path.name} failed:")
            traceback.print_exc()
            return {'success': False, 'message': 'Failed to load the motion (see the server log)'}
        return {'success': True}

    def _do_standup(self):
        """Transition robot to standing pose using send_ref_motion for smooth blending."""
        if self.controller.motion_loader is None:
            return
        backend = getattr(self.controller, 'backend', None)
        if backend is None or getattr(backend, 'd', None) is None:
            return

        # Read current base position and quaternion (wxyz in MuJoCo)
        current_base_pos = backend.d.qpos[0:3].copy()
        current_quat_wxyz = backend.d.qpos[3:7].copy()
        # Extract yaw only — discard pitch/roll so standup targets upright
        yaw = quat_to_euler(current_quat_wxyz)[2]  # quat_to_euler returns [roll, pitch, yaw]
        # Build upright quaternion with only yaw: quat_wxyz = [cos(yaw/2), 0, 0, sin(yaw/2)]
        upright_quat_wxyz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float64)
        current_quat_xyzw = upright_quat_wxyz[[1, 2, 3, 0]]  # wxyz -> xyzw

        # standing_joint_pos is already in DEPLOYMENT order — MotionLoader only remaps
        # motion_dof_poss/vels to training order, never standing_joint_pos. send_ref_motion
        # expects deployment (qpos) order, so use it directly (mirrors deploy_real.do_standup).
        ml = self.controller.motion_loader
        standing_joints_deploy = ml.standing_joint_pos.copy()

        # Build synthetic qpos_data: shape (1, 36) = [xyz(3), quat_xyzw(4), joints_deploy(29)]
        qpos_data = np.zeros((1, 3 + 4 + len(standing_joints_deploy)), dtype=np.float64)
        qpos_data[0, 0:3] = current_base_pos
        qpos_data[0, 3:7] = current_quat_xyzw
        qpos_data[0, 7:] = standing_joints_deploy

        self.controller.send_ref_motion(qpos_data)
        self._current_motion = None

    def _add_motion_routes(self):
        """Add motion tracking routes."""

        @self.app.route('/motion/start', methods=['POST'])
        def motion_start():
            result = self.send_command('motion_start')
            return jsonify(result)

        @self.app.route('/motion/select', methods=['POST'])
        def motion_select():
            data = request.get_json() or {}
            motion_name = data.get('name', '')
            result = self.send_command('motion_select', name=motion_name)
            return jsonify(result)

        @self.app.route('/motion/standup', methods=['POST'])
        def motion_standup():
            result = self.send_command('motion_standup')
            return jsonify(result)

        @self.app.route('/motion/fps', methods=['GET', 'POST'])
        def motion_fps():
            if request.method == 'GET':
                return jsonify({'fps': self._motion_fps})
            data = request.get_json() or {}
            new_fps = data.get('fps')
            if new_fps is None or not isinstance(new_fps, (int, float)) or new_fps <= 0:
                return jsonify({'success': False, 'message': 'Invalid fps value'})
            self._motion_fps = int(new_fps)
            # If a motion is currently playing, reload it at the new FPS
            path = self._motion_path(self._current_motion) if self._current_motion else None
            if path is not None:
                result = self._send_motion_file(path)
                if not result['success']:
                    return jsonify(result)
            return jsonify({'success': True, 'fps': self._motion_fps})

    def get_robot_state(self) -> Dict[str, Any]:
        """Return G1 motion tracking state with motion metadata."""
        max_idx = 0
        if self.controller.motion_data:
            max_idx = len(self.controller.motion_data[0]) - 1
        counter = getattr(self.controller, 'counter', 0)
        motion_done = (counter >= max_idx) and max_idx > 0 and self._current_motion is not None

        # Compute running frequency from wall-clock policy intervals
        policy_intervals = getattr(self.controller, 'policy_intervals', [])
        ctrl_times = getattr(self.controller, 'control_times', [])
        if len(policy_intervals) >= 5:
            avg_interval = np.mean(policy_intervals[-50:])
            freq_hz = 1.0 / avg_interval if avg_interval > 0 else 0.0
        else:
            freq_hz = 0.0
        avg_ms = np.mean(ctrl_times[-50:]) * 1000 if len(ctrl_times) >= 5 else 0.0

        return {
            'mode': 'motion_tracking',
            'counter': counter,
            'running': getattr(self.controller, 'robot_running', False),
            'has_motion': getattr(self.controller, 'motion_loader', None) is not None,
            'motions': self._motion_names,
            'current_motion': self._current_motion,
            'motion_done': motion_done,
            'motion_progress': counter / max(max_idx, 1),
            'motion_fps': self._motion_fps,
            'policy_freq_hz': round(freq_hz, 1),
            'policy_step_ms': round(avg_ms, 3),
        }

    def send_command(self, cmd_type: str, **kwargs) -> Dict[str, Any]:
        """Handle motion tracking commands."""
        if cmd_type == 'motion_start':
            self.controller.restart_motion()
            return {'success': True, 'message': 'Motion restarted'}

        if cmd_type == 'motion_select':
            name = kwargs.get('name', '')
            if not self.data_dir:
                return {'success': False, 'message': 'No data directory configured'}
            path = self._motion_path(name)
            if path is None:
                return {'success': False, 'message': 'Unknown motion'}
            result = self._send_motion_file(path)
            if not result['success']:
                return result
            self._current_motion = name
            return {'success': True, 'message': f'Motion loaded: {name}'}

        if cmd_type == 'motion_standup':
            self._do_standup()
            return {'success': True, 'message': 'Transitioning to standing'}

        if cmd_type in ('stop', 'damping'):
            self.controller.robot_running = False
            message = 'Emergency stop activated' if cmd_type == 'stop' else 'Damping mode activated'
            return {'success': True, 'message': message}

        return {'success': False, 'message': f'Unknown command: {cmd_type}'}

    def get_extra_dashboard_content(self) -> str:
        """Motion tracking UI with status bar, motion button grid, and stand up button."""
        import json
        motion_names_json = json.dumps(self._motion_names)
        initial_motion_json = json.dumps(self._current_motion)

        return f"""
        <style>
            .motion-status-bar {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .mode-badge {{
                display: inline-block;
                padding: 6px 16px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
                transition: all 0.3s;
            }}
            .mode-standing {{ background: linear-gradient(135deg, #27ae60, #2ecc71); color: #fff; }}
            .mode-playing {{ background: linear-gradient(135deg, #ff6b9d, #c44569); color: #fff; }}
            .progress-bar-container {{
                width: 100%;
                height: 6px;
                background: rgba(255,255,255,0.1);
                border-radius: 3px;
                margin-top: 10px;
                overflow: hidden;
            }}
            .progress-bar-fill {{
                height: 100%;
                background: linear-gradient(90deg, #ff6b9d, #c44569);
                border-radius: 3px;
                transition: width 0.2s;
                width: 0%;
            }}
            .motion-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
                gap: 6px;
                padding: 2px;
            }}
            .motion-btn {{
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 6px 4px;
                border: 1px solid rgba(255,107,157,0.3);
                border-radius: 6px;
                background: rgba(255,107,157,0.1);
                color: #fff;
                cursor: pointer;
                font-size: 0.75em;
                font-weight: bold;
                transition: all 0.2s;
            }}
            .motion-btn:hover {{
                background: rgba(255,107,157,0.3);
                border-color: rgba(255,107,157,0.6);
                transform: translateY(-1px);
            }}
            .motion-btn .motion-name {{
                font-size: 0.8em;
                color: #ccc;
                text-transform: capitalize;
                text-align: center;
                line-height: 1.2;
            }}
            .motion-btn.playing {{
                background: linear-gradient(135deg, #ff6b9d, #c44569);
                border-color: #ff6b9d;
                box-shadow: 0 0 10px rgba(255,107,157,0.4);
            }}
            .motion-btn.playing .motion-name {{ color: #fff; }}
            .category-header {{
                font-size: 0.75em;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #ff6b9d;
                margin: 10px 0 6px 2px;
                padding-bottom: 4px;
                border-bottom: 1px solid rgba(255,107,157,0.2);
            }}
            .category-header:first-child {{ margin-top: 0; }}
            .standup-btn {{
                background: linear-gradient(135deg, #27ae60, #2ecc71);
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 0.85em;
                font-weight: bold;
                transition: all 0.2s;
            }}
            .standup-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(39,174,96,0.4);
            }}
            .fps-control {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                margin: 10px 0;
            }}
            .fps-control label {{
                color: #888;
                font-size: 0.85em;
                font-weight: bold;
            }}
            .fps-control input[type="number"] {{
                width: 70px;
                padding: 8px;
                border-radius: 8px;
                border: 1px solid rgba(255,255,255,0.2);
                background: rgba(0,0,0,0.3);
                color: #00d9ff;
                font-family: 'Courier New', monospace;
                font-size: 0.95em;
                font-weight: bold;
                text-align: center;
            }}
            .fps-btn {{
                background: #0f4c75;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 0.85em;
                font-weight: bold;
                transition: all 0.2s;
            }}
            .fps-btn:hover {{ background: #1b6ca8; }}
        </style>

        <!-- Status Bar -->
        <div class="panel">
            <div class="motion-status-bar">
                <div>
                    <span id="motion-badge" class="mode-badge mode-standing">STANDING</span>
                    <span id="motion-name-display" style="margin-left: 12px; color: #888; font-size: 0.85em;"></span>
                </div>
                <div style="font-family: monospace; color: #888; font-size: 0.85em;">
                    Step: <span id="motion-step">0</span>
                    &nbsp;&bull;&nbsp;
                    <span id="policy-freq" style="color: #00d9ff;">-- Hz</span>
                    <span id="policy-step-ms" style="color: #666; margin-left: 4px;">(--ms)</span>
                </div>
            </div>
            <div class="progress-bar-container" id="progress-container" style="display: none;">
                <div class="progress-bar-fill" id="progress-fill"></div>
            </div>
        </div>

        <!-- Motion Controls -->
        <div class="panel">
            <div class="panel-title">Motions</div>
            <div style="display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap;">
                <button class="standup-btn" onclick="standUp()">Stand Up</button>
                <button class="btn btn-success" onclick="restartMotion()">Restart</button>
                <label style="color: #888; font-size: 0.85em; font-weight: bold; margin-left: 8px;">FPS:</label>
                <input type="number" id="motion-fps-input" min="1" max="120" value="{self._motion_fps}"
                       style="width: 60px; padding: 8px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.2); background: rgba(0,0,0,0.3); color: #00d9ff; font-family: 'Courier New', monospace; font-size: 0.95em; font-weight: bold; text-align: center;">
                <button class="fps-btn" onclick="setMotionFPS()">Set</button>
            </div>
            <div id="motion-buttons" style="padding: 4px;"></div>
        </div>

        <script>
            const motionNames = {motion_names_json};
            let currentMotion = {initial_motion_json};
            let standupPending = false;

            // Build motion button grid grouped by category
            function buildMotionButtons() {{
                const container = document.getElementById('motion-buttons');
                if (!motionNames.length) {{
                    container.innerHTML = '<p style="color: #666; font-size: 0.85em; text-align: center;">No motions found in data folder.</p>';
                    return;
                }}
                // Group motions by category (text before last _subjectN or _number pattern)
                const groups = {{}};
                for (const name of motionNames) {{
                    const match = name.match(/^(.+?)(\\d*)_subject\\d+$/) || name.match(/^(.+?)_(\\d+)$/);
                    const category = match ? match[1] : name;
                    if (!groups[category]) groups[category] = [];
                    groups[category].push(name);
                }}
                let html = '';
                for (const [category, names] of Object.entries(groups)) {{
                    const displayCat = category.replace(/([A-Z])/g, ' $1').replace(/(\\d+)$/, ' $1').trim();
                    html += '<div class="category-header">' + displayCat + '</div>';
                    html += '<div class="motion-grid">';
                    for (const name of names) {{
                        const shortName = name.replace(category + '_', '').replace(/_/g, ' ');
                        html += '<button class="motion-btn" id="mbtn-' + name + '" ' +
                                'onclick="playMotion(\\'' + name + '\\')">' +
                                '<span class="motion-name">' + shortName + '</span>' +
                                '</button>';
                    }}
                    html += '</div>';
                }}
                container.innerHTML = html;
            }}
            buildMotionButtons();

            function playMotion(name) {{
                fetch('/motion/select', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{name: name}})
                }}).then(r => r.json()).then(data => {{
                    if (data.success) currentMotion = name;
                }});
            }}

            function standUp() {{
                fetch('/motion/standup', {{method: 'POST'}})
                    .then(r => r.json())
                    .then(data => {{
                        if (data.success) currentMotion = null;
                    }});
            }}

            function restartMotion() {{
                fetch('/motion/start', {{method: 'POST'}}).then(r => r.json());
            }}

            function setMotionFPS() {{
                const fpsInput = document.getElementById('motion-fps-input');
                const fps = parseInt(fpsInput.value);
                if (!fps || fps < 1) return;
                fetch('/motion/fps', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{fps: fps}})
                }}).then(r => r.json());
            }}

            // Poll motion status at 5Hz
            setInterval(function() {{
                fetch('/status')
                    .then(r => r.json())
                    .then(data => {{
                        const badge = document.getElementById('motion-badge');
                        const nameDisplay = document.getElementById('motion-name-display');
                        const stepEl = document.getElementById('motion-step');
                        const progressContainer = document.getElementById('progress-container');
                        const progressFill = document.getElementById('progress-fill');

                        currentMotion = data.current_motion;
                        stepEl.textContent = data.counter || 0;

                        // Update frequency display
                        const freqEl = document.getElementById('policy-freq');
                        const msEl = document.getElementById('policy-step-ms');
                        if (data.policy_freq_hz > 0) {{
                            freqEl.textContent = data.policy_freq_hz.toFixed(1) + ' Hz';
                            msEl.textContent = '(' + data.policy_step_ms.toFixed(3) + 'ms)';
                        }}

                        // Sync FPS input from server
                        if (data.motion_fps) {{
                            const fpsInput = document.getElementById('motion-fps-input');
                            if (fpsInput && document.activeElement !== fpsInput) {{
                                fpsInput.value = data.motion_fps;
                            }}
                        }}

                        // Auto-standup when motion completes
                        if (data.motion_done && currentMotion && !standupPending) {{
                            standupPending = true;
                            fetch('/motion/standup', {{method: 'POST'}})
                                .then(r => r.json())
                                .then(d => {{ if (d.success) currentMotion = null; standupPending = false; }})
                                .catch(() => {{ standupPending = false; }});
                        }}

                        if (currentMotion) {{
                            const displayName = currentMotion.replace(/_/g, ' ');
                            badge.textContent = 'PLAYING';
                            badge.className = 'mode-badge mode-playing';
                            nameDisplay.textContent = displayName;
                            progressContainer.style.display = 'block';
                            const pct = Math.min((data.motion_progress || 0) * 100, 100);
                            progressFill.style.width = pct + '%';
                        }} else {{
                            badge.textContent = 'STANDING';
                            badge.className = 'mode-badge mode-standing';
                            nameDisplay.textContent = '';
                            progressContainer.style.display = 'none';
                        }}

                        // Highlight active motion button
                        for (const name of motionNames) {{
                            const btn = document.getElementById('mbtn-' + name);
                            if (btn) {{
                                if (name === currentMotion) {{
                                    btn.classList.add('playing');
                                }} else {{
                                    btn.classList.remove('playing');
                                }}
                            }}
                        }}
                    }})
                    .catch(e => {{}});
            }}, 200);
        </script>
        """
