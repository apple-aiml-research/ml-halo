# Python deployment (G1)

Python deployment of the HALO tracker on the Unitree G1: MuJoCo sim-to-sim and the real robot from a computer
connected over Ethernet. The real-time C++ deployment (onboard Jetson Orin NX, TensorRT, D435i stereo depth) is in
[`../deploy_cpp`](../deploy_cpp/README.md).

## Install

From `tracker/`:

```bash
pip install -e ".[deploy]"   # numpy, pyyaml, onnxruntime, mujoco>=3.2, flask (flask only for the web interface)
```

Use `onnxruntime-gpu` instead of `onnxruntime` to run the policy on CUDA (`HALO_POLICY_CPU=1` forces the CPU). On
macOS, MuJoCo's viewer needs the script launched with `mjpython` instead of `python`; or pass `--headless`.

Real robot only:

- `unitree_sdk2py` (DDS to the G1). It needs CycloneDDS 0.10:

  ```bash
  git clone -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds
  cmake -S cyclonedds -B cyclonedds/build -DCMAKE_INSTALL_PREFIX=$PWD/cyclonedds/install -DBUILD_EXAMPLES=OFF
  cmake --build cyclonedds/build --target install -j
  export CYCLONEDDS_HOME=$PWD/cyclonedds/install
  git clone https://github.com/unitreerobotics/unitree_sdk2_python && pip install -e unitree_sdk2_python
  ```
- `pyrealsense2` and `opencv-python-headless` for the depth student, which reads the D435i in-process on the computer
  the camera is attached to: `pip install -e ".[deploy,depth]"`. The depth source downsamples with a valid-only area
  mean and refuses to start without OpenCV.

## Policies, configs and motions

No weights ship with the repository. Export a student with `training/scripts/rsl_rl/play.py --export_only`
([training/README.md](../training/README.md#play-and-export)); it writes `<checkpoint dir>/exported/policy.onnx`.
Put it in [`models/g1/`](models/g1/README.md), the default location, or pass its path. Pick the config that matches
the policy's ONNX inputs; the controller checks the config's groups against them at start-up.

| Config | Policy | ONNX inputs |
|---|---|---|
| `g1/configs/g1_v2.yaml` | proprio student | `obs` (1, 434), `policy_temporal` (1, 30, 166) |
| `g1/configs/g1_v2_depth.yaml` | depth student | + `student_depth` (1, 1, 48, 84) |

A motion is a 30 fps CSV clip, one row per frame: root position (3), root quaternion xyzw (4), 29 joint angles in
deployment order ([training/README.md](../training/README.md#motion-data)). Run the scripts from `deploy/g1`:
`--config` is resolved relative to it, `--policy_path` and `--motion` relative to the current directory, and a bare
motion name (no `.csv`) resolves to `tracker/data/<name>.csv`.

After the last frame of a clip the reference is padded (up to 10 s): the joints ramp linearly to
`standing_joint_pos` over K frames and then hold it, while the root holds the last frame. K follows the rule of the
smooth transition at the start of a plan: 10 frames when the largest joint gap is below 0.3 rad, else
`trunc(gap / 0.3 * 10)`, at most 100 (the C++ deployment uses the same rule).

### Policy interface

The observation is declared by the config's `observation:` block (one group per ONNX input), built by
`utils/core/observation_builder.py`, and mirrors the student observation of training
(`training/.../g1/motion_tracking_scene_distill_cfg.py`):

| Input | Shape | Content |
|---|---|---|
| `obs` | (1, 434) | angular velocity 3, projected gravity 3, joint pos 29, joint vel 29, last action 29, reference 5 × 67 = 335 (frames t+4, ..., t+1, t, current frame last; per frame: joint pos, joint vel, root gravity, root lin / ang vel in body frame), reference anchor orientation 6 |
| `policy_temporal` | (1, 30, 166) | the same proprio terms + anchor orientation, 30-frame history, + 1 reference frame per row (frames t+29, ..., t; current frame in the last row) |
| `student_depth` | (1, 1, 48, 84) | torso depth, clipped to [0.2, 2.5] m → log1p; 25 Hz in sim (each frame held 2 control steps, as in training), the freshest camera frame on hardware |

The anchor orientation compares the robot attitude with the reference root. With `anchor_mode: odom` (both configs)
the robot attitude is the odometry attitude (`rt/odommodestate` on the robot, the floating base in MuJoCo), and a
motion is rebased onto the odometry pose when it is installed.

Actions are joint position targets, `target = default_angles + action_scale * action`, tracked by the config's PD
gains (the gains and scales of training). Each target is clipped so that the PD torque at the current state stays
within `max_torque`; the clipped action is fed back as `last_action`.

## MuJoCo sim-to-sim

```bash
cd deploy/g1
python deploy_mujoco.py --config configs/g1_v2.yaml --policy_path ../models/g1/policy.onnx --motion <motion.csv>
python deploy_mujoco.py --config configs/g1_v2_depth.yaml --policy_path ../models/g1/depth_policy.onnx \
    --motion <motion.csv> --headless
```

The motion is rebased onto the robot's spawn pose (yaw and position of frame 0) without a transition ramp, which
needs at least two rows (a one-row clip is played as loaded). Without `--motion` the robot tracks a standing pose.
The run ends when the viewer is closed or `--duration` elapses. The scene is flat ground.

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/g1_v2.yaml` | Config file |
| `--policy_path` | `../models/g1/policy.onnx` | ONNX policy |
| `--motion` | none (standing pose) | Motion CSV |
| `--motion_fps` | `30` | Motion FPS |
| `--headless` | off | No viewer (needed without a display) |
| `--duration` | `3000.0` | Max runtime in seconds |
| `--web` / `--port` / `--host` | off / `8080` / `127.0.0.1` | Web interface: pick motions from `tracker/data/*.csv`, stand up, restart, change FPS. It has no authentication; bind other addresses only on a trusted network |
| `--allow_no_depth` | off | Run a depth policy without a depth source (far-plane input; debugging only) |
| `--allow_depth_clip_mismatch` | off | Accept a depth clip different from the training clip |

With a depth config the training depth camera (pose, 59.26° vertical FOV, 84×48) is injected on `torso_link` and
rendered on the control thread. Without a display (`--headless` on a Linux GPU machine), `export MUJOCO_GL=egl` so
the camera can render. A preflight refuses to start a depth policy whose depth source is missing or does not
match the config, or whose depth clip differs from training.

The MuJoCo plant is close to, not identical with, the training plant:

- training delays depth by 1–3 frames (40–120 ms) and PD commands by 1–5 physics steps (5–25 ms; 15–40 ms, re-drawn
  about every 20 ms, in Phase 3, PostRL); MuJoCo has neither;
- the MJCF is the same identified lab robot as training ([training/README.md](../training/README.md#robot)), but it
  also carries the identified joint `frictionloss`, which training leaves out;
- MuJoCo collides the robot's own links (e.g. a hand with the hip) while training runs with self-collision off, so
  clips that bring a hand close to the body (some dances) can catch in sim.

## Real robot

`deploy_real.py` runs the policy on a computer connected to the robot over Ethernet; a depth config must run on the
computer the D435i is attached to. Test every policy and motion in MuJoCo first.

1. Start the robot suspended (feet off the ground) and enter low-level debug mode: hold **L2 + R2** on the remote.
2. Connect Ethernet and give the robot-facing interface a static address on `192.168.123.0/24` (netmask
   `255.255.255.0`) that no robot computer uses, e.g. `192.168.123.222`. Its name (`ip addr`) is `--net`. The robot
   computer is at `unitree@192.168.123.164` (factory default; change the password).
3. Run:

   ```bash
   cd deploy/g1
   python deploy_real.py --config configs/g1_v2.yaml --policy_path ../models/g1/policy.onnx --motion <motion.csv> --net eth0
   python deploy_real.py --config configs/g1_v2_depth.yaml --policy_path ../models/g1/depth_policy.onnx \
       --motion <motion.csv> --net eth0
   ```

`deploy_real.py` takes `--net` (default `eth0`), `--config`, `--policy_path`, `--motion` (required), `--motion_fps`,
`--duration`, `--allow_no_depth` and `--allow_depth_clip_mismatch`. `play_real_web_interface.py` takes the same plus
`--port` (`8080`) and `--host` (`127.0.0.1`; `--duration` defaults to 86400 s) and adds the web interface on top of
the remote state machine. Before START the preflight requires live odometry (`rt/odommodestate`, younger than 100 ms)
and exits if none arrives within 2 s.

| Button | State | Action |
|---|---|---|
| START | INIT | Move (2 s, reduced gains) to frame 0 of the motion and hold it |
| B (recommended) | DEFAULT_POS | Stand up, then enter CONTROL with a standing reference |
| A | DEFAULT_POS | Enter CONTROL and track the motion (installed on the press, rebased onto the robot pose with a smooth transition) |
| A / B | CONTROL | Replay the loaded motion / return to the standing pose |
| SELECT | any, including the START move | Emergency stop (damping) |

A and B act on the press. Entering CONTROL is refused, with a log line, while odometry is not live or, for a depth
policy, while the camera has no fresh frame; press again once it is live.

Safety guards, checked every loop iteration in every state (the limits of the C++ deployment):

| Condition | Response |
|---|---|
| Robot state (`rt/lowstate`) older than 3 control periods (60 ms) | Damping, stop |
| Non-finite joint position / velocity or IMU quaternion / gyro | Damping, stop |
| In CONTROL: odometry older than 100 ms | Damping, stop |
| Non-finite observation or policy action | Zero action (default pose, torque-clipped) until finite again; logged once per transition |
| Non-finite motor target | Damping, stop |
| Motion plan with a NaN / inf value | Refused; the current reference keeps playing |
| In CONTROL, depth policy: no camera frame for 0.5 s | The policy is fed the far plane; logged |

Damping is latched: restart the process.

## Plan switching

`send_ref_motion(qpos, input_fps, counter=N, align_anchor=...)` swaps in a new plan while tracking (e.g. from the web
interface). It drops the `N` already-elapsed frames and rebases the clip onto the robot pose (`"robot"`), onto the
current reference frame (`"reference"`, keeps the reference continuous) or not at all (`None`); a plan with a NaN /
inf value is refused. The swap is atomic against the control tick, so a tick sees the old or the new reference,
never a mix, and the first tick on a new plan reads its frame 0. Plans streamed by the motion generator run on the
C++ tracker through [`system/`](../../system/README.md).

## Configuration

Each config in `g1/configs/` contains:

```yaml
robot_name: g1
xml_path: '{ROOT_DIR}/assets/robots/g1/mjcf/scene_g1_29dof_macmini_sysid.xml'   # MuJoCo model (the lab robot)
simulation_dt: 0.005             # 200 Hz physics
control_decimation: 4
control_dt: 0.02                 # 50 Hz policy
num_joints: 29
clip_observation: 100.0
clip_action: 100
training_joint_order: [...]      # policy joint order
deployment_joint_order: [...]    # Unitree motor order (the order of all arrays below)
default_angles: [...]            # target = default_angles + action_scale * action
action_scale: [...]
joint_kps: [...]                 # PD gains
joint_kds: [...]
max_torque: [...]                # limits of the torque-aware action clip
sim_kp_scale: [...]              # MuJoCo only (optional): the sim realises joint_kps x sim_kp_scale and
sim_kd_scale: [...]              #   joint_kds x sim_kd_scale, the gains training simulates; the robot ignores both
standing_joint_pos: [...]        # standing reference (stand-up, end of a motion)
standing_base_pos: [...]
standing_base_quat_wxyz: [...]
observation: {...}               # policy inputs: one group per ONNX input
depth_camera: {...}              # depth config only: training camera mount for MuJoCo
```

`default_angles`, `action_scale`, `joint_kps`, `joint_kds`, `max_torque` and `standing_joint_pos` (29 values each),
`standing_base_pos` (3) and `standing_base_quat_wxyz` (4) are required; there are no defaults. `sim_kp_scale` /
`sim_kd_scale` are optional (29 positive values each). In `observation:`, `anchor_mode` (only `odom`),
`history_length` and each group's `layout` (`flat`, `stack` or `image`) are required; unknown term names, a `flatten`
other than `term_major`, an image group without `height`, `width` or `clip`, and an `encode` other than `log1p` are
errors. Optional: `use_gpu` (default `true`). The robot, gains and timing blocks are identical in the two configs and
in the C++ configs of [`../deploy_cpp/configs`](../deploy_cpp/configs).

## Code layout

| Path | What |
|---|---|
| `g1/deploy_mujoco.py`, `g1/deploy_real.py`, `g1/play_real_web_interface.py` | Entry points |
| `g1/g1_controller.py` | G1 controller: config-driven observation, reference swaps |
| `g1/depth_sources.py`, `g1/variant_support.py` | MuJoCo / RealSense depth, depth camera rig, preflight |
| `utils/core/` | `base_controller.py` (joint maps, torque-aware clip, loop timing), `observation_builder.py`, `policy_runner.py` (ONNX Runtime), `history_buffer.py`, `motion_loader.py` (CSV loading, resampling, padding), `remote_controller.py` |
| `utils/backends/` | `mujoco_backend.py` (PD at the physics rate), `unitree_backend.py` (`unitree_sdk2py`: LowCmd / LowState / odometry) |
| `utils/math/`, `utils/web_interface.py` | Types and quaternion math; Flask dashboard |
| `models/g1/` | Drop-in folder for exported policies |

| Variable | Effect |
|---|---|
| `HALO_POLICY_CPU=1` | Run the policy on the ONNX Runtime CPU provider even if CUDA is available |
| `HALO_VIEWER_FPS` | MuJoCo viewer sync rate in Hz (default 60; 0 = every physics step) |
