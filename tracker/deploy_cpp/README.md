# Real-Time C++20 G1 Deployment

Hard-real-time C++ deployment of the HALO tracker on the Unitree G1, the counterpart of the Python
[`../deploy/`](../deploy/). It targets the robot's NVIDIA Jetson Orin NX (TensorRT, PREEMPT_RT), and builds on macOS /
Ubuntu for MuJoCo simulation.

| Policy | Config | ONNX inputs |
|---|---|---|
| proprio student | `configs/g1_v2.yaml` | `obs` (1, 434), `policy_temporal` (1, 30, 166) |
| depth student | `configs/g1_v2_depth.yaml` | + `student_depth` (1, 1, 48, 84) |

Pick the config that matches the export. The loop refuses to start when the config's observation groups do not match
the policy's input names. `configs/g1_v2_sim.yaml` and `configs/g1_v2_depth_sim.yaml` are MuJoCo variants with a
finer physics step (0.5 ms × 40 instead of 5 ms × 4, same 50 Hz policy) for sit / push contacts; on the robot use
`g1_v2.yaml` / `g1_v2_depth.yaml`.

The control loop runs **one policy inference per control tick** (50 Hz, from `control_dt`), so every command sent
is freshly computed from the state read that same tick. Each tick writes a PD setpoint that the motor controllers
hold until the next tick. The loop talks to the hardware and the depth thread through lock-free triple buffers, so it
never waits on them (the observation build still makes small Eigen allocations each tick). `control_decimation` in
the YAML is **physics sub-stepping**, not an inference rate: `control_dt = simulation_dt × control_decimation`, and
the MuJoCo backend advances physics that many times per tick. It does not affect the real robot.

The depth student's input comes from a D435i: raw IR stereo pair → LiteAnyStereo (TensorRT) on the Orin → metric depth
→ 84×48 ([docs/depth.md](docs/depth.md)). In MuJoCo, `deploy_g1_mujoco --depth_render` renders the training camera.

No policy weights ship with the repository: export one with
[`../training/scripts/rsl_rl/play.py`](../training/scripts/rsl_rl/play.py) and see
[`../deploy/models/g1/`](../deploy/models/g1/README.md). No motion clips ship either; see
[`../data/README.md`](../data/README.md).

Code layout, CMake targets, RT tuning, timing traces and troubleshooting: [docs/reference.md](docs/reference.md).

---

## Build options

Quick sanity check on any host with a C++20 compiler (no robot, no GPU, no SDK):

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug -j
```

Every optional dependency is gated behind a CMake flag. None are required for the plain build above.

| Option | Default | Pulls in | Required for |
|---|:---:|---|---|
| `RT_DEPLOY_WITH_TRT` | OFF | CUDA + TensorRT | `*.plan` inference, `onnx_to_trt`, the cleaned depth producer |
| `RT_DEPLOY_WITH_ORT` | OFF | ONNX Runtime | `*.onnx` inference (the only engine on macOS) |
| `RT_DEPLOY_WITH_MUJOCO` | OFF | MuJoCo + GLFW | `deploy_g1_mujoco` |
| `RT_DEPLOY_WITH_UNITREE` | OFF | Unitree SDK2 (DDS, Linux only) | `deploy_g1_real`, `g1_state_monitor` |
| `RT_DEPLOY_WITH_REALSENSE` | OFF | librealsense2 (+ CUDA/TensorRT for the cleaned path) | D435i depth sources for `student_depth` ([docs/depth.md](docs/depth.md)) |
| `RT_DEPLOY_ENABLE_SANITIZERS` | OFF | asan / ubsan | `Debug` builds with sanitizers |
| `RT_DEPLOY_ENABLE_WERROR` | **ON** | | treats warnings as errors; turn off only to triage a new toolchain |

Three cache variables tune the target rather than gate a dependency:

| Variable | Default | Meaning |
|---|:---:|---|
| `RT_DEPLOY_JETSON_PLATFORM` | `auto` | `auto \| orin \| thor \| none`. `auto` reads `/proc/device-tree/model` at configure time ([reference](docs/reference.md#jetson-platform-auto-detection)). |
| `RT_DEPLOY_STRIP_UNITREE_SERVICE_HOOK` | `ON` | With `RT_DEPLOY_WITH_UNITREE=ON`: link a copy of the prebuilt `libunitree_sdk2.a` with Unitree's onboard-service objects removed (their static constructor can exit the process before `main`). See `cmake/FindUnitreeSDK2.cmake`. |
| `RT_DEPLOY_CUDA_ARCH` | `87` | CUDA architecture for the depth kernels (only with `RT_DEPLOY_WITH_REALSENSE=ON` + `RT_DEPLOY_WITH_TRT=ON`). `87` = Orin (sm_87). |

**A binary needs its backend flag *and* at least one inference-engine flag**, e.g. `deploy_g1_real` needs
`RT_DEPLOY_WITH_UNITREE=ON` plus one of `RT_DEPLOY_WITH_TRT=ON` / `RT_DEPLOY_WITH_ORT=ON`. **CMake skips a binary
silently when its gates aren't met**, so a missing target is a missing flag, not a broken build. The configure line
prints what got enabled:

```
-- [rt_deploy] build=Release  trt=OFF  ort=OFF  unitree=OFF  mujoco=OFF  realsense=OFF  jetson=none
```

Presets (`CMakePresets.json`) use Ninja, build into `build/<preset>`, and set `CMAKE_POLICY_VERSION_MINIMUM=3.5`.
Dev hosts: `debug`, `debug-sanitize`, `release`, `relwithdebinfo`. Jetson (`Release-RT` build type):
`release-rt-orin` (**the G1**, JetPack 5 or 6; plans `*_orin.plan`), `release-rt-thor` (no G1 ships on one) and
`release-rt` (auto-detects the platform, [reference](docs/reference.md#jetson-platform-auto-detection)).
`release-offline`: air-gapped host with vendored `third_party/` ([offline builds](#offline-builds)).

**Unconditional dependencies:** **Eigen 5.0** and **yaml-cpp**, fetched via `FetchContent` (Eigen 5.0.0,
yaml-cpp 0.8.0) on the first configure if not found system-wide. That needs outbound network to GitHub / GitLab; see
[offline builds](#offline-builds) otherwise.

---

## Build on a dev host (macOS or Ubuntu)

Build and run the tracker in MuJoCo on a laptop or workstation, with or without a viewer. **The real-robot binary
does not build on macOS**: it needs Linux plus the Unitree SDK. On macOS you get `deploy_g1_mujoco`, with ONNX Runtime
as the only inference backend (no CUDA, no TensorRT).

Core libraries only:

```bash
brew install cmake ninja eigen yaml-cpp                               # macOS
sudo apt install cmake ninja-build libeigen3-dev libyaml-cpp-dev      # Ubuntu
```

### Ubuntu

For `deploy_g1_mujoco` (headless MuJoCo with ONNX inference), add GLFW and EGL, build MuJoCo as in step 2 of the
macOS section below, and use the ONNX Runtime release tarball:

```bash
sudo apt install g++ libglfw3-dev libgl1-mesa-dev libegl-dev curl
cd tracker/deploy_cpp            # from the repository root
# MuJoCo 3.3.6 into third_party/mujoco: step 2 below, same commands
curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz \
    | tar xz -C third_party
export ONNXRUNTIME_ROOT=$(pwd)/third_party/onnxruntime-linux-x64-1.20.1
cmake --preset release -DRT_DEPLOY_WITH_MUJOCO=ON -DRT_DEPLOY_WITH_ORT=ON \
    -Dmujoco_DIR=$(pwd)/third_party/mujoco/lib/cmake/mujoco
cmake --build build/release --target deploy_g1_mujoco -j
export LD_LIBRARY_PATH=$(pwd)/third_party/mujoco/lib:$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH
```

Then [run it in MuJoCo](#run-in-mujoco), with `--no_viewer` on a machine without a display.

### macOS (with the MuJoCo viewer)

These steps add GLFW, ONNX Runtime and a MuJoCo install that ships a CMake config. MuJoCo goes under
`deploy_cpp/third_party/` (the folder is `.gitignore`d). Neither TensorRT nor the Unitree SDK exists on macOS.

```bash
# 1. system deps
brew install cmake ninja eigen yaml-cpp glfw onnxruntime

# 2. MuJoCo from source into third_party/ (no Homebrew formula; the macOS DMG has no CMake config; ~2 min)
cd tracker/deploy_cpp            # from the repository root
mkdir -p third_party && cd third_party
git clone --depth 1 --branch 3.3.6 https://github.com/google-deepmind/mujoco.git mujoco-src
cmake -S mujoco-src -B mujoco-src/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DMUJOCO_BUILD_EXAMPLES=OFF -DMUJOCO_BUILD_SIMULATE=OFF -DMUJOCO_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=$(pwd)/mujoco
cmake --build mujoco-src/build --target install -j
cd ..

# 3. configure + build
ONNXRUNTIME_ROOT=$(brew --prefix onnxruntime) \
cmake --preset release -DRT_DEPLOY_WITH_MUJOCO=ON -DRT_DEPLOY_WITH_ORT=ON \
    -Dmujoco_DIR=$(pwd)/third_party/mujoco/lib/cmake/mujoco \
    -Dglfw3_DIR=$(brew --prefix glfw)/lib/cmake/glfw3
cmake --build build/release --target deploy_g1_mujoco -j
export DYLD_LIBRARY_PATH=$(pwd)/third_party/mujoco/lib:$DYLD_LIBRARY_PATH
```

- `ONNXRUNTIME_ROOT` works as a shell env var or as a `-DONNXRUNTIME_ROOT=...` cache variable
  ([`cmake/FindONNXRuntime.cmake`](cmake/FindONNXRuntime.cmake) tries the cache var first, then
  `$ENV{ONNXRUNTIME_ROOT}`, then standard system prefixes).
- `glfw3_DIR` points at the CMake config Homebrew ships, typically `/opt/homebrew/opt/glfw/lib/cmake/glfw3`. If CMake
  still can't find it, pass `-DCMAKE_PREFIX_PATH=$(brew --prefix glfw)` as a catch-all.
- Clean reset: `rm -rf build/release third_party` (from `tracker/deploy_cpp`).

Then [run it in MuJoCo](#run-in-mujoco).

---

## Build on the G1 (Jetson Orin NX)

The G1 runs an Orin, so every command here uses the `release-rt-orin` preset (Thor:
[reference](docs/reference.md#jetson-thor-jetpack-7x-cuda-13-tensorrt-10)). Orin ships in two software stacks and the
build supports both. Check yours with `cat /etc/nv_tegra_release` (**R35** → JetPack 5, **R36** → JetPack 6).

### JetPack 6.x (L4T R36, CUDA 12, TensorRT 10, gcc-11)

```bash
sudo apt install cmake ninja-build g++-11 libeigen3-dev libyaml-cpp-dev
# CUDA 12.x + TensorRT 10.x arrive with JetPack 6.
```

### JetPack 5.x (L4T R35, CUDA 11.4, TensorRT 8.5, gcc-10): the stock G1

The G1's onboard Orin NX ships on Ubuntu 20.04 (focal). Three things differ from JetPack 6:

- **CMake ≥ 3.22** (`cmake_minimum_required`; the presets file is schema `version 2`, ≥ 3.20). focal's cmake 3.16
  doesn't understand `--preset` and fails with the misleading `The source directory "…/<preset-name>" does not
  exist`; a cmake with presets but an older schema fails with `Could not read presets … Unrecognized "version"
  field`. Stage a newer one without sudo: `pip install --user "cmake>=3.22" && hash -r` (or Kitware's apt repo).
  **Don't `apt install cmake`** afterward: it reinstalls 3.16 and can shadow the new one on `PATH`.
- **gcc-10, passed explicitly.** focal has no `g++-11`, and the default gcc-9 has no `<span>` (C++20 build fails).
  Install `g++-10` and configure with `CC=gcc-10 CXX=g++-10`.
- **CUDA 11.4 + TensorRT 8.5.** The engine and `onnx_to_trt` carry version guards, so no source change is needed, but
  a `.plan` built here is **not** portable to a JetPack 6 robot.

```bash
sudo apt install ninja-build g++-10 libeigen3-dev libyaml-cpp-dev
```

**Conda toolchain (e.g. `~/miniconda3`).** Two conda pieces collide with the CUDA-11.4-mandated g++-10:

- **conda's `cmake` is often 4.x**, which dropped `cmake_minimum_required(VERSION < 3.5)`; yaml-cpp 0.8.0 and the
  other FetchContent deps declare `3.4` and abort with *"Compatibility with CMake < 3.5 has been removed"*.
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` raises the floor for those deps only (the presets already set it). Or
  `conda install "cmake<4"`.
- **conda's `libyaml-cpp.so` is built with GCC 11+** and needs `std::__throw_bad_array_new_length()@GLIBCXX_3.4.29`,
  absent from g++-10's libstdc++; `find_package(yaml-cpp)` picks it up and the `deploy_g1_real` link dies with
  *"undefined reference to … @GLIBCXX_3.4.29"*. `-DCMAKE_DISABLE_FIND_PACKAGE_yaml-cpp=ON` builds yaml-cpp from
  source with g++-10 instead (static, ABI-matched).

The full flow that configures and builds cleanly on the stock conda G1:

```bash
cd tracker/deploy_cpp               # from the repository root
rm -rf build/release-rt-orin        # clear any half-configured cache first

CC=gcc-10 CXX=g++-10 cmake --preset release-rt-orin -DRT_DEPLOY_WITH_TRT=ON -DRT_DEPLOY_WITH_UNITREE=ON \
    -DCMAKE_DISABLE_FIND_PACKAGE_yaml-cpp=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build/release-rt-orin --target deploy_g1_real onnx_to_trt -j
```

No `libstdc++` shim or rpath is then needed. If you must link some *other* prebuilt conda library that needs
`GLIBCXX_3.4.29`, add `-DCMAKE_EXE_LINKER_FLAGS="-L<dir-with-conda-libstdc++> -Wl,-rpath,$CONDA_PREFIX/lib"` so the
link and the runtime both resolve against conda's newer libstdc++.

### Unitree SDK2

The stock G1 carries the full SDK2 C++ at `/opt/unitree_robotics` (the copy that includes
`unitree/dds_wrapper/common/crc.h`). Point the build at it with `export UNITREE_SDK2_ROOT=/opt/unitree_robotics`;
otherwise a partial header tree under `/usr/local/include` can shadow it and the build fails on a missing `crc.h`.
A from-scratch install (examples off; they do not all link):

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cmake -S unitree_sdk2 -B unitree_sdk2/build -DBUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics
cmake --build unitree_sdk2/build -j && sudo cmake --install unitree_sdk2/build
export UNITREE_SDK2_ROOT=/opt/unitree_robotics
```

MuJoCo is only needed for the `deploy_g1_mujoco` sim binary, never on the robot (`pip install mujoco` ships the
`.so` but no CMake config; build from source for the config). PREEMPT_RT kernel patches and `isolcpus` boot args:
[reference → tuning](docs/reference.md#tuning-on-the-target).

### Build

```bash
export PRESET=release-rt-orin      # the G1's Jetson
export BUILD=build/${PRESET}
cd tracker/deploy_cpp              # from the repository root

# Configure + build (TRT + MuJoCo + Unitree DDS).
cmake --preset ${PRESET} -DRT_DEPLOY_WITH_TRT=ON -DRT_DEPLOY_WITH_MUJOCO=ON -DRT_DEPLOY_WITH_UNITREE=ON
cmake --build ${BUILD} --target deploy_g1_mujoco deploy_g1_real onnx_to_trt -j

# Grant RT capabilities so SCHED_FIFO + mlockall succeed without root.
sudo setcap 'cap_sys_nice=eip cap_ipc_lock=eip' ${BUILD}/deploy_g1_real
```

**Stock G1 (JetPack 5.x):** configure with `CC=gcc-10 CXX=g++-10` and `export UNITREE_SDK2_ROOT=/opt/unitree_robotics`,
and drop `-DRT_DEPLOY_WITH_MUJOCO=ON` (plus the `deploy_g1_mujoco` target): the sim binary isn't needed on the robot
and MuJoCo isn't installed there. Add the conda flags above if the toolchain comes from conda. For the depth student
add `-DRT_DEPLOY_WITH_REALSENSE=ON` ([docs/depth.md](docs/depth.md#build)).

Re-run `setcap` after every rebuild (caps live on the inode). A setcap'd binary ignores `LD_LIBRARY_PATH`
([reference → tuning](docs/reference.md#tuning-on-the-target)).

### Offline builds

The cold configure pulls Eigen 5.0.0 and yaml-cpp 0.8.0 via `FetchContent`. If the robot can't reach `gitlab.com` /
`github.com`, pre-stage those sources on a connected machine and rsync them over with the project:

```bash
# On a host with internet, from tracker/
deploy_cpp/tools/vendor_third_party.sh
# clones into deploy_cpp/third_party/{eigen,yaml-cpp} at the pinned tags

rsync -av --exclude build/ deploy_cpp/ <robot>:~/tracker/deploy_cpp/

# On the robot
cmake --preset release-offline -DRT_DEPLOY_WITH_TRT=ON -DRT_DEPLOY_WITH_UNITREE=ON
cmake --build build/release-offline --target deploy_g1_real -j
```

`release-offline` sets `FETCHCONTENT_FULLY_DISCONNECTED=TRUE` and points `FETCHCONTENT_SOURCE_DIR_*` at
`third_party/`, so the configure step never touches the network.

---

## Convert a policy to TensorRT

Convert your exported ONNX policy to a TensorRT plan on the Orin that will run it: FP32 IO, FP16 inside.

```bash
${BUILD}/onnx_to_trt --onnx policy.onnx --plan policy.plan --weakly-typed --fp16
```

[`tools/build_onnx_to_trt.sh`](tools/build_onnx_to_trt.sh) wraps this (builds the tool into `build/g1_nx`, converts
one ONNX or every `*.onnx` in `../deploy/models/g1/`) and also decomposes LayerNorm for TRT 8.5. A `.plan` is not
portable across `{GPU SM, CUDA version, TRT version}`.

---

## Run in MuJoCo

Run in the sim first (no hardware involved). The motion is a CSV clip you supply ([`../data/README.md`](../data/README.md)).
On a dev host the policy is the `.onnx` (ONNX Runtime). A Jetson build with MuJoCo can also run the `.plan`; the stock
G1 build leaves MuJoCo out ([Build](#build)), so on the robot run the plan with `deploy_g1_real` instead.

```bash
# proprio student (policy.onnx = your own export; no weights ship)
./build/release/deploy_g1_mujoco --config configs/g1_v2.yaml \
    --policy ../deploy/models/g1/policy.onnx --motion <motion.csv> --motion_fps 30 --duration 60

# depth student: render the training depth camera into student_depth
./build/release/deploy_g1_mujoco --config configs/g1_v2_depth.yaml \
    --policy ../deploy/models/g1/depth_policy.onnx --motion <motion.csv> --depth_render

# on a Jetson built with -DRT_DEPLOY_WITH_MUJOCO=ON and MuJoCo installed (not the stock G1 build, see Build), with
# the TensorRT plan
${BUILD}/deploy_g1_mujoco --config configs/g1_v2.yaml --policy policy.plan \
    --motion ../data/<motion>.csv --motion_fps 30 --duration 60

# add --no_viewer for a headless run
```

A config with an image group (`g1_v2_depth.yaml`) is refused without `--depth_render`, and `--depth_render` is refused
with a proprio-only config.

---

## Run on the robot

### 1. Bring-up: check the DDS receive path

Before the first `deploy_g1_real` run on a new robot or network, run the read-only state monitor. It opens only the
`rt/lowstate` subscriber, prints a live dashboard of joints / IMU / remote / rx counters, and **never calls
`send_command`**: no byte is ever published back to the robot.

```bash
cmake --build build/release-rt-orin --target g1_state_monitor -j
build/release-rt-orin/g1_state_monitor --net enP2p1s0
```

If `rx` stays at 0 after 10 s, `wait_for_state` prints a targeted checklist (interface up, domain match,
`CYCLONEDDS_URI`, etc.). If `crc_rejects` climbs at the same rate as `rx`, disable the CRC check (default already
off): your Unitree SDK version differs from the robot firmware's.

### 2. Dry run (`--no_send`)

`deploy_g1_real --no_send` runs the whole pipeline on the real RT thread (DDS state, observation, inference, action
decode) but publishes nothing: every `LowCmd_` is built and CRC'd as normal, then discarded. It exists for timing runs
that need the real load but no actuation. `--start_in_control` enters `CONTROL` at tick 0 (inference only runs in
`CONTROL`); it skips the ramp to the default pose, so the binary refuses it without `--no_send`.

> `--no_send` is **not a safety mode**. The robot is never commanded at all, including the damping command normally
> sent on exit and on SELECT/E-stop; what the joints do when `lowcmd` stops arriving is firmware behaviour this
> program does not control. Support the robot on a stand and disable its motors or engage the E-stop. At exit the
> backend prints `DRY RUN: N commands built and withheld; 0 published`. The interlock lives only in
> `deploy_g1_real`'s argument parser.

### 3. Real run

```bash
# Press Start to move to default; A to begin control; SELECT for E-stop.
${BUILD}/deploy_g1_real --net eth0 --config configs/g1_v2.yaml --policy policy.plan \
    --motion ../data/<motion>.csv --rt_prio 80 --rt_cpu 1 --lock_memory --profile
```

The depth student adds `--config configs/g1_v2_depth.yaml --depth_config configs/depth_d435i.yaml`
([docs/depth.md](docs/depth.md#run)).

### Operator controls (Unitree wireless remote)

| Input | Effect |
|---|---|
| **Start** | Move to the default position (2 s cosine ramp, kp × 0.5). |
| **A** | Begin policy control (DEFAULT_POS → CONTROL). Ignored until every input the policy needs is live: odometry (the observation's anchor, `anchor_mode: odom`), and a depth frame for the depth student. A refused press must be released and pressed again. |
| **B** (in CONTROL) | Switch the reference to the standing motion, yaw-aligned to the robot's current heading (mirrors Python `do_standup`). In the `system/cpp` tracker nodes (no CSV motion) the first B also latches `tracker_ready`, which tells the generator to start. |
| **A** (in CONTROL) | Replay the originally loaded CSV motion from frame 0, yaw-aligned (mirrors Python `replay_original_motion`). No-op in the `system/cpp` tracker nodes, which boot without a CSV motion. |
| **SELECT** | Emergency stop (damps + exits). |
| **Ctrl-C / SIGTERM** | Clean shutdown via damping. Do not use SIGKILL: the process must reach `cleanup()` to damp. |

When a `--motion` clip ends before `--duration`, the reference does not jump to the standing pose: the joints ramp
linearly from the last clip frame to `standing_joint_pos` over K frames (K = 10 for a gap under 0.3 rad, else
trunc(gap / 0.3 × 10), at most 100, i.e. 2 s at 50 Hz), then hold it; the root holds the last frame. The Python
deployment pads with the same rule.

### Safety behaviour

- **Odometry.** The binary exits (code 1) if no odometry (`rt/odommodestate`) arrives within 2 s of the first state.
  Odometry older than 100 ms counts as lost; losing it in CONTROL damps and exits like a stale state (fault
  `stale_odom`). There is no fallback to the IMU attitude.
- **Stale or bad state.** No fresh DDS state for > 60 ms (3 × the 20 ms control period) latches `fault=stale_state`;
  a NaN IMU reading latches `fault=state_non_finite`. Both damp. The joint-limit latch is off by default
  ([troubleshooting](docs/reference.md#troubleshooting)).
- **Control rate.** Both G1 binaries force the rate to `1/control_dt` from the config and ignore `--control_hz`: a
  wrong rate on real hardware is unsafe.
- **Depth.** The depth-student config refuses to start without a depth source unless `--no_send`. Depth producer
  failures are fatal at startup; after start, stale depth falls back to the far plane
  ([docs/depth.md](docs/depth.md#depth-staleness-during-a-live-run)).
- **Torque-aware target clip.** Each target is clipped so that the motor's PD torque stays inside `max_torque`
  ([reference](docs/reference.md#action-decoding-and-sim-plant-drive)).

---

## Runtime-pushed plans

The tracker is a pure tracker/control library with **zero ZMQ**. Motion plans streamed at run time by the HALO motion
generator are served by the `system/` package of this repository: its tracker nodes
(`system/cpp/nodes/motiontrack_{mujoco,real}_zmq.cpp`, [`../../system/cpp`](../../system/cpp)) link `rt_deploy::g1`
and wire this control loop (`run_control_loop` + the public `TripleBuffer` / `RawMotionPlan` / `ActivePlanSnapshot` /
trim-atomic seam) to the `system/comm/` (halo_comm) protobuf middleware. Resampling, joint remap, derivatives and the
anchor rebase (`g1::send_ref_motion`) happen on the control loop; only the wire lives outside. See
[`system/README.md`](../../system/README.md) for the bring-up, the wire contract and the node flags, and
[reference → plan-switch contract](docs/reference.md#plan-switch-contract) for what the plan server relies on.

---

## CLI flags

All G1 binaries use `snake_case` flags. Numeric values are parsed strictly: a typo such as `--rt_cpu 1x` is an error,
not `1`. "both" = `deploy_g1_real` and `deploy_g1_mujoco`.

| Flag | Binaries | Default | Meaning |
|---|---|---|---|
| `--config PATH` | both | `configs/g1_v2.yaml` | YAML config |
| `--policy PATH` | both | required | `.plan` (TensorRT) or `.onnx` (ONNX Runtime) |
| `--engine auto\|trt\|ort` | both | `auto` (by file extension) | Inference backend |
| `--motion PATH` | both | required | CSV motion file; a bare name resolves to `tracker/data/<name>.csv` (no clips ship, see [`../data/README.md`](../data/README.md)) |
| `--motion_fps N` | both | 30 | CSV input rate, 1..1000 |
| `--duration S` | both | 60000 | Max runtime, seconds (> 0; rounded to whole ticks, at least one) |
| `--control_hz N` | both | from `control_dt` | 1..1000; both binaries override it from `cfg.control_dt` (50 Hz with the stock config) |
| `--rt_prio N` | both | 0 (off) | SCHED_FIFO priority 1..99 |
| `--rt_cpu N` | both | -1 (unpinned) | Pin the control thread to a core |
| `--lock_memory` | both | off | `mlockall(CURRENT\|FUTURE)` |
| `--profile` | both | off | Per-thread Hz + compute stats every 2 s |
| `--trace_csv PATH` | both | off | Per-iteration timing to CSV ([reference](docs/reference.md#loop-timing-traces)) |
| `--depth_stats_every N` | both | 0 (off) | Log the actual `student_depth` tensor + action every N inference ticks ([reference](docs/reference.md#depth--action-tensor-dump)) |
| `--depth_stale_far_ticks N` | both | 25 (500 ms at 50 Hz) | Feed the far plane after N stale depth ticks (0 = hold forever; -1 = keep the default) |
| `--allow_depth_clip_mismatch` | both | off (refuse) | Start even if the config's depth clip differs from the training clip |
| `--sim_pause_after_plan_s S` | `deploy_g1_mujoco` | 0 (off) | Sim eval only: hold the playback counter S s after a plan installs; refused by `deploy_g1_real` and `motiontrack_real_zmq` |
| `--net IFACE` | `deploy_g1_real` | `eth0` | DDS network interface |
| `--depth_config YAML` | `deploy_g1_real` | none | Start a depth producer from `configs/depth_d435i.yaml`; omit it and no camera is opened. Required by `g1_v2_depth.yaml` unless `--no_send` (then the loop warns and feeds the far plane). See [depth](docs/depth.md). |
| `--depth_source cleaned\|raw` | `deploy_g1_real` | from the YAML | Override the producer choice, for raw-vs-cleaned A/B runs |
| `--depth_plan PATH` | `deploy_g1_real` | from the YAML | Override the stereo TensorRT plan (cleaned producer only) |
| `--no_send` | `deploy_g1_real` | off | Dry run: build every command, publish none ([dry run](#2-dry-run---no_send)) |
| `--start_in_control` | `deploy_g1_real` | off | Enter CONTROL at tick 0; refused without `--no_send` |
| `--no_viewer` | `deploy_g1_mujoco` | off | Headless run |
| `--no_align_motion` | `deploy_g1_mujoco` | off | Play the motion in its own world frame: no XY, yaw or Z anchoring onto the robot's spawn pose (by default frame 0 is placed on the robot) |
| `--depth_render` | `deploy_g1_mujoco` | off | Render the sim depth camera into the depth inbox (the sim counterpart of `--depth_config`). The render is **synchronous inside `step()` on the control thread**, every `obs_decim`-th step, so it adds per-tick work. Requires a config with an image group. |

`g1_state_monitor` flags:

| Flag | Default | Effect |
|---|---|---|
| `--net IFACE` | `eth0` | DDS network interface |
| `--domain N` | 0 | Non-default DDS domain |
| `--verify_crc` | off | Enable the application-level lowstate CRC check |
| `--rate HZ` | 10 | Dashboard redraw rate, 1..200 |
| `--once` | off | Print a single snapshot and exit (good for logs / CI) |
| `--verbose` | off | Also print joint velocities and dim inactive buttons |

---

## Credits

- The C++ deployment mirrors the Python deployment in [`../deploy/`](../deploy/): same joint order, gains,
  observation layout and action decoding.
- Unitree SDK2 and DDS conventions follow the official `unitree_sdk2` headers and examples.
