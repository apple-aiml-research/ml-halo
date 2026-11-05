# system C++ tracker nodes

The **production tracker** side of the HALO ZMQ system. These C++
nodes run tracker's hard-real-time G1 control loop and expose it to an external
motion **generator** over the halo_comm middleware (protobuf services + a state
topic).

They are the only trackers: the Python generator node
(`motiongen_zmq.py`, see [`../README.md`](../README.md))
drives either over the **identical wire channels**.

## Layout

```
system/cpp/
├── nodes/
│   ├── motiontrack_mujoco_zmq.cpp   # tracker on a MuJoCo sim backend
│   └── motiontrack_real_zmq.cpp     # tracker on the real Unitree G1 (DDS) backend
├── transport/
│   └── tracker_server.hpp  # TrackerServer: the 4 channels ⇄ tracker's control loop
└── CMakeLists.txt
```

## How it fits together

Dependency direction: **`halo_system → { tracker, halo_comm }`**.

- **tracker** (`rt_deploy::*`, pulled in via `add_subdirectory`) — pure control:
  the 50 Hz G1 control loop + backends (MuJoCo / Unitree DDS). No transport.
- **halo_comm** (`halo_comm::halo_comm`) — transport only: the ROS 2-style `Node` (protobuf
  over ZMQ/TCP). No tracker dependency.
- **`transport/tracker_server.hpp`** — the glue that lives *here*: a `TrackerServer`
  binds the four channels on a `halo_comm::Node` and drives tracker's control loop
  through its public `TripleBuffer` / `RawMotionPlan` / `ActivePlanSnapshot` /
  trim-atomic seam, with protobuf on the wire.

### Channels (defined here — the wire's source of truth)

`transport/tracker_server.hpp` binds:

| Channel | Kind | Port | Message |
|---|---|---|---|
| `g1/get_state` | service (REQ/REP) | 5555 | `StateRequest → RobotState` (+ optional ego `Image`) |
| `g1/put_plan`  | service (REQ/REP) | 5556 | `MotionPlan → PlanAck` |
| `g1/get_trim`  | service (REQ/REP) | 5557 | `TrimRequest → TrimReply` (RTC) |
| `g1/state`     | topic (PUB/SUB, 50 Hz, conflate) | 5558 | `RobotState` (no image) |
| `g1/depth` | topic (PUB/SUB) | 5553 | `DepthImage` (MuJoCo node with --depth_render) |

See [`../README.md`](../README.md) for the
message-field details and the generator-side RTC loop. Protobuf schema lives in
[`../comm/msgs/`](../comm/README.md).

### Threading

One server thread runs the unified loop: it is the **sole consumer** of the state
outbox (refreshing a cached snapshot once per iteration, read by both the
`get_state` handler and the 50 Hz publisher) and the **sole producer** of the
motion inbox. The 50 Hz control loop touches only TripleBuffers / atomics; all
protobuf (de)serialization happens on the server thread, **off** the control
thread — so the SPSC TripleBuffer contract is never violated.

## Build

Built from `system/cpp/` via CMake. `system/cpp` `add_subdirectory`s both
tracker and halo_comm, so a single configure+build produces the tracker node.
`RT_DEPLOY_BUILD_TESTS=OFF` skips tracker's googletest FetchContent (needs
network); Eigen3 / yaml-cpp / glfw3 resolve from Homebrew on macOS.

**Protobuf (one-time / on schema change):**
```bash
cd system/comm && make protoc29 && make protos    # protoc -> python/halo_comm/msgs/*_pb2.py
```
Run `make protoc29` once first; it fetches the pinned protoc 29 (gencode 5.29.x,
matching the protobuf 5.29 runtime), and `make protos` refuses any other protoc.
See system/comm/README.md.

**Python (transport + control lib + orchestration):**
```bash
pip install -e system/comm/python       # halo_comm (Node + pb2)
pip install -e tracker   # tracker control lib
pip install -e system              # generator + tracker nodes
```

**C++ — MuJoCo tracker (dev box):** `build_motiontrack_mujoco_zmq.sh` builds MuJoCo
3.2.5 into `system/build/third_party/mujoco-3.2.5` on its first run; the manual form:

```bash
MUJOCO_CMAKE_DIR=$(pwd)/system/build/third_party/mujoco-3.2.5/lib/cmake/mujoco
cmake -S system/cpp -B system/build/mac -G Ninja \
  -DRT_DEPLOY_WITH_MUJOCO=ON -DRT_DEPLOY_WITH_ORT=ON \
  -DRT_DEPLOY_BUILD_TESTS=OFF -DRT_DEPLOY_ENABLE_WERROR=OFF \
  -DONNXRUNTIME_ROOT=$(brew --prefix onnxruntime) \
  -DEigen3_DIR=/opt/homebrew/share/eigen3/cmake \
  -Dyaml-cpp_DIR=/opt/homebrew/lib/cmake/yaml-cpp \
  -Dmujoco_DIR="$MUJOCO_CMAKE_DIR" \
  -Dglfw3_DIR=$(brew --prefix glfw)/lib/cmake/glfw3
cmake --build system/build/mac --target motiontrack_mujoco_zmq
```

Or use the helpers (run setup once, then build; the build script also builds MuJoCo 3.2.5 on first run):
```bash
bash system/cpp/scripts/setup_dev_env.sh            # idempotent env setup
bash system/cpp/scripts/build_motiontrack_mujoco_zmq.sh   # -> build/mac/motiontrack_mujoco_zmq
```

**C++ — real-robot tracker (Linux + Unitree SDK):** the NX / TensorRT build is
below (§ *TensorRT (Jetson / NX)*); the short form on a Linux box:

```bash
cmake -S system/cpp -B system/build/linux -G Ninja \
  -DRT_DEPLOY_WITH_UNITREE=ON -DRT_DEPLOY_WITH_ORT=ON
cmake --build system/build/linux --target motiontrack_real_zmq
```

Build gating (from [`CMakeLists.txt`](CMakeLists.txt)):

- `motiontrack_mujoco_zmq` requires `RT_DEPLOY_WITH_MUJOCO=ON` **and** `RT_DEPLOY_WITH_ORT=ON`.
- `motiontrack_real_zmq` requires `RT_DEPLOY_WITH_UNITREE=ON` **and** (`_ORT` or `_TRT`).
  Run it with `CAP_SYS_NICE` + `CAP_IPC_LOCK` (RT scheduling + locked memory).

### TensorRT (Jetson / NX)

On the NX, run the policy through **TensorRT** instead of ONNX Runtime: build with
`-DRT_DEPLOY_WITH_TRT=ON`, convert the `.onnx` policy to a serialized TensorRT
`.plan`, and run the tracker with that `.plan`.

tracker's [`cmake/FindTensorRT.cmake`](../../tracker/deploy_cpp/cmake/FindTensorRT.cmake)
finds TRT under the JetPack default `/usr/lib/aarch64-linux-gnu` (and
`/usr/local/tensorrt`), or set `TENSORRT_ROOT` for a custom install. CUDA comes
from `find_package(CUDAToolkit)`. (Verified against TensorRT 10.x.)

#### Quick path — helper scripts

Two scripts wrap the steps below. Each builds the thing it owns: the **tracker**
is a halo_system binary, the **onnx_to_trt** tool is a tracker tool, so the
scripts live next to their respective CMake sources. **Run both on the NX** (a
`.plan` is device-specific).

| # | Script (run on the NX) | Builds | Output |
|---|---|---|---|
| 1 | `system/cpp/scripts/build_motiontrack_real_zmq.sh` | `motiontrack_real_zmq` (TensorRT + Unitree) | `system/build/g1_nx/motiontrack_real_zmq` |
| 2 | `tracker/deploy_cpp/tools/build_onnx_to_trt.sh` | `onnx_to_trt` + converts ONNX → `.plan` | `<onnx-dir>/<onnx-basename>.plan`, next to each source ONNX |

**NX prerequisites (JetPack 5 / Ubuntu 20.04).** The system packages are too old
for this C++20 + protobuf codebase, so install newer ones once:
```bash
sudo apt-get install -y gcc-10 g++-10 ninja-build cmake    # system gcc 9 lacks <span>
conda install -c conda-forge libprotobuf                   # matched modern protoc + libprotobuf (system 3.6 untested)
```
Both scripts then pick these up **automatically**: they default `CC=gcc-10`/
`CXX=g++-10`, add `$CONDA_PREFIX` to `CMAKE_PREFIX_PATH` when a conda env is
active (so CMake finds conda's modern `protoc`+`libprotobuf`), and bridge conda's
newer `libstdc++.so.6` into the gcc-10 link (conda's prebuilt libs need a newer
libstdc++ than gcc-10 ships — see wall ③ below).

```bash
# 1. Build the real-robot tracker (motiontrack_real_zmq, TensorRT + Unitree):
system/cpp/scripts/build_motiontrack_real_zmq.sh
#    -> system/build/g1_nx/motiontrack_real_zmq

# 2. Build the onnx_to_trt tool AND convert ONNX policies to .plan files
#    (FP32 IO, FP16 inside, as in tracker/deploy_cpp/README.md):
tracker/deploy_cpp/tools/build_onnx_to_trt.sh [model.onnx] [out.plan] --weakly-typed --fp16
#    model.onnx omitted: converts every *.onnx in tracker/deploy/models/g1/
#    out.plan   default: <onnx-dir>/<onnx-basename>.plan (next to the ONNX)
#    -> e.g. tracker/deploy/models/g1/motiontrack_all_data.plan
#    The tool itself lands in tracker/deploy_cpp/build/g1_nx/onnx_to_trt.
```

Env knobs (both scripts): `CC`/`CXX` (compiler; default `gcc-10`/`g++-10`, use
`g++-11` on JetPack 6), `CONDA_PREFIX` (auto-added to `CMAKE_PREFIX_PATH`;
`tracker/deploy_cpp/tools/build_onnx_to_trt.sh` also honours `NO_CONDA_PREFIX=1`), `TENSORRT_ROOT`, `BUILD_TYPE` (default
`Release`), `JOBS`. Extra arguments: `build_motiontrack_real_zmq.sh` passes them to CMake
(`-D…`); `build_onnx_to_trt.sh` passes them to `onnx_to_trt`, which accepts only
`--fp16 --weakly-typed --workspace <GB> --device <N> --verbose`. The
scripts auto-wipe their `build/g1_nx` when the compiler changes. Keep the conda
env active at **runtime** too (so `libprotobuf.so` resolves), or set
`LD_LIBRARY_PATH=$CONDA_PREFIX/lib`.

> **JetPack 5 troubleshooting (the three "too old" walls).** This stack targets
> **JetPack 6** (gcc-11, TensorRT 10, modern protobuf). On JetPack 5 you must work
> around all of these: ① `fatal error: span` → gcc 9 lacks C++20 `<span>`; use
> `gcc-10`/`g++-10` (or `g++-11`). ② `kSTRONGLY_TYPED is not a member` → TensorRT
> < 8.6; the converter falls back to `kEXPLICIT_BATCH`, but the **runtime needs
> TensorRT ≥ 8.5** (`enqueueV3`) — TRT 8.4 can't work, reflash to JetPack 6.
> ③ At the **final link**, `undefined reference to
> ...@GLIBCXX_3.4.29/30/32` → conda's prebuilt libs (protobuf/abseil/yaml-cpp)
> were built against a newer libstdc++ than gcc-10 ships; link against conda's
> own `libstdc++.so.6` (the scripts do this via a shim `-L` dir; manually:
> `-L<dir-with-libstdc++.so-symlink-to-$CONDA_PREFIX/lib/libstdc++.so.6>
> -Wl,-rpath,$CONDA_PREFIX/lib`). If you're fighting all three, reflashing to
> JetPack 6 is usually less total effort.

The manual equivalent of each step is below.

**1. Build the tracker on the NX** (the converter is a tracker tool — step 2):
```bash
# Real-robot tracker on the NX, TensorRT engine. (export TENSORRT_ROOT=/path
# only if TRT isn't under the standard JetPack location.)
cmake -S system/cpp -B system/build/g1_nx -G Ninja \
  -DRT_DEPLOY_WITH_UNITREE=ON -DRT_DEPLOY_WITH_TRT=ON
cmake --build system/build/g1_nx --target motiontrack_real_zmq
# -> system/build/g1_nx/motiontrack_real_zmq
```

**2. Build `onnx_to_trt` + convert ONNX → TensorRT `.plan`.** `onnx_to_trt` is a
tracker target (`deploy_cpp/CMakeLists.txt`); it needs only
`RT_DEPLOY_WITH_TRT=ON` (no MuJoCo / Unitree), so build it from halo_tracker's own
CMake:
```bash
cmake -S tracker/deploy_cpp \
  -B tracker/deploy_cpp/build/g1_nx -G Ninja \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DRT_DEPLOY_BUILD_TESTS=OFF \
  -DRT_DEPLOY_WITH_TRT=ON
cmake --build tracker/deploy_cpp/build/g1_nx --target onnx_to_trt
# -> tracker/deploy_cpp/build/g1_nx/onnx_to_trt

tracker/deploy_cpp/build/g1_nx/onnx_to_trt \
  --onnx  tracker/deploy/models/g1/motiontrack_all_data.onnx \
  --plan  motiontrack_all_data.plan \
  --fp16            # optional; see note below
# other flags: --weakly-typed  --workspace <GB>  --device <N>
```
> ⚠️ **A `.plan` is NOT portable.** TensorRT serializes for the *specific* GPU
> (SM/compute capability) and TRT version. **Run `onnx_to_trt` on the NX itself**
> (or an identical unit) — a plan built on a dev box / different Jetson will fail
> to deserialize (`deserializeCudaEngine failed (check TRT version vs plan)`).
>
> Networks are built **strongly-typed** by default (TRT 10 best practice), so
> precision follows the ONNX model's own typing and `--fp16` is a no-op there;
> use `--weakly-typed --fp16` if you need the legacy FP16 precision-control path.

**3. Run the tracker with the `.plan`:**
```bash
system/build/g1_nx/motiontrack_real_zmq \
  --config tracker/deploy_cpp/configs/g1_v2.yaml \
  --policy motiontrack_all_data.plan --net eth0
```
The engine backend auto-selects TensorRT when `--policy` ends in `.plan` (or pass
`--engine trt` explicitly); a `.onnx` policy uses ONNX Runtime.

## Run

```bash
# MuJoCo tracker — binds 5555–5558:
system/build/mac/motiontrack_mujoco_zmq \
  --config tracker/deploy_cpp/configs/g1_v2.yaml \
  --policy <policy.onnx> --scene_xml <scene.xml>
```

The nodes run the proprio student (`g1_v2.yaml`: ONNX inputs `obs`, `policy_temporal`) or the depth student
(`g1_v2_depth.yaml`, + `student_depth`). The MuJoCo node renders depth with --depth_render (run_sim passes it when
the robot config has an image group) and publishes it on g1/depth (5553, DepthImage) for the viewer. The real node
reads the D435i with `--depth_config tracker/deploy_cpp/configs/depth_d435i.yaml` (build with
`-DRT_DEPLOY_WITH_REALSENSE=ON`; same flags and start-up checks as `deploy_g1_real`); a depth config without it is
refused at start.

Then drive it with the generator (`motiongen_zmq.py`). The manual two-terminal
run (MuJoCo + real robot, tracker + generator flag tables) is in
[`../README.md`](../README.md#run-real).

## See also

- **Python nodes** (generator, orchestrator): [`../README.md`](../README.md)
- **Transport middleware:** [`../comm/README.md`](../comm/README.md)
- **Motion-system doc** (architecture, build/run/test): [`../README.md`](../README.md)
- **Real-robot C++ bring-up** (RT scheduling, `setcap`, Unitree SDK): [`../../tracker/deploy_cpp/README.md`](../../tracker/deploy_cpp/README.md#run-on-the-robot) (also the CLI flags); RT tuning: [`../../tracker/deploy_cpp/docs/reference.md`](../../tracker/deploy_cpp/docs/reference.md)
