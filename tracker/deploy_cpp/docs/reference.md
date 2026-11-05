# Reference

Lookup material not needed for a first run. Build, run and the CLI flags are in [the README](../README.md).

[← deploy_cpp](../README.md) · [Depth / vision](depth.md)

**Contents**
- [Code layout](#code-layout) · [CMake target shape](#cmake-target-shape)
- [Jetson platform auto-detection](#jetson-platform-auto-detection): presets, Orin and Thor
- [Tuning on the target](#tuning-on-the-target): PREEMPT_RT, `isolcpus`, `setcap`
- [Loop timing traces](#loop-timing-traces) (`--trace_csv`) · [Depth / action tensor dump](#depth--action-tensor-dump) (`--depth_stats_every`)
- [Action decoding and sim plant drive](#action-decoding-and-sim-plant-drive)
- [Plan-switch contract](#plan-switch-contract): what the runtime plan server relies on
- [Troubleshooting](#troubleshooting): symptom → cause → fix

---

## Code layout

Robot-agnostic modules plus the G1 subtree. Headers live in `include/rt_deploy/<module>/`, implementations in
`src/` (mirrors `include/`).

| Path | Contents |
|---|---|
| `core/` | clock, rt_thread, spsc_ring, triple_buffer, jitter_histogram, rate_probe, loop_trace, expected |
| `math/` | quat, types (`RobotState<N>` / `MotorCommand<N>` / `ControlState` / `key_map`) |
| `inference/` | i_inference_engine, trt_engine, ort_engine, cuda_compat (engine-agnostic) |
| `policy/` | joint_remap, motion_loader (G1 CSV), motion_processing (resample / derive / yaw-align / transition), raw_motion_plan (runtime-pushed plan buffer), yaml_helpers (shared YAML loader bits) |
| `robot/` | i_robot_backend, remote_controller, safety_monitor, mujoco_backend (templated on N) |
| `vision/` | D435i depth sources (`RT_DEPLOY_WITH_REALSENSE`, [depth.md](depth.md)): realsense_stereo (IR pair capture), stereo_depth_engine (LiteAnyStereo on TRT), depth_pipeline (30 Hz cleaned producer), raw_depth_source (on-chip Z16 fallback), depth_source (factory + IDepthSource), depth_ops (shared depth math), stereo_kernels (CUDA launchers), stereo_frame, camera_intrinsics, config |
| `g1/` | config, action_decoder, observation_builder, motion_switcher, state_machine, control_loop, send_ref_motion, active_plan_snapshot, executed_ref_ring, depth_frame, depth_clip_registry, unitree_backend |
| `apps/g1/` | entry points deploy_real, deploy_mujoco, state_monitor; common.hpp wiring helper, depth_setup.hpp |
| `configs/` | g1_v2 (proprio student), g1_v2_depth (depth student), `*_sim` variants (finer MuJoCo step), depth_d435i.yaml (depth-source settings) |
| `cmake/` | Find{TensorRT,ONNXRuntime,UnitreeSDK2,RealSense2}.cmake |
| `tools/` | onnx_to_trt (plan builder), build_onnx_to_trt.sh, export_las_onnx.py (LiteAnyStereo ONNX for the stereo depth plan), vendor_third_party.sh, decompose_layernorm.py |

Header imports use `#include "rt_deploy/<module>/<file>.hpp"`. No `using namespace` at file scope anywhere. Class
names inside `rt_deploy::g1` drop the robot prefix (e.g. `g1::ObservationBuilder`, `g1::ActionDecoder`), because the
namespace already records the robot.

## CMake target shape

| Target | Kind | Notes |
|---|---|---|
| `rt_deploy::core` / `::policy` / `::inference` / `::robot` | static, always built | `::inference` links the enabled engines (TRT, ORT); `::robot` adds the MuJoCo backend with `RT_DEPLOY_WITH_MUJOCO=ON` |
| `rt_deploy::g1` | static | G1 modules (`config`, `control_loop`, `observation_builder`, `motion_switcher`, `send_ref_motion`, `depth_clip_registry`); the `system/cpp` tracker nodes link it |
| `rt_deploy_g1_unitree` | static | Unitree DDS backend on top of `rt_deploy::g1`; only with `RT_DEPLOY_WITH_UNITREE=ON` |
| `rt_deploy::vision` | static | D435i depth sources; only with `RT_DEPLOY_WITH_REALSENSE=ON`; consumed by `deploy_g1_real`. The cleaned (TensorRT) producer needs `RT_DEPLOY_WITH_TRT=ON` too; RealSense alone is a raw-only build with no CUDA toolkit |
| `rt_deploy::vision_cuda` | static | The single `.cu` unit, isolated: links **only** `CUDA::cudart`, neither `rt_deploy_warnings` nor `rt_deploy_rt_flags` (nvcc rejects the host warning flags and the `-march`/`-ffast-math` RT flags) |
| `rt_deploy_nan_safe` | interface | `-fno-finite-math-only` under `Release-RT`, so the NaN guards in the depth path (and `std::isfinite` in the observation builder) survive `-ffast-math`. Link it **after** `rt_deploy_rt_flags`: CMake emits a target's own options before inherited ones, so a `PRIVATE` flag would land left of `-ffast-math` and be silently re-enabled |
| `deploy_g1_mujoco`, `deploy_g1_real`, `g1_state_monitor`, `onnx_to_trt` | binaries | `onnx_to_trt` is the TRT plan compiler |

The templated MuJoCo backend (`robot::make_mujoco_backend<N>`) is explicitly instantiated for N=29 (G1) in
`src/robot/mujoco_backend.cpp`, with matching `extern template` declarations in the header.

## Jetson platform auto-detection

`cmake --preset release-rt` (without the `-orin` / `-thor` suffix) reads `/proc/device-tree/model` at configure time
and picks Orin vs Thor flags automatically. Useful for CI runners that build natively on a Jetson without knowing
which model they're on. Override explicitly with `-DRT_DEPLOY_JETSON_PLATFORM=orin|thor|none`.

| Jetson | Preset | TRT plan suffix |
|---|---|---|
| Orin (**the G1**) | `release-rt-orin` | `*_orin.plan` |
| Thor | `release-rt-thor` | its own plan (Orin plans do not load) |
| Auto | `release-rt` | matches the detected platform |

### Jetson Thor (JetPack 7.x, CUDA 13, TensorRT 10)

Thor is a supported build target, but no G1 ships on one. Same packages as Orin; the key difference is the CPU
microarchitecture (Neoverse-V3AE / ARMv9.2-A vs Orin's Cortex-A78AE / ARMv8.2-A). CUDA 13 renames a handful of
runtime symbols; the build is clean under `-Werror` because all CUDA touchpoints go through
[`include/rt_deploy/inference/cuda_compat.hpp`](../include/rt_deploy/inference/cuda_compat.hpp).

No source changes are required between Orin and Thor builds, only a different CMake preset and a per-target TensorRT
plan (plans are never portable across `{GPU SM, CUDA version, TRT version}`).

## Tuning on the target

1. **Kernel**: `uname -v` should include `PREEMPT_RT`. If not, apply the RT patches and rebuild.
2. **CPU isolation**: add `isolcpus=1 nohz_full=1 rcu_nocbs=1` to the kernel cmdline. Leave DDS callbacks on CPU 0
   (default) and pin the control thread, which also runs the policy inference, to CPU 1 (`--rt_cpu 1`). Depth
   threads go on a non-isolated core (see `configs/depth_d435i.yaml`).
3. **RT capabilities**: `cap_sys_nice` enables `SCHED_FIFO` (`--rt_prio`) and `cap_ipc_lock` enables `mlockall`
   (`--lock_memory`). Needs `libcap2-bin` for `setcap`/`getcap`, and must be **re-run after every rebuild**: caps
   live on the inode, so re-linking the binary drops them. Run as a non-root user afterward; no sudo needed.

   ```bash
   sudo setcap 'cap_sys_nice=eip cap_ipc_lock=eip' ./deploy_g1_real
   getcap ./deploy_g1_real   # verify → cap_ipc_lock,cap_sys_nice=eip
   ```

   > **Gotcha:** a capability-bearing binary is a secure-execution program, so the dynamic loader **ignores
   > `LD_LIBRARY_PATH`**. If the binary relies on it to find CUDA / TensorRT / DDS shared libs (common on JetPack 5,
   > where CUDA lives under `/usr/local/cuda-11.4/.../lib`), it runs fine *before* `setcap` and then fails with
   > `error while loading shared libraries` after. Register those dirs with the loader instead
   > (`echo <libdir> | sudo tee /etc/ld.so.conf.d/local.conf && sudo ldconfig`) or bake them into the binary's RUNPATH.

4. **RT throttling off**: `echo -1 | sudo tee /proc/sys/kernel/sched_rt_runtime_us`.
5. **Jetson perf mode**: `sudo nvpmodel -m 0 && sudo jetson_clocks` for max sustained performance.
6. **CUDA MPS** (optional): `nvidia-cuda-mps-control -d` before launch if multiple GPU processes coexist.

## Loop timing traces

`--profile` reports aggregate percentiles; `--trace_csv PATH` keeps **every** iteration, to plot the loop over a whole
run or overlay the timing on a third-person video of the robot.

```bash
./deploy_g1_real --config configs/g1_v2.yaml --policy policy.plan \
    --motion <motion.csv> \
    --rt_prio 80 --rt_cpu 1 --lock_memory --trace_csv runs/loop_cpp.csv
```

| Column | Meaning |
|--------|---------|
| `iter` | Control-tick index (never reset) |
| `t_rel_s` | Seconds since the start of the trace (CLOCK_MONOTONIC) |
| `t_wall_s` | Unix epoch seconds; **use this to sync with video** |
| `period_ms` | Interval between consecutive tick wake-ups (the achieved rate) |
| `loop_ms` | Work done in the tick, **excluding** the wait for the next period |
| `obs_ms` | `read_state` + `poll_remote` + observation build |
| `infer_ms` | Policy inference (0 outside CONTROL) |
| `act_ms` | Action decode + `send_command` + `backend.step()` |
| `jitter_ms` | `period_ms` minus the nominal control period |
| `state` | Control-state name (`INIT`, `MOVING`, `DEFAULT`, `CONTROL`, ...) |
| `depth_seq` | Producer `seq` of the depth frame this tick used. `-1` = no depth inbox attached, or none published yet (a proprio-only run is `-1` on every row) |
| `depth_stale` | Control ticks since `depth_seq` last advanced. `0` = a new frame arrived this tick. At 50 Hz control against 30 Hz depth the healthy value alternates 0/1 |
| `depth_age_ms` | This tick's wake-up minus the frame's publish stamp: how old the image the policy saw was. `-1` = unknown (no frame yet, or a producer that does not stamp; the MuJoCo `--depth_render` renderer stamps its render time, so sim runs report a real age). Read `depth_seq` first: a frame published just *after* this tick's wake-up can give a small negative age |

- Pair the depth columns with `period_ms` on the same row: if a stalled producer ever cost the loop a deadline, the
  two would move together. Compare traces with and without `--depth_config` as in
  [depth.md](depth.md#control-loop-jitter-with-vs-without-depth).
- `jitter_ms` is the *period* error, not the ticker's wake-vs-deadline phase error, so it compares with a loop that has
  no scheduled deadline (e.g. a Python tracker). The RT phase error is in the `[rt_deploy] ... jitter:` line at exit
  and under `--profile`.
- The header line reads `# rt_deploy loop trace v2` (13 columns); read columns **by name**. `#` lines carry the run's
  metadata and the `t0_wall_ns` / `t0_mono_ns` clock pair, so `pandas.read_csv(path, comment="#")` gives a clean
  frame. `# depth_inbox=0|1` records whether a depth producer was attached; `# depth_stats_every=N` whether the tensor
  dump below ran (such a trace is **not** a clean jitter baseline). A `# samples_dropped=` trailer reports samples the
  writer thread was too starved to take.
- Recording never blocks the control thread: `LoopTrace::record()` (`include/rt_deploy/core/loop_trace.hpp`) does one
  lock-free SPSC push of a POD (no allocation, lock, syscall or I/O); a writer thread created on the **non-RT** main
  thread does all formatting and file I/O. Per-tick cost is ~8 `clock_gettime` calls (~200 ns on Orin) plus the push,
  and one branch per phase with `--trace_csv` unset. The depth columns add no clock read.

## Depth / action tensor dump

`--depth_stats_every N` checks that the depth policy is fed depth in the units and encoding it was trained on, and
returns sane actions. Every N **inference** ticks it prints the built `student_depth` group tensor as the engine
received it (post-encode), the raw metric `g1::DepthFrame` behind it, and the returned action vector. `0` (the
default) is off, and off costs one integer compare per tick.

```bash
# once per second at 50 Hz, with stderr kept off the terminal
./deploy_g1_real --config configs/g1_v2_depth.yaml --policy depth_student.plan \
    --motion <motion.csv> \
    --net eth0 --depth_config configs/depth_d435i.yaml \
    --depth_stats_every 50  2> runs/depth_stats.log
```

```
[control_loop] depth_stats @ infer_tick=250  engine=kOk
  student_depth tensor  rows=1 cols=4032 n=4032   (image 48x84)
    values             min=0.262364 max=1.386294 mean=0.871420 nonfinite=0
    expected bounds    near=log1p(0.300)=0.262364  far=log1p(3.000)=1.386294
    at bounds          at_near=2 at_far=3  far_fraction=0.07%
  raw DepthFrame (m)   seq=42 stale=0 ticks  age=1.46 ms
    values             min=0.0000 max=10.0000 mean=1.5010 nonfinite=1 no_return=1
  action (train order) n=29 min=-0.500000 max=0.250000 nonfinite=0  clip=+/-100.000
```

| Field | Read it as |
|---|---|
| `rows`/`cols` vs `(image HxW)` | Engine-facing shape next to the image geometry. `(1, 4032)` against `48x84` is correct; anything else is a config / export mismatch. |
| expected vs observed bounds | `near=`/`far=` is what the encode *should* saturate to, next to the observed `min`/`max`, so a clip range that no longer matches training is visible without arithmetic. |
| `far_fraction` | `100.00%` = every pixel at the far plane: the depth source is dead or blank and the policy is told the world is empty. Tagged `[!! 100% FAR BOUND]`. |
| raw metres vs encoded | `raw max=10.0000` with `encoded max=1.386294` proves the encode ran exactly once and in the right direction. A raw max near 3000 would be millimetres; near 1.386 (log1p of the far clip) an already-encoded frame encoded twice. |
| `nonfinite` | Must be 0 on the tensor row (the encode maps every invalid sample to the far plane). On the raw row it counts what the camera delivered. |
| action row | Sampled before the action clip, so a `±1e30` output shows as `±1e30`, not as a plausible `±clip_action`; tagged `[!! NON-FINITE ACTION]` on a NaN. |
| `age` | Measured against the same tick wake-up as the trace's `depth_age_ms`, so a log line and its CSV row agree exactly. |

**It writes to stderr from the RT control thread.** stderr is unbuffered, so on a pipe with a slow reader that write
blocks the control loop. Keep `N` at a second or more of ticks on a powered run, and redirect stderr to a file. The
scan itself is a few thousand float loads, allocation-free.

## Action decoding and sim plant drive

**Torque-aware target clip.** The motor applies `tau = kp*(target - q) - kd*v`, so the action decoder clips each
target to the window `q + [(-max + kd*v)/kp, (max + kd*v)/kp]` (`max` = the config's `max_torque`). The Python tracker
(`BaseController.compute_motor_targets` in `tracker/deploy/utils/core/base_controller.py`) uses the same window.

**Sim plant drive (MuJoCo only).** The configs' optional `sim_kp_scale` / `sim_kd_scale` (29 positive values,
deployment order) are the lab robot's identified drive: the MuJoCo backend realises `joint_kps * sim_kp_scale` and
`joint_kds * sim_kd_scale`, the stiffness and damping training simulates, while the command (and the clip window
above) keeps the commanded gains, as on the robot. The real backend ignores both keys. The Python MuJoCo backend
applies the same scales.

## Plan-switch contract

What the runtime plan server (the `system/cpp` tracker nodes, [`system/README.md`](../../../system/README.md)) relies
on:

- **Reference index.** Tick `k` after a counter reset reads window frame `k` (training reads frame `k` at decision
  `k`). `get_time_indices` floors with a small relative tolerance, because `control_dt` is the float `0.02F` and
  plain truncation would read frame `k-1` on almost every tick.
- **`ActivePlanSnapshot`.** Row 0 is the live window's row 0 and `install_tick` is always 0, so the server's
  `idx = tick - install_tick` is the window frame being played. `master_offset` is the exact (fractional) master row
  of row 0.
- **Trim commit** (`trim_install_pending`, `policy::commit_trim_cut`). The cut is the tick the server read. The loop
  advances the window and `master_offset` by the cut, records the trim phase, and keeps the ticks played since the
  read (`inference_tick -= cut`), so no frame is replayed and the next RTC install's `start_offset` is measured from
  the trim. A cut of 0 (trim read right after an install) keeps the counter and still records the phase. A cut larger
  than the current tick was read before a plan install reset the tick; it is dropped (stale trim). A cut clamped to
  `frame_count` (playback ran past the window) keeps only the held final frame and restarts the tick there.
- **Exact switch timing.** An install keeps the raw plan from the whole input row before its exact playback point
  `phase + start_offset * input_fps / output_fps` and records the remainder as `master_offset`. A trim returns whole
  stored master rows (never re-resampled) from the one NEAREST the exact playback point (`policy::master_trim_start`);
  the gap (`trim_phase`, within half an input frame) is carried to the next auto-start-offset install at the same
  input fps, which shifts its 50 Hz upsample grid by it (`resample_motion(..., start_input_row)`). The new plan plays
  exactly where the old one would have, with no per-install quantization.
- **Install acknowledgement.** `ControlLoop::plans_consumed()` counts plans taken from the inbox (installed or
  rejected: arrived too late, non-finite values, bad header) and `plans_installed()` the ones that became the active
  reference. A server that publishes a plan and waits (bounded) until `plans_consumed()` moves can reply with the real
  outcome instead of acking on receipt; the deployed system did this with a 60 ms wait. Plans are consumed only in
  CONTROL, so a plan sent before CONTROL stays queued.
- **Ordering.** Plan install and trim commit run after the safety check, `send_command` and the sim `step()`. This
  tick's command was decoded from the old plan either way; the counter only moves at the next inference, so the
  timing above is unchanged. A plan or trim raised before tick N's install/commit (e.g. during its send) is played by
  tick N+1; one that arrives after it is applied at the end of tick N+1 and plays from tick N+2.
- **Known limits (need plan-server changes).** The server can pair a snapshot with a tick from the other side of an
  install (a stale prefix the loop cannot detect). A dropped stale trim still leaves the next prefix from the replaced
  plan. Closing both fully needs a plan sequence number in the trim handshake.

## Troubleshooting

| Symptom | Likely cause | Where to look |
|---------|--------------|---------------|
| cmake: `The source directory "…/<preset>" does not exist` | `cmake` too old for `--preset` (< 3.19; focal's stock is 3.16) | `cmake --version`; upgrade: `pip install --user "cmake>=3.22"` ([JetPack 5.x](../README.md#jetpack-5x-l4t-r35-cuda-114-tensorrt-85-gcc-10-the-stock-g1)) |
| cmake: `Could not read presets … Unrecognized "version" field` | `cmake` older than the presets schema (`version 2` → ≥ 3.20) | Same upgrade; the project floor is `cmake_minimum_required(3.22)` |
| apt: `Unable to locate package g++-11` | JetPack 5.x / focal has no g++-11 | Install `g++-10`, configure with `CC=gcc-10 CXX=g++-10` |
| cmake: `Compatibility with CMake < 3.5 has been removed` (at `_deps/yaml-cpp-src/CMakeLists.txt`) | conda's `cmake` 4.x; yaml-cpp 0.8.0 declares `cmake_minimum_required(3.4)` | `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` (the presets set it) or `conda install "cmake<4"` |
| link: `undefined reference to std::__throw_bad_array_new_length()@GLIBCXX_3.4.29` | conda's `libyaml-cpp.so` (GCC 11+) linked with the CUDA-mandated g++-10 | `-DCMAKE_DISABLE_FIND_PACKAGE_yaml-cpp=ON` (builds yaml-cpp from source) |
| `unitree/dds_wrapper/common/crc.h: No such file` | An incomplete Unitree SDK under `/usr/local/include` is shadowing the full one | `export UNITREE_SDK2_ROOT=/opt/unitree_robotics`, then clear the cache and reconfigure: `cmake -U "UnitreeSDK2_*" -U unitree_sdk2_DIR <build>` ([Unitree SDK2](../README.md#unitree-sdk2)) |
| setcap'd binary: `error while loading shared libraries` | File caps put the loader in secure mode → `LD_LIBRARY_PATH` is ignored | `ldconfig` the CUDA / DDS lib dirs ([tuning](#tuning-on-the-target)), or rebuild with them on the binary's RUNPATH |
| `ControlLoopExit::kStateTimeout` | DDS isn't producing `rt/lowstate`; wrong `--net`; robot powered off | `ip link`, `cyclonedds_ping`, Unitree connection docs; `g1_state_monitor` ([bring-up](../README.md#1-bring-up-check-the-dds-receive-path)) |
| `FATAL: obs group '…' matches no engine input` / `engine input … is fed by no observation group` | The config's `observation.groups` names do not match the exported policy's input names | Rename the groups to the ONNX input names, or use the config that matches the export |
| `inference FAILED (kSizeMismatch)`, then `inference failed 25 ticks in a row; stopping` | A group's tensor size differs from the engine's input shape (wrong `history`, `horizon` or term list for this export) | Compare the policy's input shapes with the config's `observation:` block |
| `engine IO tensors are not FP32` (TRT) / `ORT input/output is not float32` | Policy was exported/built with FP16 (half) or other-dtype IO; the host/device buffer + copy path is FP32-only | Re-export the ONNX with FP32 IO, or rebuild the `.plan` so its bound IO tensors are `kFLOAT` (weights may still be FP16) |
| `MuJoCo model is not a floating-base model …` | `xml_path` in the config points at a fixed-base or non-standard MJCF (no free joint, wrong qpos/qvel layout) | Use a free-joint model with ≥ `7+N` qpos and `6+N` qvel (the stock robot MJCFs satisfy this) |
| `fault=stale_state` latch | No fresh DDS state for > 60 ms (3 × the 20 ms control period), usually kernel preemption of the I/O thread | `/proc/sys/kernel/sched_rt_runtime_us`, PREEMPT_RT patch |
| `fault=state_non_finite` | IMU returned NaN: hardware / calibration, not software | Robot-side diagnostics; do not fix in software |
| `fault=joint_limit` | Policy output pushed a joint past a configured hard limit. Off by default: both apps leave `SafetyConfig::joint_min/max` at ±∞ (matching the Python reference, which has no joint-limit latch); set them in the app glue to arm it | If you armed it, recheck `joint_min/max`; inspect the last pre-latch command |
| High `jitter_ms` in `--trace_csv` | No CPU isolation or SCHED_FIFO | `cat /proc/cmdline`, `getcap deploy_g1_real` |
| G1 policy action ≈ 0 | ONNX/TRT inputs do not match the config's `observation:` block | Compare the policy's input names/shapes with the config (the loop refuses a group/input mismatch at start) |
| G1 motion loads but standup looks wrong | xyzw/wxyz confusion or joint-order mismatch | CSV root quaternion must be xyzw (converted to wxyz at load); check the config's `deployment_joint_order` / `training_joint_order` |
| Damping-on-exit not engaging | Process killed before `cleanup()` | Send SIGTERM or Ctrl-C, not SIGKILL |
