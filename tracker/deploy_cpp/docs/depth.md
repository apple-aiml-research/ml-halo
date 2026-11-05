# Depth / vision: clean depth for the G1 depth student

Replaces the D435i's noisy on-chip depth with a learned stereo estimate, computed on the robot at 30 Hz and fed to the
motion-tracker policy's `student_depth` observation, **without perturbing the 50 Hz control loop**. The D435i's block
matcher produces speckle, holes and edge-fattening that are hard to reproduce in simulation, so a policy trained on
clean rendered depth meets a very different input distribution on hardware. Running LiteAnyStereo over the raw IR
pair (projector off) closes most of that gap.

[← deploy_cpp](../README.md) · [Reference](reference.md)

Code: `include/rt_deploy/vision/`, `src/vision/` (the producers) and `apps/g1/depth_setup.hpp`; built only with
`-DRT_DEPLOY_WITH_REALSENSE=ON`, off by default. The ObservationBuilder `image` group (clip → log1p → `(1,1,48,84)`),
the depth-clip check and the control loop's depth inbox (staleness, far-plane fallback, depth stats) are in `g1/`.

**You must provide** (no weights ship): a depth-student export, a 3-input FP32 ONNX (`obs` (1, 434),
`policy_temporal` (1, 30, 166), `student_depth` (1, 1, 48, 84)), and a LiteAnyStereo ONNX export
([Build the engine](#build-the-engine-on-the-orin-that-will-run-it)); no LiteAnyStereo exporter ships here.
`ControlLoop`'s constructor refuses a name or shape mismatch rather than silently zeroing an input.

A config with an image group (e.g. `g1_v2_depth.yaml`) is refused at start (`ControlLoopExit::kConfigError`) when no
depth inbox is attached: in `deploy_g1_mujoco` without `--depth_render`, and in `deploy_g1_real` without
`--depth_config` unless `--no_send` (dry run: nothing actuates, so the loop warns and feeds the far plane; the
no-depth timing baseline).

---

## Architecture

```
[rs-capture]  D435i IR1/IR2, Y8 424x240@30, emitter OFF, live fx + baseline
      │        blocks in wait_for_frames(); two memcpy per frame, no allocation
      │  TripleBuffer<StereoFrame>          wait-free, latest-wins, drops stale pairs
      ▼
[depth-infer] PeriodicTicker @ 30 Hz, priority BELOW the control loop
      │        ir_to_chw (CUDA) → LiteAnyStereo (TensorRT, FP16) → fx*B/disp
      │        → occlusion mask → clamp → reduce to 84x48 metres (CUDA)
      │  TripleBuffer<g1::DepthFrame>       wait-free, latest-wins
      ▼
[g1 control loop @ 50 Hz, SCHED_FIFO, main thread]
        tick step 1c: try_acquire_latest()  ← the ONE added instruction on the hot path
        image group: invalid→2.5, clamp[0.2,2.5], log1p → student_depth (1,1,48,84)
        infer_multi({obs, policy_temporal, student_depth})
```

Both hand-offs are latest-wins (a newer frame overwrites an unread one), with no mutex. End-to-end latency is one
capture plus one inference no matter how the stages drift.

**The control loop never blocks on depth.** No shared mutex, no condition variable, no allocation on any produce path
after `start()`. A stale, missing or failed frame degrades depth freshness only, and past `--depth_stale_far_ticks`
the loop stops trusting it and feeds the far plane instead of a frozen image
([staleness](#depth-staleness-during-a-live-run)).

| `depth_source` | Path | Needs | Use |
|---|---|---|---|
| `cleaned` | IR pair → LiteAnyStereo → metric depth | GPU, CUDA, TensorRT, a `.plan` | The sim2real path |
| `raw` | D435i on-chip Z16 → metres | librealsense only | Fallback + A/B reference |

Both producers publish the identical `g1::DepthFrame` contract through the identical sink, so the control loop cannot
tell them apart. Switch with `--depth_source`, no rebuild.

---

## Build

```bash
cmake --preset release-rt-orin \
      -DRT_DEPLOY_WITH_UNITREE=ON -DRT_DEPLOY_WITH_TRT=ON \
      -DRT_DEPLOY_WITH_REALSENSE=ON -DRT_DEPLOY_CUDA_ARCH=87
cmake --build build/release-rt-orin -j
```

On the stock G1 image, add the compiler and dependency workarounds from the
[JetPack 5.x build](../README.md#jetpack-5x-l4t-r35-cuda-114-tensorrt-85-gcc-10-the-stock-g1)
(`CC=gcc-10 CXX=g++-10`, …).

- Needs `librealsense2-dev`, and `libnvinfer-plugin-dev` for the `cleaned` path. On the G1, the robot setup in
  [system/README.md, "Run real"](../../../system/README.md#run-real) installs both (it builds librealsense from source
  where apt has no arm64 package). LiteAnyStereo's
  `InstanceNormalization` is lowered to a TRT plugin that is required at **deserialize** time. A missing plugin
  library is a confusing runtime failure, not a build error.
- `RT_DEPLOY_WITH_REALSENSE=ON` alone gives a **raw-only** build with no CUDA dependency, useful for bring-up before
  the engine exists. `enable_language(CUDA)` is reached only when TRT is also on, so the default developer build never
  probes nvcc.
- **TensorRT 8.5 (JetPack 5: CUDA 11.4, TRT 8.5.2, gcc-10, the stock G1) and 10 (JetPack 6) are both supported.**
  On TRT 8.5 `onnx_to_trt` falls back to a weakly-typed network with `BuilderFlag::kFP16` (`kSTRONGLY_TYPED` starts
  at TRT 8.6, so `--weakly-typed` is a no-op there) and adds `kEXPLICIT_BATCH` (removed in TRT 10).
  `stereo_depth_engine.cpp` uses the name-based IO tensor API (`getNbIOTensors` / `setInputTensorAddress` /
  `enqueueV3`, from TRT 8.5). CUDA-graph capture uses the 5-argument `cudaGraphInstantiate` under CUDA 11 and the
  3-argument form from CUDA 12.

## Build the engine (on the Orin that will run it)

Export the LiteAnyStereo ONNX with `tools/export_las_onnx.py`, which calls the upstream repository's `export_onnx.py`
(clone github.com/TomTomTommi/LiteAnyStereo and put the checkpoints from its README's "Checkpoints" section under
`checkpoints/`; the Hugging Face page has only the LAS2 weights, so take the LAS1 `LiteAnyStereo.pth` from the
README's Google Drive mirror). Its defaults reproduce the deployed plan: LAS1 with `checkpoints/LiteAnyStereo.pth`,
max disparity 192, opset 17, 424×240 rounded up to 448×256. A different checkpoint or max disparity changes the depth
values without any error, so keep them. Nothing in `rt_deploy` is specific to one release: LAS1 and LAS2-S/M/L/H share the IO contract, and `StereoDepthEngine` discovers
the plan's IO names, dtypes and padded dims at load and validates them against the depth YAML. The export has:

- input geometry: the capture size from `depth_d435i.yaml` (424×240) rounded **up to a multiple of 32** (448×256);
  the pad divisor is 32;
- **opset 17** or lower (TRT 8.5's ONNX parser stops at 17; upstream defaults to 18). `InstanceNormalization` is a
  plugin op there, which is why the stereo engine calls `initLibNvInferPlugins()` before `deserializeCudaEngine`;
- FP32 inputs `left` / `right` `(1,3,H,W)` holding raw 0–255 values (ImageNet normalisation is inside the graph), FP32
  output `disparity` `(1,1,H,W)` in full-resolution pixels. So `launch_ir_to_chw` (Y8 IR replicated to 3 channels,
  left at 0–255) and `disp_to_depth_clamped` (`depth = fx*B/disp`) apply unchanged.

Export on any machine with torch, then build the plan on the Orin:

```bash
python3 tools/export_las_onnx.py --las_repo <LiteAnyStereo checkout> \
    --restore_ckpt <LiteAnyStereo checkout>/checkpoints/LiteAnyStereo.pth --output las_448x256.onnx
./build/release-rt-orin/onnx_to_trt --onnx las_448x256.onnx --plan las_424x240_orin.plan \
        --weakly-typed --fp16
```

`--weakly-typed --fp16` is deliberate: the engine asserts **FP32 tensor IO** and lets TRT choose FP16 internally.
**A `.plan` is not portable** across GPU, CUDA version or TRT version.

### Model variants: LAS1 and LAS2-S/M/L/H

A model swap is **a different `plan_path`, not a code change**. LAS2 weights are not in the repo
([huggingface.co/tomtomtommi/LiteAnyStereoV2](https://huggingface.co/tomtomtommi/LiteAnyStereoV2)).

```bash
# build a plan from a LAS2-S export
./build/release-rt-orin/onnx_to_trt --onnx las2s_448x256.onnx --plan las2s_448x256_orin.plan \
        --weakly-typed --fp16

# select it at runtime, no rebuild and no edited config
./deploy_g1_real ... --depth_config configs/depth_d435i.yaml \
                     --depth_plan las2s_448x256_orin.plan
```

Run one model at a time on the Orin: two `DepthPipeline`s would mean two TensorRT contexts contending with the policy
for one iGPU, the most likely source of control-loop jitter. Compare models sequentially
([below](#comparing-two-depth-models-on-the-control-loop)).

## Run

```bash
# proprio-only policy: nothing changes, no camera is opened
./deploy_g1_real --config configs/g1_v2.yaml --policy policy.plan --motion <motion.csv> --net eth0

# depth student with cleaned depth
./deploy_g1_real --config configs/g1_v2_depth.yaml --policy depth_student.plan --net eth0 \
                 --motion <motion.csv> --depth_config configs/depth_d435i.yaml

# same, raw D435i depth (A/B reference; --depth_source raw also turns the IR emitter on)
./deploy_g1_real ... --depth_config configs/depth_d435i.yaml --depth_source raw
```

Depth producer failures are **fatal at startup**: a depth policy running on the blank far-plane fallback would be
driving blind, and refusing before the robot moves is far safer than discovering it mid-motion. After start, failures
degrade gracefully.

---

## Encode and downsampling

### The encode clip is a training contract

`student_depth` is `clip([near, far]) -> log1p`. Get the clip wrong and **every pixel is mis-scaled while nothing
reports a problem**: the tensor shape and the ONNX signature still match, the policy runs, the robot moves. So it is
enforced rather than defaulted:

- **`clip` is mandatory** on every image group. There is no default, in the C++ parser or the Python one. A config
  that omits it is refused at startup.
- **It is cross-checked against the training clip** (`DEPTH_CLIP = (0.2, 2.5)` in
  `tracker/training/.../g1/motion_tracking_scene_distill_cfg.py`) at startup by `src/g1/depth_clip_registry.cpp`,
  which mirrors the Python deployment's `deploy/g1/variant_support.py::EXPECTED_DEPTH_CLIP`. A mismatch refuses;
  `--allow_depth_clip_mismatch` is the escape hatch for a checkpoint trained with a different clip.
- **The depth source's `clip_min_m`/`clip_max_m` must equal the group's clip.** `deploy_g1_real` loads both files and
  refuses to start if they disagree; `deploy_g1_mujoco` takes the render clip straight from the spec.

Invalid / no-return pixels are encoded at the far clip, and they dominate a real D435i, so a far clip that differs
from training shifts a large fraction of the image by a constant in log1p space.

### Downsampling: holes must not drag the mean

Training **never downsampled**: it rendered 48×84 natively (one ray per output pixel), so its pixel dropout set
*whole* output cells to far. On hardware we reduce 424×240 → 84×48, so how invalid pixels are aggregated is a
sim2real decision with no exact training counterpart:

| `downsample` | Behaviour | Verdict |
| --- | --- | --- |
| **`area_valid`** | Mean over the pixels the sensor actually measured. Holes excluded; genuine far readings far-clamped and kept; a cell reads far only if its whole footprint is invalid. | **Default.** Matches the Python deployment. |
| `area` | Plain mean, so far-filled holes enter the average. **Biased farther by local hole density**, so a near wall reads farther than it is. | A/B reference only. Do not deploy. |
| `min` | Closest depth in the footprint. Immune to the far-fill bias and never hides an obstacle, but biases nearer than training. | Safety-first fallback. |
| `point` | Nearest-neighbour at the footprint centre. Closest to the training renderer, but keeps the speckle the means suppress. | |

Two load-bearing details, both mirrored from the Python `deploy/g1/depth_sources.py::sanitize_depth`:

- **Far-clamp valid samples *before* aggregating.** A saturated 65.54 m reading is real data, just out of range;
  letting it into a mean turns that 0.5 m wall into 1.14 m at the deployed ratio.
- **A saturated reading is not a hole.** It is clamped and counted, not dropped.

`area_valid` needs validity *before* invalid samples are mapped onto `clip_max` (after sanitizing, a hole and a
genuine far reading are the same number), so both producers fuse the sanitize into the reduction
(`reduce_block_raw_m` raw, `reduce_disp_block` cleaned); for the three original modes both are bit-identical to the
two-pass reference in `depth_ops.hpp`.

### Depth rate: `obs_decim` applies in sim, not on hardware

The depth student trains with `obs_decim=2` in its depth observation term: each depth frame is held for 2 control
steps, i.e. **25 Hz** against a 50 Hz control loop. The image group declares that as `obs_decim: 2`:

| | Depth source | `obs_decim` | Why |
| --- | --- | --- | --- |
| `deploy_g1_mujoco` | MuJoCo render, **synchronous** with the loop | **Applied**: the backend renders every 2nd `step()` | Without it the renderer emits a fresh frame every tick: twice the training rate, on the only exteroceptive input |
| `deploy_g1_real` | D435i, **asynchronous** at ~30 Hz | **Not applied** | The sensor rate already *is* the hold. The correct frame is the newest one; holding on top would only add latency |

Decimating the *render* (the inbox serves the last frame in between) keeps the mechanism inside the sim backend, so
the hardware path cannot inherit it. Both binaries print the resolved rate at startup; `deploy_g1_mujoco` warns when
`obs_decim: 1` leaves depth at 50 Hz (a wrong rate changes no shape, so it is invisible everywhere else).

---

## Validation

### Control-loop jitter, with vs. without depth

```bash
./deploy_g1_real --config configs/g1_v2_depth.yaml --policy depth_student.plan --motion <motion.csv> \
                 --net eth0 --no_send --start_in_control --trace_csv /tmp/depth_off.csv
./deploy_g1_real --config configs/g1_v2_depth.yaml --policy depth_student.plan --motion <motion.csv> \
                 --net eth0 --no_send --start_in_control --trace_csv /tmp/depth_on.csv \
                 --depth_config configs/depth_d435i.yaml
```

The depth-off run needs `--no_send` (image-group config), so pass it on both runs. The DDS publish never runs, so the
comparison measures the depth overhead, not powered-run absolute timing; `--start_in_control` runs inference without
the gamepad. Run both back to back for ≥200 s each with the same policy and RT flags, and compare `period_ms`,
`jitter_ms` and `infer_ms` ([loop timing traces](reference.md#loop-timing-traces)). Watch `infer_ms` as closely as
jitter: GPU contention shows up there, and CPU-side jitter will not reveal it.

### Comparing two depth models on the control loop

Same procedure, with the depth thread running on both sides and only `--depth_plan` changing:

```bash
./deploy_g1_real ... --trace_csv /tmp/las1.csv  --depth_config configs/depth_d435i.yaml \
                     --depth_plan las_424x240_orin.plan
./deploy_g1_real ... --trace_csv /tmp/las2s.csv --depth_config configs/depth_d435i.yaml \
                     --depth_plan las2s_448x256_orin.plan
```

The trace header records `depth_source` and `depth_plan`, so two traces of the same policy under two stereo models can
be told apart. The cleaned producer's 1 Hz `[depth_pipeline]` log line reports each model's inference time and
publish rate against the 33.3 ms (30 Hz) budget.

### Depth staleness during a live run

The control loop tracks `DepthFrame::seq` and reacts in **two stages**, because a slightly old frame and a frozen one
are different problems:

| Stall length | Behaviour | Knob |
| --- | --- | --- |
| Short (default < 10 ticks / 200 ms) | Keep the last-known-good frame, silently. The policy trained on 40–120 ms of staleness, so this is in-distribution; blanking on every dropped camera frame would be worse. | |
| Warn (≥ 10 ticks) | Still hold the frame, but say so once per stall. | `ControlLoopConfig::depth_stale_warn_ticks` (fixed at 10; no CLI flag) |
| Long (≥ 25 ticks / 500 ms) | **Abandon the frozen frame and feed the far plane.** | `--depth_stale_far_ticks` |

Past half a second a frozen frame is fiction: a policy steering around geometry that has gone fails with no other
symptom. Far is the honest encoding of "no information", and what training used for every dropped pixel. The
threshold matches the Python `RealSenseDepthSource(stale_s=0.5)`. Neither stage blocks, skips or delays a tick.
`--depth_stale_far_ticks 0` disables the fallback (hold forever) for a bench run that needs the frozen-frame
behaviour. Before the producer's first frame the loop already feeds the far plane (`last_depth_` begins null).

The end-of-run summary reports frames consumed, the longest stale run, and how many times the far-plane fallback
engaged; a nonzero count means the policy was depth-blind for part of the run. At 50 Hz control against 30 Hz depth
the healthy steady-state stale run is 1–2 ticks.

To inspect the tensor the policy actually receives, use `--depth_stats_every`
([depth / action tensor dump](reference.md#depth--action-tensor-dump)).

---

## Status: what is and is not measured

**Timing is target-specific**: measure every latency and jitter number on the Orin. A macOS dev host has no CUDA,
TensorRT or librealsense, `rt_thread` is an inert stub and `MonotonicClock` falls back to `steady_clock`, so **no
dev-host timing transfers to the target**. `stereo_kernels.cu` and `stereo_depth_engine.cpp` are not built on a dev
host; the CUDA kernel's fused reduction shares its arithmetic with the raw CPU path through `RT_DEPLOY_HD`
(`depth_ops.hpp`), and the device build is only exercised on the target.

**Chosen backend: TensorRT FP16 at 424×240.** TRT over Torch because the GPU is shared with the policy's own inference
and TRT's fixed-shape CUDA-graph submit has far lower and more predictable overhead. 424×240 over 640×480 because it
is much cheaper and its focal length, once reduced to 84×48, stays close to the training camera's 42.19 px.
LiteAnyStereo is single-shot (no refinement loop), so its latency is data-independent, a useful property against a
hard 33 ms budget.

## Parity with the Python deployment

The Python twin ([`deploy/utils/core/observation_builder.py`](../../deploy/utils/core/observation_builder.py),
[`deploy/g1/g1_controller.py`](../../deploy/g1/g1_controller.py)) implements the same `student_depth` contract: a
source delivers 48×84 metric depth, the builder applies invalid→far, `clip[0.2, 2.5]`, `log1p`. The encode chains are
identical, including the invalid threshold (`> 1e-6` on both sides), and the downsample rule matches (`area_valid`).
Python's depth sources (`deploy/g1/depth_sources.py`) are a MuJoCo renderer and the D435i's own Z16 depth read
in-process with `pyrealsense2` (the C++ `raw` path). The learned-stereo `cleaned` path exists only in C++.

Both sides refuse to start a depth policy with no depth source. One difference in the override is **deliberate**:

| | Python | C++ (here) | Why it's intentional |
|---|---|---|---|
| **Missing depth source** | refuses to start; `--allow_no_depth` overrides it and feeds a blank far-plane image (debugging only) | refuses to start (`ControlLoopExit::kConfigError`); no override, except under `--no_send` | A depth policy fed a blank far plane believes the world is empty. The C++ path has no flag that lets it actuate the robot on that input; the only blind run is the `--no_send` dry run. |

Structurally, Python's `get_depth()` is a **synchronous pull on the control thread**, so `RealSenseDepthSource`
(`tracker/deploy/g1/depth_sources.py`) captures on a background thread and returns the newest cached frame under a
lock (a direct `wait_for_frames()` would stall the loop by the camera's latency). In C++ the producer pushes into a
wait-free buffer that the control loop only reads.

## Open risks

1. **GPU contention with the policy's own TRT context.** The control loop calls `engine.infer_multi()` on the RT
   thread and ends in `cudaStreamSynchronize` *on that thread*. Stereo kernels in a second TRT context on the same
   iGPU time-slice SMs with it, so the sync can queue behind depth work and inflate policy-inference p99 inside the
   20 ms budget; CPU-side jitter does not show it. The most likely way this feature hurts the tracker. Mitigations:
   `cudaStreamCreateWithPriority` (policy high, depth low), a lower depth rate, INT8, or moving depth to DLA.
2. **Orin NX throughput must be measured.** A 7.6 M-parameter network with a 48-hypothesis correlation volume may or
   may not fit 33 ms while sharing the GPU. The cleaned producer's 1 Hz `[depth_pipeline]` line (inference time,
   publish rate) is a go/no-go check.
3. **The downsample rule is a judgement call** ([downsampling](#downsampling-holes-must-not-drag-the-mean)). A/B it
   offline against a MuJoCo `--depth_render` frame from the same pose before trusting any mode on hardware.
4. **~3° FOV mismatch.** Training used HFOV 89.74°/VFOV 59.26°; a real D435i is nominally 87°×58°, so a full-frame
   resize compresses the scene ~3% horizontally relative to training.
5. **Self-occlusion.** Training renders the robot's own body into the depth image (the body-mesh ray-caster in
   `motion_tracking_scene_distill_cfg.py`). Do **not** mask self-body pixels at deploy.
6. **Thermal and power.** Continuous GPU load from a second network on a shared Orin NX may trigger throttling that
   degrades *both* networks. Unmeasured.
7. **Extrinsics.** The nominal camera transform is the URDF `torso_link → d435i_link` chain (50.749° downtilt).
   Training randomised ±2.5 cm / ±2.5°; a camera remounted outside that band needs the URDF updated.
