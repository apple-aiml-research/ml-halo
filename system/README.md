# system

Closed-loop system: the generator plans motion from the camera and a command, the tracker follows it, and the
orchestrator picks the subtasks. Runs in MuJoCo (`run_sim.py`) or on the Unitree G1 (`run_real.py`).

## Setup

Run from the repository root, in the generator's conda env (it brings torch and the pinned transformers):

```bash
bash generator/setup.sh && conda activate halo_generator
bash system/cpp/scripts/setup_dev_env.sh   # apt (Linux) or Homebrew (macOS), repo packages, MuJoCo, Hugging Face models
bash system/cpp/scripts/build_motiontrack_mujoco_zmq.sh
```

The multi-task orchestrator's model runs on MLX, so `multi_task/` needs an Apple Silicon Mac with at least 32 GB of
RAM. Offline box: `SKIP_BREW=1 SKIP_HF_MODELS=1`, and copy `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B` and
`models--mlx-community--Qwen3.6-35B-A3B-4bit` from a box that has them.

## Models

No weights ship with the repository. Each task config needs:

| Key | What | From |
|---|---|---|
| `checkpoint` | generator weights (`best_model.pt`) | `generator/scripts/train.py` ([generator/README.md](../generator/README.md)) |
| `gen_config` | the YAML the checkpoint was trained with | the `--config` of that training run |
| `policy_path` | tracker ONNX matching `robot_config`: the depth student for `single_task/` (`g1_v2_depth_sim.yaml`), the proprio student for `multi_task/` (`g1_v2_sim.yaml`) | `tracker/training/scripts/rsl_rl/play.py --export_only` ([tracker/training/README.md](../tracker/training/README.md#play-and-export)) |

The generator's base model `Qwen/Qwen3.5-2B` must be in the Hugging Face cache, since `run_sim.py` runs offline;
`setup_dev_env.sh` downloads it (with `SKIP_HF_MODELS=1`:
`python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-2B')"`).

The `multi_task/` configs plan subtasks with the orchestrator, whose model runs on `halo_serve` with MLX (Apple silicon
only). The `single_task/` configs have the orchestrator off and also run on Linux with a CUDA GPU.

## Tasks

Tasks are in `system/halo_system/scripts/configs/` (`single_task/`: push_box, crouch_under_table, go_to_man,
sit_on_chair, sidestep_door; `multi_task/`: door_then_sit). Copy one and replace its `<...>` paths (see Models).
`datagen/` produces training data for `sit_on_chair` only; the other tasks need a generator trained on data that is
not part of this repository.

## Run sim

```bash
HALO_COMM_REGISTRY=127.0.0.1:5560 KMP_DUPLICATE_LIB_OK=TRUE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=cgl \
  python system/halo_system/scripts/run_sim.py --task_config <task>.yaml --web
```

`MUJOCO_GL=cgl` is for macOS; on Linux `run_sim.py` switches it to `egl`. `--web` opens the viewer at
`http://127.0.0.1:8090`. `--duration S` stops after S seconds. No logs are
written; all node output goes to the terminal. Each run folder in `eval_closedloop_sim_multi_output/` holds
only the halo_serve stats and `live_plan.npz` (the viewer's live plan). Operator chat lines typed in the viewer
are appended to `~/.cache/halo/mission_human_input.jsonl` (the orchestrator's inbox; it starts reading at the
end of the file).

## Run real

```bash
# Mac mini, once: install here, copy the repo to the Orin NX and build the tracker there
ssh-copy-id unitree@192.168.123.164
# Ensure both mac mini and Orin NX have access to public internet.
bash system/cpp/scripts/setup_real_deployment.sh --robot_ssh unitree@192.168.123.164 --robot_repo /home/unitree/halo

# Orin NX: start the tracker with the depth policy. Its depth comes from the D435i IR pair through LiteAnyStereo
# (TensorRT); see tracker/deploy_cpp/docs/depth.md (the setup installs librealsense2 and libnvinfer-plugin-dev on the robot). Once:
# run the setup with POLICY_ONNX=<depth policy .onnx> TRACKER_CMAKE_ARGS=-DRT_DEPLOY_WITH_REALSENSE=ON, export
# las_448x256.onnx with tracker/deploy_cpp/tools/export_las_onnx.py (any machine with torch), and build its plan on
# the robot computer:
tracker/deploy_cpp/tools/build_onnx_to_trt.sh las_448x256.onnx tracker/deploy/models/g1/las_424x240_orin.plan \
    --weakly-typed --fp16
export HALO_COMM_REGISTRY=192.168.123.164:5560
sudo -E ./system/build/g1_nx/motiontrack_real_zmq --config tracker/deploy_cpp/configs/g1_v2_depth.yaml \
    --policy <depth_policy>.plan --depth_config tracker/deploy_cpp/configs/depth_d435i.yaml \
    --depth_plan tracker/deploy/models/g1/las_424x240_orin.plan --rt_prio 80 --lock_memory --rt_cpu 1

# Mac mini: generator + orchestrator + viewer
export HALO_COMM_REGISTRY=192.168.123.164:5560
python system/halo_system/scripts/run_real.py --task_config <task>.yaml --motiongen --orchestrator --web
```

On the remote: **Start** → **A** (stand) → **B** (go); **Select** is the emergency stop. The viewer is at
`http://<mini IP>:8090`.
