# Training

The HALO motion tracker is trained in Isaac Lab in three phases, named as in the paper. The task IDs, and the text
below, use the code names `PreRL` for Phase 1 and `PostRL` for Phase 3. Every command below runs from the `tracker/`
directory.

| Phase | Task | Algorithm | Output |
|---|---|---|---|
| 1. Privileged experts | `Halo-MotionTracking-{Robot_Only,Robot_Object,Robot_Terrain}-PreRL` | PPO, one expert per dataset | expert checkpoints (actor + critic): the teachers of Phases 2 and 3 |
| 2. Multi-teacher distillation | `Halo-MotionTracking-Distillation-{Proprio,ProprioDepth}` | on-policy DAgger from per-motion routed teachers | deployable student |
| 3. Multi-critic fine-tuning | `Halo-MotionTracking-PostRL-{Proprio,ProprioDepth}` | PPO with a shared actor and one critic per teacher | final student → ONNX |

Each task has a `-Play` variant for evaluation and export: fewer envs, relaxed terminations, and the robot's domain
randomization off (for PostRL also its sim-to-real terms, with one fixed command lag per reset). The depth
corruption stays on.

- **Experts** see proprioception, a 10-frame motion-reference window with extremity errors, a 30-frame privileged
  history and a foot-relative height map of terrain and objects (`motion_tracking_scene_env_cfg.py`).
  `Robot_Only` tracks free motions on flat ground, `Robot_Object` pushes a box, `Robot_Terrain` sits on a sofa.
- **Students** see only what the robot has onboard: proprioception, the IMU, a 5-frame motion-reference window and a
  30-frame history (`-Proprio`), plus a 48×84 torso depth image (`-ProprioDepth`). A temporal 1-D CNN encodes the
  history, a CNN the depth image, and a SimBa residual MLP maps both to joint-position targets
  (`motion_tracking_scene_distill_cfg.py`, `agents/rsl_rl_distill_cfg.py`).
- **Sim-to-real**: every phase trains on a system-identified G1 (per-joint identified gains, armature and torque
  limits; see [Robot](#robot)) with an explicit PD actuator and a randomized command latency (1–5 physics steps;
  15–40 ms, jittering, in PostRL), action scale `0.25 · τ_max / kp`, a knee negative-power penalty, domain
  randomization (PostRL adds gain, armature, torque-limit, link-mass and per-link CoM DR around the identified
  values, payload mass / inertia / CoM DR, joint stiction, a floor DR that puts a third of the flat-ground envs on a
  compliant, carpet-like floor (contact stiffness 1.3–6.5 kN/m, damping ratio 0.3–1.0), and replaces the contact
  penalty with one that skips the contacts the reference makes, e.g. sitting) and a depth corruption pipeline (5 mm Gaussian noise, 5 % dropout, clutter, latency, side masking,
  camera-mount jitter) on a depth camera that also sees the robot's own body.

## Contents

- [Installation](#installation)
- [Motion data](#motion-data)
- [Robot](#robot)
- [Scene assets](#scene-assets)
- [Phase 1: Privileged experts (PreRL)](#phase-1-privileged-experts-prerl)
- [Phase 2: Multi-teacher distillation](#phase-2-multi-teacher-distillation)
- [Phase 3: Multi-critic fine-tuning (PostRL)](#phase-3-multi-critic-fine-tuning-postrl)
- [Play and export](#play-and-export)
- [Command-line flags](#command-line-flags)
- [Environment variables](#environment-variables)
- [rsl_rl extensions](#rsl_rl-extensions)

## Installation

Isaac Lab 2.3.2 with Isaac Sim 5.1.0 (Python 3.11):

```bash
conda create -n tracker python=3.11
conda activate tracker
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
# Isaac Lab pins flatdict 4.0.1, a source package whose build needs pkg_resources (gone from current setuptools)
pip install "setuptools<80" wheel && pip install --no-build-isolation flatdict==4.0.1
pip install isaacsim[all,extscache]==5.1.0 isaaclab[isaacsim,all]==2.3.2 --extra-index-url https://pypi.nvidia.com
pip install pyarrow   # optional: data/scripts/datagen_to_csv.py --parquet
# minimal Linux images (no X / GL libraries): Isaac Sim's renderer needs libGL for --video
sudo apt-get install -y libgl1 libglu1-mesa libxt6
```

Isaac Sim asks to accept its EULA on the first launch; `export OMNI_KIT_ACCEPT_EULA=YES` accepts it for unattended
runs.

Install the official `rsl-rl-lib` and this extension. Isaac Lab pins an older `rsl-rl-lib`; the tracker needs the
5.5 line, so pip reports a version conflict for `isaaclab`, which is expected and harmless:

```bash
pip install "rsl-rl-lib>=5.5.1,<5.6"
pip install -e training/source/halo_tracker
python training/scripts/list_envs.py      # lists the Halo-* tasks
```

Optional: `wandb login`, then pass `--logger wandb --log_project_name <project>` to `train.py`.

## Motion data

No motion data ships with the repository. Clips are NPZ files produced by kinematically replaying a G1 joint-space
motion (CSV, 30 fps, one row per frame: `x y z qx qy qz qw` + 29 joint angles) in Isaac Lab, which records the body
poses and velocities the tracker needs. The env advances one clip frame per 50 Hz control step, so clips must be
exported at 50 fps (the default; training refuses other rates). Conversion: [data/README.md](../data/README.md).

The commands below and [`configs/teacher_registry_scene.yaml`](configs/teacher_registry_scene.yaml) use this layout:

```
data/g1_processed/
├── robot_only/<dataset>/*.npz             # free-motion clips, any number of dataset folders
├── robot-object/pushbox/push_v2_*.npz     # push-box clips, bound to the push box
└── robot-terrain/labsofa/labsofa_*.npz    # sit clips, bound to the lab sofa
```

Folder names are free; file names are not. A clip's file name binds it to its scene asset (`mdp/commands.py`:
`OBJECT_KEYS_FROM_FILENAME` maps `push_v2*` to the box, `TERRAIN_PREFIX_CATEGORIES` maps `labsofa_*` to the sofa), and
each env only plays clips compatible with the asset it spawned. Push-box clips additionally carry the box trajectory
(`object_pos_w`, `object_quat_w`, `object_lin_vel_w`, `object_ang_vel_w`, world frame), which the CSV converter does
not produce. Sit clips must be expressed in the frame of `labsofa.urdf` (`data/scripts/datagen_to_csv.py` does this
for HALO datagen episodes, [data/README.md](../data/README.md#from-halo-datagen)). Each leaf folder is capped at 1000
clips; when a folder holds more, clips whose file name contains `_original` (e.g. the source clips of augmented
variants) are kept first. A configured folder without NPZ clips stops the run at start-up, and so do clips under a
`robot-terrain` path of which none has a terrain prefix (misnamed sit clips).

## Robot

All phases train on one robot, `robot_cfg.get_g1_sysid_cfg()`: a G1 carrying a rear Mac mini enclosure and head cameras,
identified on hardware (`<repo>/assets/robots/g1/urdf/g1_29dof_macmini_sysid.urdf` + `_isaaclab.json`
with the per-joint stiffness, damping, armature and torque limit). The actuator is Isaac Lab's explicit
`DelayedPDActuator` with a 1–5 physics-step command latency. Self-collision is off: with it on, the hands catch on
the body while pushing and the simulation is slower. The action scale is the nominal `0.25 · τ_max / kp`,
which the deploy configs share. Its MuJoCo twin, `<repo>/assets/robots/g1/mjcf/g1_29dof_macmini_sysid.xml` (same
kinematics, masses, colliders, armature and torque limits, plus the identified joint friction that training leaves
out), is the robot of the MuJoCo deployment.

## Scene assets

Scene objects live in `<repo>/assets/objects`:

| Asset | Clips | Used by |
|---|---|---|
| `pushbox/pushbox.urdf` | `push_v2_*` (object type `box`) | `Robot_Object-PreRL`; SuperScene object |
| `labsofa/labsofa.urdf` + `labsofa.obj` | `labsofa_*` | `Robot_Terrain-PreRL`; SuperScene terrain |

The push box is a free rigid body (0.46 × 0.50 × 0.56 m, 1.0 kg; friction 0.6–0.8 and +0–0.5 kg randomized). The sofa
is fixed; its placement in `labsofa.urdf` matches where the clips' reference robot sits. A clip's file name binds it to
its asset (`mdp/commands.py`: `OBJECT_KEYS_FROM_FILENAME`, `TERRAIN_PREFIX_CATEGORIES`).

The shared scene of Phases 2 and 3 (`SuperScene`) gives each env a flat plate (a 4 × 4 m, 2 cm static box generated in code; its
envs play the robot-only and push-box clips) or the sofa (weights 300 : 57, about 16 % of the envs) and spawns the push box
hidden below the floor, activated for the envs whose current clip pushes it. Pass the sofa clips too, or the sofa envs
fall back to robot-only clips next to an idle sofa.

PostRL swaps the flat plates for 16 × 16 m, 20 cm thick ones (top still at +1 cm, invisible) and gives each a floor
material (`mdp.randomize_floor_compliance`): rigid, or for a third of them compliant, raised by 1 cm like a rug so a
foot can sink up to 2 cm before the ground stops it. The plates are the envs' ray-cast targets, so the height map and
the depth camera see the floor's top. The sofa envs keep the rigid ground; the `-Play` tasks use rigid floors only.


## Phase 1: Privileged experts (PreRL)

Train one privileged expert per motion dataset, with the task of the dataset's type (HALO: 50k iterations, 8192 envs
on one GPU each):

| Task | Scene | Datasets (one expert each) |
|---|---|---|
| `Halo-MotionTracking-Robot_Only-PreRL` | flat ground | `robot_only/<dataset>`: free motion, e.g. LAFAN1 locomotion, dance, … |
| `Halo-MotionTracking-Robot_Object-PreRL` | flat ground + push box | `robot-object/pushbox` |
| `Halo-MotionTracking-Robot_Terrain-PreRL` | lab sofa | `robot-terrain/labsofa` |

```bash
for d in data/g1_processed/robot_only/*/; do        # one Robot_Only expert per dataset folder
  python training/scripts/rsl_rl/train.py --task Halo-MotionTracking-Robot_Only-PreRL --headless \
      --motion_data_folder "$d" --run_name "ro_$(basename "$d")"
done
python training/scripts/rsl_rl/train.py --task Halo-MotionTracking-Robot_Object-PreRL --headless \
    --motion_data_folder data/g1_processed/robot-object/pushbox --run_name object_pushbox
python training/scripts/rsl_rl/train.py --task Halo-MotionTracking-Robot_Terrain-PreRL --headless \
    --motion_data_folder data/g1_processed/robot-terrain/labsofa --run_name labsofa
```

Use `--num_envs` to fit a smaller GPU. A single Robot_Only expert on all of `data/g1_processed/robot_only` also works
(the registry template's `robot_only` entry). With one teacher per dataset, register each as its own entry with
`path_substrings: ["robot_only/<dataset>"]` and a priority above the generic `robot_only` entry's (or remove that
entry), since a motion goes to its highest-priority match.

Height maps scan the shared ground plus each env's terrain mesh and objects, so they are valid wherever a clip
takes the robot; the push-box expert's map includes the box, the same object-aware map the experts see later, as teachers, inside
distillation and PostRL.

Checkpoints land in `logs/rsl_rl/motion_tracking_scene_prerl/<timestamp>_<run_name>/model_<iter>.pt` (the last one
is `model_<max_iterations - 1>.pt`). Copy (or symlink) each teacher's final checkpoint to the `checkpoint_path` of its
entry in [`configs/teacher_registry_scene.yaml`](configs/teacher_registry_scene.yaml) (`checkpoints/prerl/<name>.pt`),
and delete the registry entries of experts you do not train:

```bash
mkdir -p checkpoints/prerl
cp logs/rsl_rl/motion_tracking_scene_prerl/<timestamp>_lafan/model_49999.pt checkpoints/prerl/robot_only.pt
cp logs/rsl_rl/motion_tracking_scene_prerl/<timestamp>_labsofa/model_49999.pt checkpoints/prerl/labsofa.pt
# no push-box clips: delete the object_pushbox entry from configs/teacher_registry_scene.yaml
```

Registry format (`distillation/teacher_registry.py`): one entry per teacher, keyed by an arbitrary name, with
`checkpoint_path` (required; relative to `tracker/` or absolute), the match rules `path_substrings` (full NPZ path),
`filename_prefixes` / `filename_substrings` / `filename_excludes` (file stem) and `priority` (higher wins), plus the
informational `task_id` / `motion_folder`. A motion goes to the highest-priority entry whose non-empty rules all
match. A motion that matches no entry goes to the entry named after its coarse filename category (`robot_only`,
`object`, `terrain`) if one exists, else to the teacher named by the top-level `_default`.

## Phase 2: Multi-teacher distillation

All teachers are distilled into one student on the SuperScene (flat plates + sofa, push box). Each motion is routed
to the teacher trained on it (the registry's match rules), and 20% of the envs roll out with their teacher
(DAgger mixture). HALO: 30k iterations, 4096 envs.

```bash
python training/scripts/rsl_rl/train.py --task Halo-MotionTracking-Distillation-ProprioDepth --headless \
    --teacher_registry_yaml training/configs/teacher_registry_scene.yaml --max_iterations 30000 --num_envs 4096 \
    --motion_data_folder data/g1_processed/robot_only data/g1_processed/robot-object/pushbox \
        data/g1_processed/robot-terrain/labsofa
```

Pass only the folders whose teachers are in the registry: without push-box clips, drop
`data/g1_processed/robot-object/pushbox` here and in the PostRL and play commands (a configured folder without NPZ
clips stops the run at start-up). The push box is then spawned but stays hidden. Use `-Proprio` for the blind student.
Checkpoints: `logs/rsl_rl/motion_tracking_scene_distill_proprio_depth/`.

## Phase 3: Multi-critic fine-tuning (PostRL)

Multi-critic PPO on the distilled student: the actor starts from the student (its exploration std reset to 0.2),
each critic from its teacher's critic, and each critic regresses only the returns of the motions routed to it
(advantages normalized per critic). HALO: 50k iterations, 4 GPUs × 4096 envs.

```bash
NUM_ENVS=4096 python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    training/scripts/rsl_rl/train.py --task Halo-MotionTracking-PostRL-ProprioDepth --headless --distributed \
    --load_checkpoint_path logs/rsl_rl/motion_tracking_scene_distill_proprio_depth/<run>/model_<iter>.pt \
    --teacher_registry_yaml training/configs/teacher_registry_scene.yaml \
    --num_envs 4096 --max_iterations 50000 --motion_data_folder <same folders as distillation>
```

`NUM_ENVS` must match `--num_envs`: it sets `num_mini_batches` (with `WORLD_SIZE`) to keep ~49k-transition
minibatches across ranks.

PostRL also widens the push-box DR (friction 0.3-0.9, mass -0.4 to +5 kg) and samples push clips with probability 0.42
on flat ground; `-Play` keeps the distillation settings.

On Blackwell GPUs (B200) the first multi-GPU launch in a fresh container takes several minutes to start: the NCCL
bundled with torch 2.7 has no sm_100 kernels, so the driver JIT-compiles them into `~/.nv/ComputeCache` (keep it on
persistent storage, or set `CUDA_CACHE_PATH`, to compile only once).

## Play and export

`play.py` loads a checkpoint, exports the deployable policy (the actor, or the distilled student) to
`<checkpoint_dir>/exported/policy.onnx` and `policy.pt`, and plays it:

```bash
python training/scripts/rsl_rl/play.py --task Halo-MotionTracking-PostRL-ProprioDepth-Play --num_envs 64 \
    --checkpoint logs/rsl_rl/motion_tracking_scene_postrl_proprio_depth/<run>/model_<iter>.pt \
    --teacher_registry_yaml training/configs/teacher_registry_scene.yaml \
    --motion_data_folder data/g1_processed/robot_only data/g1_processed/robot-object/pushbox \
        data/g1_processed/robot-terrain/labsofa
# add --export_only to export without stepping the sim, --video to record
```

Headless recording is slow (about 2 s per step with 16 envs on one GPU) and prints nothing until it finishes: the
default 30 s video (`--video_length` 1500 steps) takes about an hour, `--video_length 300` about 10 minutes.

The Distillation and PostRL tasks rebuild their teachers / critics from `--teacher_registry_yaml` before the checkpoint
loads, so the registry's teacher checkpoints must be present for `play.py` too.

The ONNX inputs are the deploy contract: `obs` (1, 434), `policy_temporal` (1, 30, 166) and, for the depth student,
`student_depth` (1, 1, 48, 84) (z-depth clipped to [0.2, 2.5] m → log1p; a new frame every second control step,
i.e. 25 Hz, matching the deployment configs). Deploy it with `deploy/g1/configs/g1_v2(_depth).yaml`
or `deploy_cpp/configs/g1_v2(_depth).yaml` ([deploy/README.md](../deploy/README.md)).

## Command-line flags

`train.py`:

| Flag | Effect |
|---|---|
| `--task` | gym task id (`Halo-MotionTracking-*`) |
| `--num_envs`, `--max_iterations`, `--seed` | override the env count, iteration budget and seed of the task cfgs |
| `--motion_data_folder DIR [DIR ...]` | NPZ folders (globbed recursively by the scene tasks) |
| `--motion_data FILE [FILE ...]` | explicit NPZ files (ignored when `--motion_data_folder` is given) |
| `--teacher_registry_yaml` | teacher registry; required by the Distillation and PostRL tasks |
| `--load_checkpoint_path` | warm start from a checkpoint of another run (iteration count restarts at 0); PostRL: the distillation checkpoint |
| `--resume_optim True\|False` | with `--load_checkpoint_path`: also load the optimizer (default `True`; a PostRL start from a distillation checkpoint always uses a fresh one) |
| `--resume`, `--load_run`, `--checkpoint` | continue a run of the same experiment (`logs/rsl_rl/<experiment>/`) |
| `--experiment_name`, `--run_name` | log folder: `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/` |
| `--distributed` | multi-GPU training under `torch.distributed.run` |
| `--logger {tensorboard,wandb,neptune}`, `--log_project_name` | logging backend |
| `--video`, `--video_length`, `--video_interval` | record training videos |
| `--verbose_isaac` | keep the Isaac Lab / Omniverse warnings (muted by default) |

Isaac Lab's `AppLauncher` flags (`--headless`, `--device`, `--enable_cameras`, ...) and Hydra overrides of the task
cfgs (e.g. `env.scene.num_envs=1024`) are accepted as well.

`play.py` takes `--task`, `--num_envs`, `--motion_data_folder` / `--motion_data`, `--teacher_registry_yaml`,
`--checkpoint` (or `--load_run` to pick the latest checkpoint of a run), plus `--export_only`, `--video`,
`--video_length` (default 30 s), `--robot_id` (env to follow with the video camera, 1-based), `--real-time` and
`--disable_fabric`.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `NUM_ENVS` | `4096` | PostRL env count per GPU used to size `num_mini_batches` (must match `--num_envs`) |

## rsl_rl extensions

Training uses the official [`rsl-rl-lib`](https://github.com/leggedrobotics/rsl_rl) plus a small extension package,
[`halo_tracker.rsl_rl_ext`](source/halo_tracker/halo_tracker/rsl_rl_ext), referenced from the agent configs by
qualified class name:

- `HaloCNNModel`: `CNNModel` with a SimBa residual MLP trunk, orthogonal init, and `(B, T, F)` history groups encoded
  by a temporal 1-D CNN.
- `MultiTeacherDistillation`: per-motion routed teachers and the DAgger rollout mixture (run by the upstream
  `DistillationRunner`).
- `MultiCriticPPO`: PPO with a shared actor and per-motion routed critics.

PreRL uses the upstream `PPO`. `train.py` also replaces upstream's NaN check (which raises) with
`rsl_rl_ext.utils.sanitize_nan`: a NaN observation or reward is zeroed and its env marked done, so one diverging env
does not end the run.
