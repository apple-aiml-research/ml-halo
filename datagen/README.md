# datagen

The synthetic data pipeline behind HALO's training data. It builds indoor rooms with
[Infinigen](https://github.com/princeton-vl/infinigen) and bakes their lighting into textures, generates sit-on-chair
motions for the Unitree G1 with [Kimodo](https://github.com/nv-tlabs/kimodo) (checked by geometric verifiers, `motion_check`),
places each motion into a room and renders the robot's head camera (OAK-1W fisheye) and a third-person view with
Isaac Sim. The rendered dataset trains the **generator**; the motions train the **tracker**.

```
1. scene gen (Infinigen + bake) ───────┐
                                       ├─► 3. render (placement + Isaac + visual DR, stitch) ─► dataset ─► generator
2. motion gen (Kimodo + motion check) ─┤
                                       └──────────────────────────────────────────────────────► specs ───► tracker
```

## Requirements

Linux with an NVIDIA driver (R580; R595 crashes Isaac Sim 5.1), `git`, `curl`, `ffmpeg` with libx264. The setup
scripts install system packages with `apt-get` (Debian / Ubuntu, as root) and Miniconda if `conda` is missing.

| Step | Hardware (per shard) | Software (installed by the step's setup script) |
|---|---|---|
| 1. Scene gen | 1 NVIDIA GPU, ~60 GB RAM, 300 GB disk | Blender 5.2.0, Infinigen `892947be` (Python 3.11, `bpy==4.2.0`) |
| 2. Motion gen | 1 CUDA GPU (tested on 80 GB H100 / B200), 200 GB disk | Python 3.10, torch 2.11 cu128, Kimodo `58e78189`, 4 Hugging Face models (~17.5 GB, below) |
| 3. Render | 1 NVIDIA RTX GPU, > 16 GB RAM, 500 GB disk | Isaac Lab 2.3 (Isaac Sim 5.1) |

## Setup

```bash
git clone https://github.com/apple-aiml-research/ml-halo.git && cd ml-halo/datagen
export HALO_S3_ROOT=/data/halo   # where every step reads and writes: a local / NFS dir, or s3://bucket/prefix
                                 # (standard AWS_* credentials; AWS_ENDPOINT_URL for MinIO / R2 / ...)
export HF_TOKEN=hf_...           # step 2: accept the Llama-3 license at
                                 # https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct first
```

Step 2 downloads [`nvidia/Kimodo-G1-RP-v1`](https://huggingface.co/nvidia/Kimodo-G1-RP-v1),
[`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp),
[`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised`](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised)
and [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) (gated);
each model card has its license. Steps 1 and 3 need no weights; the robot and chair assets ship in `../assets/`.

Each step's setup script runs automatically (`launch/run_stage.sh` calls it before the step; `--no-setup` skips
it) and skips whatever is already installed:

| Step | Setup script | What it installs |
|---|---|---|
| 1 | `scene_gen/env/setup_scene_gen.sh` | Blender 5.2.0 from blender.org → `/opt/blender52`; Infinigen at the pinned commit → `/opt/infinigen_src`; its Python 3.11 env → `/opt/infinigen_env` (override with `BLENDER_DIR`, `INFINIGEN_ROOT`, `INFINIGEN_ENV`) |
| 2 | `motion_gen/pod/setup_kimodo.sh` | conda env `kimodo` (torch, Kimodo); downloads the 4 Hugging Face models into `~/.cache/huggingface` |
| 3 | `render/pod/setup_render_gpu.sh` | Python packages into Isaac Lab. Run step 3 inside the official Isaac Lab container `nvcr.io/nvidia/isaac-lab:2.3.0` (`ACCEPT_EULA=Y`), or point `ISAACLAB_SH` at your `isaaclab.sh` |

Step 3 in the container, from the repo root (`ISAACLAB_SH` defaults to the container's
`/workspace/isaaclab/isaaclab.sh`; pass your data root and, for `s3://`, the `AWS_*` variables):

```bash
docker run --rm --gpus all -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e HALO_S3_ROOT \
    -v $PWD:/halo -v /data/halo:/data/halo -w /halo/datagen --entrypoint bash nvcr.io/nvidia/isaac-lab:2.3.0 \
    launch/run_stage.sh render --shard 0 --num-shards 250
```

Step 3 without the container: pip-install Isaac Sim 5.1.0 + Isaac Lab 2.3.0 into a Python 3.11 env and point
`ISAACLAB_SH` at a stand-in that only does `-p` (the tracker's Isaac Lab 2.3.2 install works too):

```bash
conda create -y -p /opt/isaac_env python=3.11 && PIP=/opt/isaac_env/bin/pip
$PIP install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
$PIP install "setuptools<80" wheel && $PIP install --no-build-isolation flatdict==4.0.1   # flatdict needs pkg_resources
$PIP install "isaacsim[all,extscache]==5.1.0" "isaaclab[isaacsim,all]==2.3.0" --extra-index-url https://pypi.nvidia.com
printf '#!/bin/bash\n[ "$1" = -p ] && shift\nexec /opt/isaac_env/bin/python "$@"\n' > /opt/isaaclab.sh
chmod +x /opt/isaaclab.sh && export ISAACLAB_SH=/opt/isaaclab.sh OMNI_KIT_ACCEPT_EULA=YES
```

Step 2 without `HF_TOKEN` on every machine: download the models once (step 2's setup with `HF_TOKEN`), then copy
the cache to `$HALO_S3_ROOT/kimodo-models/`; the setup syncs from there when it exists:

```bash
for m in nvidia--Kimodo-G1-RP-v1 McGill-NLP--LLM2Vec-Meta-Llama-3-8B-Instruct-mntp \
         McGill-NLP--LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised meta-llama--Meta-Llama-3-8B-Instruct; do
    launch/halo_s3 sync ${HF_HOME:-~/.cache/huggingface}/hub/models--$m/ $HALO_S3_ROOT/kimodo-models/models--$m/ --exclude 'blobs/*'
done
```

## Run the pipeline

Run from `datagen/`, one command per shard. Start it once per shard on any machine or scheduler
(`--shard i --num-shards n`); every step resumes from what is already at its output, so rerunning a shard is safe.
`--dry-run` prints what a command would run.

| Step | Command (one shard) | Expect |
|---|---|---|
| 1. Scenes (scene generation, baking) | `launch/run_stage.sh scene_gen single --shard 0 --num-shards 300 LIGHT_GAIN=2.0` | 4 scenes, ~30-60 min each: `scenes/single/scene_NNNNNN/` with `asset/{scan.obj,scan.mtl,scan.usdc,tex/}`, `anchor.json`, `views_64.json`, `scene_analysis.json`, `bake/`, `logs/`; a scene is done when `scenes/single/_done/<scene>.json` exists |
| 2. Motion (kinematic motion generation, filtering) | `launch/run_stage.sh gen --shard 0 --num-shards 250` | 200 episodes in ~2.5-3.5 h: `halo_sit_n50k/specs/shard_000/episode_NNNN/{episode_spec.json,g1_qpos.npy,meta.json,plan.npy}`; the log shows `[sit] SHARD 0 DONE (200/200 eps)` |
| 3. Render (scenario staging, egocentric rendering) | `launch/run_stage.sh render --shard 0 --num-shards 250` | 200 episodes over 4 scenes in ~1-1.5 h (H100 or B200): `halo_sit_n50k/{data,videos,episode_specs,meta/shard_NNN}/`; log ends `=== multiscene done: 200 episodes, 4 scenes ===` |

Render shard `i` renders rows `4i .. 4i+3` of its scene list (default `common/manifests/render_scenes_1000.txt`, the
scenes of the paper data), not the scenes that scene shard `i` built. To render scenes you built yourself, list their
names (`scene_NNNNNN`, one per line, all present under `scenes/single/`) in a file and add `SCENE_MANIFEST=<file>`
together with a new `SHARED_PREFIX_ROOT=$HALO_S3_ROOT/rendered_output/<name>` (the render resumes from there, and
episodes rendered in other scenes would count as done).
A listed scene that your scene gen did not build (it is under `scenes/single/_failed/`, or still missing after
the shard reran) stops the render shard that needs it: rerun its scene shard, or put a built scene in its row of your
own `SCENE_MANIFEST` with a new `SHARED_PREFIX_ROOT` ([scene_gen/README.md](scene_gen/README.md#pitfalls)).
Render reads the motions at `SPEC_S3_PATH` (default `halo_sit_n50k/specs`, what step 2 writes); the shard counts
of steps 2 and 3 must match.

After the last render shard, stitch the per-shard meta into the top-level `meta/` (prints
`[PASS] meta == data == fpv == tpv (N episodes)`). It runs on CPU in any Python 3.10+ env with
`pip install -r requirements.txt` (plus the AWS CLI for an `s3://` root); inside the Isaac Lab container use `isaaclab.sh -p -m render.stitch ...`:

```bash
python -m render.stitch --mode stitch --source-prefix $HALO_S3_ROOT/halo_sit_n50k
```

Full scale (the paper data): 300 scene shards, 250 motion shards (50,000 episodes), then 250 render shards over the
1,000 scenes in `common/manifests/render_scenes_1000.txt`. Steps 1 and 2 are independent; step 3 needs both.
`scene_gen dual --num-shards 500` builds the dual-room scenes (Door skill: step sideways through a half-open door) under `scenes/dual/`. To try a
change first, run a small chain and watch one FPV and one TPV video (the robot walks inside the room, turns and
sits on the chair). Its motions, dataset and render resume state go under `try/` and `rendered_output/try`, never
into the paper prefixes; use a new name per try (both steps skip episodes already there):

```bash
launch/run_stage.sh scene_gen single --shard 0 --num-shards 300 LIGHT_GAIN=2.0   # 4 scenes
launch/run_stage.sh gen --shard 0 --num-shards 250 EPS_PER_SHARD=4 SPEC_S3_PATH=try/specs   # the first 4 motions of shard 0
launch/halo_s3 ls $HALO_S3_ROOT/scenes/single/_done/ | awk '{print $NF}' | sed 's/\.json$//' > my_scenes.txt
launch/run_stage.sh render --shard 0 --num-shards 250 EPS_PER_SHARD=4 MAX_TRAJS=3 SCENE_MANIFEST=my_scenes.txt \
    SPEC_S3_PATH=try/specs DIRECT_WRITE_DEST=try SHARED_PREFIX_ROOT=$HALO_S3_ROOT/rendered_output/try
python -m render.stitch --mode stitch --source-prefix $HALO_S3_ROOT/try
```

## Output data layout

Everything lands under `HALO_S3_ROOT`:

```
$HALO_S3_ROOT/
├── scenes/single/                            step 1 (scenes/dual/ for the dual-room set)
│   ├── scene_NNNNNN/                         asset/{scan.obj,scan.mtl,scan.usdc,tex/}, anchor.json, views_64.json,
│   │                                         scene_analysis.json, bake/bake_manifest.json, logs/gen.log, ...
│   └── _done/scene_NNNNNN.json               one marker per finished scene
├── halo_sit_n50k/
│   ├── specs/shard_NNN/episode_NNNN/         step 2, the motions: episode_spec.json, g1_qpos.npy, meta.json, plan.npy
│   ├── data/episode_NNNNNN.parquet           step 3 + stitch, the training dataset
│   ├── videos/observation.images.{fpv,tpv}_cam/episode_NNNNNN.mp4
│   ├── episode_specs/episode_NNNNNN.json
│   └── meta/{info.json,episodes.jsonl,tasks.jsonl}, meta/shard_NNN/
└── rendered_output/<RENDER_BATCH>/shard_NNN/episodes/   per-episode renders, the render's resume state
```

Who reads what:

| Reader | Reads | How |
|---|---|---|
| generator | `halo_sit_n50k/` (`data/`, `videos/`, `meta/`; no `dataset/` level) | set `data.data_dir` to `$HALO_S3_ROOT/halo_sit_n50k` (local dir or `s3://`); formats: [docs/data_formats.md](docs/data_formats.md) |
| tracker | `halo_sit_n50k/specs/` | copy it to a local dir (`--specs` takes a local path), then `tracker/data/scripts/datagen_to_csv.py --specs <local specs> --out <csv> --accepted_only` (NumPy only) and `batch_process.py` to NPZ (Isaac Lab's python with the tracker installed: [tracker/training/README.md](../tracker/training/README.md), [tracker/data/README.md](../tracker/data/README.md)). Training keeps at most 1000 clips per folder (`max_motions_per_subfolder`) |

## More

- [docs/data_pipeline.md](docs/data_pipeline.md): every step in detail, settings, resume, pitfalls
- [docs/data_formats.md](docs/data_formats.md): output formats (the contract with the loader); [docs/coordinate_frames.md](docs/coordinate_frames.md)
- Per folder: [`scene_gen/`](scene_gen/README.md), [`motion_gen/`](motion_gen/README.md), [`motion_gen/motion_check/`](motion_gen/motion_check/README.md) (the gate), [`render/`](render/README.md) (render flow, visual DR, stitch)

| Folder | What |
|---|---|
| `scene_gen/` | step 1: Infinigen scene generation + bake + scene analysis, setup (`env/`), scene lists (`spec/`) |
| `motion_gen/` | step 2: episode spec, sit task, Kimodo generation, `motion_check/` gate, setup + run scripts (`pod/`) |
| `render/` | step 3: Isaac + MuJoCo render, cameras, visual DR, placement, TPV choice, dataset writer, stitch (`pod/`) |
| `common/` | shared: storage CLI (`storage.py`), render scene lists (`manifests/`), placement |
| `launch/` | `run_stage.sh`, `halo_s3` (storage CLI), `vkfix.sh`, `ensure_storage.sh` |
| `configs/` | `pipeline/` (one env file per step), `motion_check/` (thresholds) |
