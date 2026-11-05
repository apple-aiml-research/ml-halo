# Staged data pipeline

How the HALO sit-on-chair training data is made, step by step; formats: [data_formats.md](data_formats.md). Paths
are relative to `datagen/`, data paths relative to `$HALO_S3_ROOT`. Setup: [../README.md](../README.md#setup).

## Running a step

```bash
export HALO_S3_ROOT=s3://my-bucket/halo        # or a local / NFS directory
launch/run_stage.sh <stage> [variant] [--shard I --num-shards N] [KEY=VALUE ...] [--dry-run] [--no-setup]
launch/run_stage.sh render eval SCENES=10 --shard 3 --num-shards 250 --dry-run    # print what it would run
```

`run_stage.sh` exports `configs/pipeline/<stage>.env` with `VARIANT` and the `KEY=VALUE` overrides (visible to
the env file, which picks its per-variant values from them, and applied again after it), sets `SHARD_IDX` /
`SHARD_ID` / `TOTAL_SHARDS` and `OUTPUT_DIR` (default `outputs/<STAGE_TAG>/shard_<I>`, e.g.
`outputs/render_eval.scene10/shard_003`), then runs `STAGE_SETUP` and `STAGE_CMD`. Start it once per shard from
any scheduler; `STAGE_GPUS` and each env file's header say what to run it on. Shards resume from what exists at
their output prefix, so rerunning is safe. All I/O goes through `common/storage.py` / `launch/halo_s3`: an `s3://`
root uses the AWS CLI (`AWS_*` credentials, `AWS_ENDPOINT_URL` for other S3 stores), a local root plain copies.

| # | Step | Command | Entry point | Output |
|---|---|---|---|---|
| 1 | Scene generation + bake + analysis | `scene_gen [single\|dual]` | `scene_gen/pod/run_shard.sh` | `scenes/{single,dual}/scene_NNNNNN/` (with `scene_analysis.json`) |
| 2 | Motion generation | `gen [prod\|eval]` | `motion_gen/generate_sit_traj.py` | `<dataset>/specs/shard_NNN/episode_NNNN/` |
| 3 | Scenario staging (placement, se2_fit) + egocentric rendering + visual DR | `render [prod\|eval\|mujoco] [SCENES=N]` | `render/render_episodes.py` | `<dataset>/{data,videos,episode_specs,meta/shard_NNN}/` |
| | stitch, after the last render shard | CLI | `python -m render.stitch --mode stitch` | `<dataset>/meta/` |

Datasets (all 250 shards × 200 = 50,000 episodes, the default variant first):

| Variant | Specs (`SPEC_S3_PATH`) | Render (`DIRECT_WRITE_DEST`) | Use |
|---|---|---|---|
| `gen` / `render` | `halo_sit_n50k/specs` | `halo_sit_n50k` (1000 scenes) | the paper dataset |
| `render SCENES=1\|10\|100` | same specs | `halo_sit_scene{1,10,100}_n50k` | scene-count ablation |
| `gen eval` / `render eval [SCENES=N]` | `halo_sit_eval_n50k/specs` | `halo_sit_eval[_sceneN]_n50k` | closed-loop eval stance |
| `render mujoco [SCENES=N]` | same specs | `halo_sit_mujoco[_sceneN]_n50k` | the MuJoCo render (same placements as Isaac) |

`RENDER_BATCH` is a seed string: it seeds placement and visual DR per episode, one per prod scene count, one for
all eval counts (same DR seeds); keep the values in `render.env` to reproduce the paper data. A small test run
(`EPS_PER_SHARD=4`, render `MAX_TRAJS=3`, see the README) reproduces the first episodes of shard 0 (motion seeds
depend only on `SEED_BASE`, shard, episode). Give it its own `SPEC_S3_PATH` / `DIRECT_WRITE_DEST` /
`SHARED_PREFIX_ROOT`, new per try: both steps skip episodes already there, and the render resumes from
`SHARED_PREFIX_ROOT` (default `rendered_output/<RENDER_BATCH>`, the paper run's).

## Inputs

Everything comes from the repo or public sources; nothing has to be staged under `HALO_S3_ROOT` first.

| What | Where | Used by |
|---|---|---|
| Blender 5.2.0, Infinigen `892947be` | download.blender.org, github.com/princeton-vl/infinigen (`scene_gen/env/setup_scene_gen.sh`) | step 1 |
| Kimodo-G1 and LLM2Vec / Llama-3-8B weights (~16 GB, Llama-3 is gated: `HF_TOKEN`) | Hugging Face (`motion_gen/pod/setup_kimodo.sh`; if `$HALO_S3_ROOT/kimodo-models/` holds a copy of the HF cache it is synced from there instead); the run itself is offline | step 2 |
| THOR robot, the chair | `../assets/robots/g1/`, `../assets/objects/chairs/arm_chair.usdz` | steps 2-3 |
| scene lists | `scene_gen/spec/scenes_{single,dual}.json` (what step 1 builds); `common/manifests/render_scenes_{1,10,100,1000}.txt` (what step 3 renders); held-out `heldout_scene_ids.txt`, `heldout_val20.txt` | steps 1, 3, eval |
| keyframes | `motion_gen/kimodo/keyframes/` (stand / walk end / turn end / step / sit; `single_kf_eval.npz` is the `default_angles` of `tracker/deploy/g1/configs/g1_v2.yaml`) | step 2 |

## Step 1: scene generation, bake and analysis

Infinigen in Blender, one scene per scene-list entry, lighting baked into unlit textures so MuJoCo and Isaac show
the Blender image without engine lights. Commands, knobs, outputs, scene lists and pitfalls:
[../scene_gen/README.md](../scene_gen/README.md).

Scene analysis, the last step of each scene in `scene_gen/pod/run_shard.sh` (before its upload and done marker),
writes `scene_analysis.json` per scene (object boxes, floor height, 2D occupancy grid, spawn and chair seeds,
ranked third-person views); se2_fit placement and the TPV choice need it (the render raises without
it). A scene it cannot analyse still ships, with a warning in the log.

- One scene by hand: `python -m render.synthetic_scene.analyze_scene --scene-dir <dir>` (reads `asset/scan.obj`,
  `views_64.json`; writes `scene_analysis.json`, `scene_analysis_topdown.png`).
- Keep `OBSTACLE_MIN_HEIGHT_M` (`render/synthetic_scene/analyze_scene.py`) at 0.08 m (it changes placements).
  Floors sit at z ≈ 0.05-0.15 m; everything downstream lifts to `floor_z`.

## Step 2: motion generation

Per episode: sample a sit spec, generate walk → turn → sit with Kimodo, gate each candidate with `motion_check`,
keep the best. Motion is scene-free (chair at the origin, no scene pulled) and placed in
step 3, so one gen feeds every scene count.

- `gen --shard I --num-shards 250` (settings in `gen.env`: standing keyframe `standpose`, eval `eval`; plateau
  turn `TURN_NF=50`; `CHAIR_SET=pair` = the paper's 50/50 chair draw, both slots the arm chair). Without the launcher, on CUDA:
  `python motion_gen/generate_sit_traj.py --n 4 --out ./sit_test --device cuda`
  (`--help` for knobs).
- Spec (`motion_gen/tasks/sit.py`, `motion_gen/episode_spec.py`): seat 0.45 m high, stop 0.55 m in front, spawn
  1.5-3.0 m away within ±70° of the chair's facing; visual DR from a separate seed stream into
  `spec.cam_artifacts`.
- Generation (`motion_gen/kimodo/{model,segments}.py`): 16 WALK candidates per batch ranked by (hard failures,
  foot skate, score); TURN and SIT start from the previous segment's last frame, 2 retries each; if they fail
  the WALK is resampled (up to 5 resamples, 6 rounds). The best chain is always kept; `meta.json` `final_gen_accepted` says
  whether it passed (77.3% of the 50k).
- Bit-reproducible on the GPU type the paper data used (H100): a re-run on H100 reproduces the paper motions
  bit for bit, with the same gate verdict per episode (checked on shards 0-49, 10,000 episodes). Other GPUs (B200) give
  different but statistically similar motions (`cudnn.benchmark`, TF32, batch-dependent Kimodo forward); the spec
  part (DR, chair, spawn) is identical everywhere.
- Gate: `motion_check` (`configs/motion_check/`, tier `industry`, process pool in `motion_gen/check_worker.py`),
  see [../motion_gen/motion_check/README.md](../motion_gen/motion_check/README.md).

## Step 3: scenario staging (placement), egocentric rendering, visual DR and stitch

Placement: one 2D transform (x, y, yaw) that puts the whole walk path and the chair in free space, moving the
chair with it, feet lifted to the scene floor: `se2_fit_place` (the only placement mode, both renderers),
seeded by `render/driver_common.py::seed_for_episode(RENDER_BATCH, shard, episode)`. If nothing fits it prints
`no path-free placement; falling back to static scene spawn`. TPV camera per episode:
`render/tpv.py::select_infinigen_tpv`.

Render: replays each placed motion, records the OAK-1W head-camera fisheye (THOR model; as FPV) and a third-person camera,
applies sensor DR to the head camera only, writes training-dataset episodes to the dataset prefix. Code flow and DR knobs:
[../render/README.md](../render/README.md).

- `render --shard I --num-shards 250` (Isaac RTX); `render mujoco` is the CPU MuJoCo path (same scenes and
  placement). Scenes: shard `i` renders rows `i*4 .. i*4+3` of `SCENE_MANIFEST` (the specs carry no scene),
  its 200 episodes round-robin; meta goes to `meta/shard_<i*4+k>/`. Gen and render must use the same
  `TOTAL_SHARDS`.
- Resume: `episodes/` and `dataset/` upload every `ROLLING_UPLOAD_SECS`; a restarted shard pulls its prefix,
  redoes the last `ROLLBACK_EPS`=2 episodes and continues. Use a new `SHARED_PREFIX_ROOT` for a new run (also
  whenever `SCENE_MANIFEST` changes) and never write into an existing dataset prefix. A shard that renders 0 episodes exits non-zero.
- TPV vs the paper datasets: when none of a scene's ranked views sees the path (~5-10% of episodes), the
  fallback camera now stays inside the room; the paper data used an older fallback that could put it
  outside, so those TPV videos differ. FPV, specs and placement are unchanged.
- A missing chair asset raises. Isaac RTX is not bit-reproducible (~36 dB Y-PSNR between identical runs); watch
  one OAK and one TPV video before scaling up. `PUSH_TO_S3=0` keeps output local.

Stitch, after the last render shard: concatenates the per-shard meta into the top-level meta (no parquet / video
download), then checks the meta episode set equals the `data/` and `videos/` file sets:

```bash
python -m render.stitch --mode stitch --source-prefix $HALO_S3_ROOT/<dataset>
```

No `--expected-shards` for a multi-scene render (one meta dir per (shard, scene)); rerun it if it fails. Joint
order is `info.json` `features.action.names` (deployment order); do not remap it.

## Reproducing the paper data

Run steps 1-3 in order with new output prefixes (`OUT_S3`, `SPEC_S3_PATH`, `DIRECT_WRITE_DEST`,
`SHARED_PREFIX_ROOT`) and `LIGHT_GAIN=2.0` for scenes, then stitch. Check episode counts per
step (specs → parquet → fpv → tpv, 1:1) and rerun missing shards before stitching. Infinigen is not
byte-deterministic after the seed: the room shell, size, floor, style and windows match; furniture, decor and mesh
detail can differ run to run, and with them the placements and TPV views. Expect about 80% of the episodes to get
the paper's chair placement; most rebuilt scenes match the paper in every episode, about one in seven in none. The
differences come from Infinigen run to run, not from the GPU type.

Not shipped: the gated / truncate-pad subsets, the go_to and pick_up tasks, the unit and golden test suites, the
reproduction metrics, the scene-bake cross-engine QA renders and the debug / visualization tools.
