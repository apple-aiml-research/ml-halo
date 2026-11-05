# `render/` — render stage (episode spec + qpos → OAK / TPV video → training dataset)

Replays each generated trajectory inside an Infinigen room with Isaac Sim (the reference dataset) or MuJoCo,
renders the G1's OAK-1W head camera (THOR model; the FPV) and a fixed third-person view (TPV), applies the seeded
visual domain randomization (cam_fx) to the OAK video, and writes the episode into the training dataset.

## Inputs and outputs

In: `--npy-dir/episode_*/{g1_qpos.npy, episode_spec.json}` (scene-free sit specs, DR drawn at gen time)
and one room `--scene-dir` (`asset/scan.usdc` for Isaac, `asset/scan.obj` for MuJoCo, plus
`scene_analysis.json`). Out under `--output-dir`: `episodes/<ep>/{fpv,tpv,oak}.mp4, traj.npz, meta.json`
(resume state) and `dataset/{data,videos,episode_specs}/` with dataset-wide episode indices (the trailing
number of the spec dir) and `meta/shard_NNN/` for the stitch ([data_formats.md](../docs/data_formats.md)).
A shard with 0 episodes exits non-zero; set `SHARED_PREFIX_ROOT` for any re-render. After the last shard,
`python -m render.stitch --mode stitch --source-prefix $HALO_S3_ROOT/<dataset>` writes the top-level `meta/`.

## Flow

```
configs/pipeline/render.env
 └ render/pod/run_render_{isaac,mujoco}.sh     pull specs, group by scene (render_common.sh), pull rooms
    └ render/render_episodes.py                   one call per scene
       └ render/isaac/driver.py run_isaac         boot Kit, load room, prespawn the chairs, per episode:
           1. place_episode                       se2_fit_place (synthetic_scene/robot_placement.py)
                                                  + TPV pick (tpv.py select_infinigen_tpv)
           2. place_episode_objects               isaac/episode_objects.py (chair from the prop pool)
           3. set_episode_lighting                the spec's dome + overhead light
           4. run_simulator                       isaac/replay.py → OAK (native fisheye) + TPV frames
           5. _finalize                           cam_fx on OAK (cam_fx.py), mp4, meta.json, dataset write
```

MuJoCo (`mujoco/driver.py run_mujoco`) builds the unlit room + THOR robot scene once
(`synthetic_scene/build_infinigen_mjcf.py`), adds the chair per episode (`mujoco/episode_objects.py`),
renders with `mujoco/replay.py` (OAK as a 1600×1000 pinhole), and the dataset writer warps OAK to the
calibrated fisheye once (`fisheye_warp.py`); cam_fx is applied after the warp. Placement and seeds are
the same as Isaac's.

## Visual DR

Seeded per episode (`driver_common.seed_for_episode`). Drawn at gen time into the spec
(`motion_gen/tasks/sit.py`, `VDR_LEVEL=light`: `cam_artifacts` ranges, `motion_blur_gain=0.3`) and applied
by `cam_fx.py` to the OAK video only: brightness → contrast → gamma → color tone → saturation →
exposure guard → chroma → motion blur → defocus → noise → salt/pepper → posterize → scanline →
vignette. One RNG per camera per episode, so the draw order is part of the output; Isaac draws serially
and applies on 12 threads (identical output).

## Module map

| path | role |
|---|---|
| `driver_common.py` | per-episode seed, resume check, spec pairing, placement, dataset index, cam_fx frame apply |
| `render_episodes.py` | the per-scene CLI the stage scripts call |
| `dataset_writer.py` | episode parquet + videos + episode spec, per-shard meta, ffmpeg pipe writer |
| `stitch.py` | `--mode stitch`: the per-shard meta of a direct-write render → top-level `meta/` |
| `tpv.py` | per-episode TPV camera choice |
| `pod/` | stage scripts of `configs/pipeline/render.env`: `setup_render_{gpu,cpu}.sh`, `run_render_{isaac,mujoco}.sh`, `render_common.sh` |
| `isaac/driver.py`, `mujoco/driver.py` | the per-backend render loops |
| `isaac/replay.py`, `mujoco/replay.py` | per-frame replay + camera capture |
| `isaac/scene_builder.py` | THOR G1 from URDF, two-tone colours, OAK fisheye / TPV cameras |
| `isaac/episode_objects.py`, `mujoco/episode_objects.py` | spec props → USD prims / MJCF bodies |
| `assets.py` | chair lookup (`assets/objects/chairs`) and the USDZ mesh reader both engines use |
| `cam_fx.py` | sensor artifacts |
| `cameras.py`, `fisheye_warp.py`, `oak_intrinsics.json` | OAK mount + intrinsics; pinhole → fisheye warp |
| `synthetic_scene/` | scene analysis (run by scene_gen), MuJoCo / Isaac room loaders, se2_fit placement |

## Invariants

- Isaac: end with `os._exit(0)` after flushing stdio; `SimulationApp.close()` hangs on Kit teardown.
