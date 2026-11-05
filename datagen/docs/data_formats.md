# Data Formats

File formats produced and consumed by the data pipeline (`docs/data_pipeline.md`). The training-dataset layout is what the
training loader (`generator/data/dataset.py`) reads; do not change it.

## Motion stage output (per episode)

```
specs/shard_NNN/episode_NNNN/
  g1_qpos.npy          float32 [T, 36]
  episode_spec.json    motion_gen.episode_spec.EpisodeSpec.to_dict()
  meta.json            gen record: phases, rounds_used, n_candidates, final_gen_accepted,
                       kept_score, kept_blocking, final_checks, staged timings
  plan.npy             [Np, 3] planned (x, y, yaw) approach
```

`g1_qpos.npy` columns: `[root_x, root_y, root_z, qw, qx, qy, qz, j0 … j28]`
- root xyz in metres, world Z-up (floor at z=0; the render lifts it to the scene `floor_z`)
- quaternion wxyz (w first)
- joints in radians, MuJoCo / deployment joint order (`render/dataset_writer.py::G1_JOINT_NAMES`)
- 30 fps; sit episodes in the reference dataset are typically 300-360 frames

## Render output (per episode, before the dataset writer)

```
<output_dir>/episodes/<episode_name>/
  oak.mp4              head camera (OAK-1W fisheye; Isaac renders it natively, MuJoCo as a pinhole the writer
                       warps); becomes the dataset's fpv_cam
  tpv.mp4              third-person view
  fpv.mp4              MuJoCo only: its pinhole head camera (not used by the dataset)
  traj.npz             qpos[T, 36] + num_frames
  meta.json            render/driver_common.py::episode_meta: scene, seq_name, episode_name, valid_frames,
                       source, episode_spec (the placed spec), timing (render_s, encode_wait_s, mode)
```

`render/driver_common.py::episode_is_complete` treats an episode as done when `meta.json`, `traj.npz`,
`tpv.mp4` and `oak.mp4` exist and are ≥256 bytes; a shard with fewer complete episodes than it was given exits 1.

## Training-dataset layout

```
<dataset>/
  data/episode_NNNNNN.parquet            per-frame robot state + action
  videos/observation.images.fpv_cam/episode_NNNNNN.mp4
  videos/observation.images.tpv_cam/episode_NNNNNN.mp4
  episode_specs/episode_NNNNNN.json      the placed spec of the render: floor_z (floor the parquet pose
                                         stands on), scene_id, scene_s3_path (key under HALO_S3_ROOT),
                                         scene_dir (always empty; the room is scene_s3_path)
  meta/
    info.json                            codebase_version, robot_type, totals, fps, paths, features
    episodes.jsonl                       per-episode index, length, tasks
    tasks.jsonl                          task vocabulary
    shard_NNN/                           per (shard, scene) meta written in DIRECT_WRITE mode; stitch
                                         concatenates them into the three files above
```

Episode ids (here and in `specs/` dir names) are gapped: `gid = shard * 1000 + local_index` (0..199, 1000..1199,
...); `generator/data/dataset.py` looks episodes up by `episode_index` value.

`info.json` `features` lists `action` and `observation.joint_pos`, each `[29]` with joint names in qpos order.
Eval code maps model output to joints through `features.action.names`.

Videos: libx264 yuv420p, crf 23, 30 fps. FPV: OAK-1W fisheye 1280x800 with visual DR; TPV: 848x480, fixed, no DR.

## Parquet row schema (per frame)

| Column | Type | Description |
|---|---|---|
| `action` | float32[29] | joint angle change to the next frame (`q[t+1] - q[t]`), 0 on the last frame |
| `observation.joint_pos` | float32[29] | joint angles (rad) = `qpos[7:]` |
| `observation.body_pos` | object[30] | body positions; only `[0]` (pelvis xyz) filled, rest zero |
| `observation.body_quat` | object[30] | body quaternions wxyz; only `[0]` (pelvis) filled, rest identity |
| `info.task_name` | str | task string |
| `info.objective_category` | str | target object id |
| `info.verb_category` | str | task type (`sit`) |
| `info.task_type` | str | `sit` |
| `info.target_object_id` | str | target placement name |
| `info.target_xyz` | float32[3] | target position |
| `info.scene_id` | str | the infinigen scene name (e.g. `scene_000000`) |
| `timestamp` | float32 | `frame_index / fps` |
| `frame_index` | int64 | frame within the episode |
| `episode_index` | int64 | global episode id (direct-write) |
| `index` | int64 | same as `frame_index` |
| `task_index` | int64 | always 0 |

The parquet row count is the episode length recorded in `meta/episodes.jsonl`.
