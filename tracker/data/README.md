# Motion data

The tracker trains on NPZ motion clips. No data ships with this repository.

## Source clips (CSV)

A clip is a G1 joint-space motion at 30 fps, one row per frame:

```
x, y, z, qx, qy, qz, qw, joint_0 ... joint_28
```

(root position in metres, root quaternion xyzw, 29 joint angles in the G1 order of
`scripts/extract_sequences.py::CSV_JOINT_NAMES`). The Unitree-retargeted LAFAN1 clips
(e.g. `dance1_subject1.csv`, from the
[LAFAN1 retargeting dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), `g1/` folder;
the [quick start](../README.md#quick-start) downloads it to `data/lafan1/g1/`) use this layout; motion plans from HALO's motion generator can be exported to it as well.

## From HALO datagen

[`datagen/`](../../datagen/README.md) produces walk → turn → sit-on-chair episodes under a prefix of its
`HALO_S3_ROOT` (`$HALO_S3_ROOT/halo_sit_n50k` for the full run, or `$HALO_S3_ROOT/try` for datagen's small try chain): the motion stage's
`<prefix>/specs/` (`shard_NNN/episode_NNNN/` with `g1_qpos.npy` + `episode_spec.json` + `meta.json`, the tracker's
input) and the rendered dataset in `<prefix>/` itself (`data/episode_*.parquet` + `episode_specs/`, the generator's
input; also accepted here). Copy the folder to a local disk first if `HALO_S3_ROOT` is an `s3://` URI. `datagen_to_csv.py` turns them into tracker CSVs bound to the lab sofa: the tracker's `labsofa` mesh is datagen's `arm_chair` mesh, so each episode is moved by one rigid planar
transform (yaw + XY, heights kept) that puts its chair onto the sofa pose of `assets/objects/labsofa/labsofa.urdf`;
the scene floor height (`floor_z`) is removed and a 1 s seated hold is appended (`--hold_s`). Files are named
`labsofa_<tag>.csv`, the prefix that binds a clip to the sofa in training. The chair
(`arm_chair`; the paper data also rendered a second chair) is for rendering only: the motion is generated without the
chair, around the same seat target, so every episode binds to the sofa (`--chairs <name>` keeps one chair). NumPy only (`pyarrow` for parquet), no
Isaac Sim.

```bash
cd tracker
../datagen/launch/halo_s3 sync <datagen prefix>/specs data/datagen_specs   # --specs / --parquet take local folders
python data/scripts/datagen_to_csv.py --specs data/datagen_specs --accepted_only --out data/datagen_csv/labsofa
#   --accepted_only keeps the episodes the motion check accepted (meta.json final_gen_accepted)
#   or: --parquet <local copy of the datagen prefix>   (the rendered dataset: data/ + episode_specs/)
python data/scripts/batch_process.py --input-dir data/datagen_csv/labsofa \
    --output-dir data/g1_processed/robot-terrain/labsofa --headless
```

Train the Sit expert on that folder (`Robot_Terrain` task, [training/README.md](../training/README.md)); the
converter also writes `manifest.json` (source, chair asset, frames, applied yaw per clip).

## CSV → NPZ

`extract_sequences.py` replays a CSV kinematically in Isaac Lab (needs the training install, see
[training/README.md](../training/README.md)) and writes the NPZ the motion command reads: joint positions /
velocities, tracked body poses and velocities, root state, `body_names`, `joint_names` and `fps`. Flags:
`--input_fps` (default 30), `--output_fps` (default 50; training requires 50, one frame per control step) and
`--frame_range START END` (1-indexed, inclusive); `--input` / `--output` take several paths, replayed in one Isaac Sim
launch. `batch_process.py` runs it on every CSV in `--input-dir`, `--chunk_size` clips (default 50) per launch with a
per-clip time budget (`--timeout_s`), reports clips whose NPZ was not written (`--motion_names` for a subset,
`--skip-existing`) and forwards any other flag (e.g. `--headless`).

```bash
cd tracker
# every CSV in a folder -> one dataset folder of the training layout
python data/scripts/batch_process.py --input-dir <csv dir> --output-dir data/g1_processed/robot_only/lafan --headless
# one clip
python data/scripts/extract_sequences.py --input data/lafan1/g1/walk1_subject1.csv \
    --output data/g1_processed/robot_only/lafan/walk1_subject1.npz --headless
```

Push-box clips also need the box trajectory, `object_pos_w`, `object_quat_w`, `object_lin_vel_w`, `object_ang_vel_w`
(per frame, world frame), added to the NPZ; the converter does not produce it. See [training/README.md](../training/README.md#motion-data) for the folder layout and the
file-name conventions that bind clips to terrain / object assets.

## Viewer (no Isaac Sim)

```bash
pip install viser yourdfpy trimesh numpy
python data/scripts/visualize_motion_web.py --csv data/lafan1/g1/walk1_subject1.csv   # browse a CSV motion
```
