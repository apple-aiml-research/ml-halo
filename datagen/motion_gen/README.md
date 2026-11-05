# `motion_gen/`: motion generation (episode spec, sit task, Kimodo, motion_check gate)

Stage 2 of [`../docs/data_pipeline.md`](../docs/data_pipeline.md): scene-free sit-on-chair motions for the
render to place into rooms.

## Run

From `datagen/` (setup, models and `HF_TOKEN`: [../README.md](../README.md#setup)):

```bash
launch/run_stage.sh gen --shard 0 --num-shards 250     # one shard: 200 episodes -> halo_sit_n50k/specs/shard_000/
launch/run_stage.sh gen --shard 0 --num-shards 250 EPS_PER_SHARD=4 SPEC_S3_PATH=try/specs   # a quick test
python motion_gen/generate_sit_traj.py --n 4 --out ./sit_test --device cuda   # without the launcher (in the kimodo env)
PYTHONPATH=. python -m motion_gen.motion_check.run --dataset ./sit_test --task sit_on_chair --tier industry   # re-check
```

Each episode dir holds `episode_spec.json`, `g1_qpos.npy` `[T, 36]` (30 fps), `plan.npy` and `meta.json`
(`final_gen_accepted` = passed the gate); formats in [../docs/data_formats.md](../docs/data_formats.md).

```
motion_gen/
├── generate_sit_traj.py      stage driver: sample a sit spec, walk -> turn -> sit with Kimodo, best-of-N gate
├── check_worker.py           process pool worker for the motion_check gate
├── episode_spec.py           EpisodeSpec: the per-episode contract between motion gen and render
├── tasks/sit.py              sit spec builder (chair, spawn, visual DR draw)
├── unicycle.py               accel-limited root walk along an A* corridor (common/placement.py)
├── kimodo/                   model.py, segments.py, motion_gen.py (staged Kimodo generation), constraints.py
│                             (canonicalization + constraint helpers), walk.py, keyframes/ (sit keyframes +
│                             standing anchors)
├── motion_check/             the motion-quality gate (README.md)
└── pod/                      STAGE_SETUP / STAGE_CMD of configs/pipeline/gen.env: setup_kimodo.sh,
                              run_kimodo_sit_syn.sh
```

## Invariants

- Robot: THOR (`../assets/robots/g1/urdf/g1_29dof_v2_thor.urdf`, `ROBOT_THOR_URDF` overrides).
