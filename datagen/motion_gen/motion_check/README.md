# motion_check: motion quality checker

Scores a Kimodo-generated G1 motion with per-frame geometric checks (per check
`{value, flag, score, time, comment}`) and returns accept / reject plus regeneration feedback. It is the gate of
the motion stage ([data_pipeline.md](../../docs/data_pipeline.md#step-2-motion-generation)).

```bash
# env: mujoco (3.14 in the motion-gen env), numpy, scipy, trimesh, pyyaml
# (without pyyaml the thresholds silently fall back to the code defaults)
PYTHONPATH=. python -m motion_gen.motion_check.run --dataset DIR --task sit_on_chair     # DIR of <tag>_qpos.npy
PYTHONPATH=. python -m motion_gen.motion_check.run --episode DIR --tag ep0 --task sit_on_chair --tier industry --trace
```

Task `sit_on_chair` (= walk_and_sit). Tier `industry` is the gate (`configs/motion_check/tier_industry.yaml`, the
only tier file shipped; another `--tier` name loads no file); no tier → the task default. Judge CUDA-generated
motion only (Kimodo on MPS diverges ~8 cm).

Inputs: raw qpos `<dir>/<tag>_qpos.npy` `[T,36]` (root xyz + quat wxyz + 29 joints, 30 fps, optional
`<tag>_meta.json`), a motion-stage episode dir, or training-dataset parquet (qpos = `observation.body_pos[:,0]` +
`observation.body_quat[:,0]` + `observation.joint_pos`, no joint remapping).

## Architecture

`kinematics.build_tracks` is the only MuJoCo module: one FK sweep gives numpy `Tracks` (foot sole z vs ray-cast
floor, self / scene min distance via `mj_geomDistance`, body / foot / palm positions); every check in
`checks/*.py` (`@register`, `check(ctx) -> CheckResult`) is pure numpy over them. `evaluator.py`
(`MotionQualityEvaluator.for_task(task, tier=).evaluate(ep)`) runs them, `judge.py` (`RuleJudge`) rejects on any
failing hard check (soft checks lower the score), `config.py` `CheckConfig` holds all thresholds (overlay
`default.yaml` → `sit.yaml` (sit tasks) → `<task>.yaml` → `tier_<tier>.yaml`), `version.py` the `CHECKER_VERSION`,
`run.py` the CLI.

Checks: dynamics `teleport`(H), `excessive_speed`, `sudden_jump`(H), `small_hop`, `crab_walk`,
`upper_body_speed`, `turn_rate`, `vertical_drift`, `back_and_forth`, `jitter`; foot `foot_sink`(H),
`foot_float`, `stance_width`, `step_clearance`, `flight_phase`, `feet_close`, `foot_skate`(H),
`foot_skate_severe`(H); collision `self_collision`(H), `scene_collision`(H). (H) = hard by default;
`cfg.severity_overrides` (`{name: soft|hard}` in the task yaml) changes it. The walk_and_sit calibration is
`configs/motion_check/sit_on_chair.yaml` + `tier_*.yaml`.

## Output and reproducibility

`MotionQualityReport.to_dict()`: the compact `{check: [value, flag, score, time, comment]}` map, full per-check
dicts and `verdict{accept, score, blocking, feedback}`. Every report carries
`checker_id = mqc-<VERSION>+cfg.<hash>+chk.<hash>`; two reports are comparable if and only if their ids match.
Bump `CHECKER_VERSION` on logic / threshold changes; record the id with results.

New task: thresholds do not transfer. Label good and bad episodes (defect + frame window), plot candidate
signals over time (aggregates hide shuffles, hops, jerks), set each threshold to fire on no good motion in
`configs/motion_check/<task>.yaml` (strictness in `tier_*.yaml`), bump `CHECKER_VERSION`.

## Pitfalls

- Floor: use the flat z=0 robot-only floor; feet sit 2-3 cm below it (Kimodo root offset), and loading a scan
  scene breaks every foot check.
- Scene / chair collision is not checked: a non-convex scan collides in MuJoCo as one convex hull.
