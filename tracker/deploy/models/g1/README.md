# G1 policy drop-in folder

No policy weights ship with this repository. Put exported policies here (`*.onnx`, `*.plan` are git-ignored):

- `policy.onnx`: `training/scripts/rsl_rl/play.py --export_only` writes `<checkpoint dir>/exported/policy.onnx`; the
  Python deploy scripts default to `policy.onnx` here with `configs/g1_v2.yaml` (proprio student). Use
  `configs/g1_v2_depth.yaml` for the depth student (inputs `obs`, `policy_temporal`, `student_depth`).
- `*.plan`: TensorRT engines built on the target Jetson by `deploy_cpp/tools/build_onnx_to_trt.sh`.
- `motiontrack_all_data.onnx` / `.plan`: the name the `system/` real-robot setup script looks for; copy or symlink
  your exported proprio policy to it.
