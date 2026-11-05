# G1 deployment scripts

Run from this folder. Install, configs, MuJoCo sim-to-sim, the real-robot procedure and safety guards are in
[`../README.md`](../README.md).

```bash
python deploy_mujoco.py --config configs/g1_v2.yaml --policy_path ../models/g1/policy.onnx --motion <motion.csv>
python deploy_real.py   --config configs/g1_v2.yaml --policy_path ../models/g1/policy.onnx --motion <motion.csv> --net eth0
```

| File | What |
|---|---|
| `deploy_mujoco.py` | MuJoCo sim-to-sim (`--headless` without a display; `MUJOCO_GL=egl` for the depth student on headless Linux) |
| `deploy_real.py`, `play_real_web_interface.py` | Real G1 over Ethernet (`unitree_sdk2py`), the latter with a web motion picker |
| `g1_controller.py` | Config-driven observation, action decoding, reference swaps |
| `depth_sources.py`, `variant_support.py` | MuJoCo / RealSense depth, depth camera rig, preflight |
| `configs/g1_v2.yaml`, `configs/g1_v2_depth.yaml` | Proprio and depth student configs |
