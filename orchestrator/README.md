# orchestrator

`halo_orchestrator` turns the operator's language request and the robot's egocentric image into a plan of subtasks
(`go_to_box`, `push_box`, `sit`, ...), then decides for each running step: done, retry, insert a step, or ask
the operator. `run_sim.py` and `run_real.py` start it for you (node: `system/halo_system/nodes/orchestrator_zmq.py`).

## Setup

`system/cpp/scripts/setup_dev_env.sh` installs it with everything it needs ([system/README.md](../system/README.md#setup)).
By hand, from the repository root: `pip install -e orchestrator -e "system/serve/python[mlx]"` (it calls the model
through `halo_serve`).

## Model

[`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) (~20 GB), a local
MLX model served by `halo_serve` on Apple silicon with at least 32 GB of RAM, sharing the GPU with the generator;
`setup_dev_env.sh` downloads it into the Hugging Face cache. The subtasks are listed in `halo_orchestrator/skills.yaml`.

## Run

The `multi_task/` task configs turn it on: [system/README.md](../system/README.md#run-sim) (`run_sim.py`) and
[Run real](../system/README.md#run-real) (`run_real.py --orchestrator`).
