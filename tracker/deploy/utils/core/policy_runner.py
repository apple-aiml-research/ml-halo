#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
ONNX Policy Runner.

Handles neural network policy loading and inference for robot control.
"""

import os
import numpy as np
import onnxruntime as ort


def create_onnx_session(model_path: str, use_gpu: bool = True) -> ort.InferenceSession:
    """Create an optimized ONNX Runtime inference session.

    Uses the CUDA execution provider when available and ``use_gpu`` is set;
    ``HALO_POLICY_CPU=1`` forces the CPU provider (e.g. to compare CPU and GPU results).
    """
    if os.environ.get("HALO_POLICY_CPU", "0") == "1":
        use_gpu = False
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_cpu_mem_arena = True
    session_options.enable_mem_pattern = True
    # Disable intra-op thread spin-waiting: on a contended CPU (control loop and other
    # threads share cores), spinning between 50 Hz steps burns a core and adds jitter.
    # Harmless no-op on ORT builds that don't recognize the key.
    try:
        session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    except Exception:
        pass

    if use_gpu:
        try:
            available = ort.get_available_providers()
            providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                        if 'CUDAExecutionProvider' in available
                        else ['CPUExecutionProvider'])
        except Exception:
            providers = ['CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    return ort.InferenceSession(model_path, sess_options=session_options, providers=providers)


class PolicyRunner:
    """ONNX policy runner that feeds every named input (optional GPU acceleration).

    The tracker student policies export ONNX with named inputs:
      * ``obs              (1, 434)``     — flat actor input.
      * ``policy_temporal  (1, 30, 166)`` — temporal CNN branch.
      * ``student_depth    (1, 1, 48, 84)`` — depth student only.

    ``infer`` takes a ``{name: ndarray}`` mapping and forwards every input.
    """

    def __init__(
        self,
        policy_path: str,
        clip_action: float = 100.0,
        clip_observation: float = 100.0,
        use_gpu: bool = True,
    ):
        """Initialize policy runner.

        Args:
            policy_path: Path to ONNX model file
            clip_action: Action clipping bound
            clip_observation: Observation clipping bound
            use_gpu: Whether to try GPU acceleration
        """
        self.policy_path = policy_path
        self.clip_action = clip_action
        self.clip_observation = clip_observation

        self.session = create_onnx_session(self.policy_path, use_gpu)
        print(f"[PolicyRunner] Using provider: {self.session.get_providers()[0]}")

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.input_shapes: dict[str, tuple] = {
            inp.name: tuple(inp.shape) for inp in self.session.get_inputs()
        }
        print(f"[PolicyRunner] Loaded policy from {policy_path}")
        for name, shape in self.input_shapes.items():
            print(f"[PolicyRunner]   in : {name:20s}  shape={shape}")
        for out in self.session.get_outputs():
            print(f"[PolicyRunner]   out: {out.name:20s}  shape={tuple(out.shape)}")

    def infer(self, feeds) -> np.ndarray:
        """Run one forward pass with a dict of named inputs.

        Args:
            feeds: ``{input_name: tensor}``. Every name in ``self.input_names``
                must be present. Tensors are clipped and cast to float32.

        Returns:
            Action vector clipped to ``[-clip_action, +clip_action]``.
        """
        missing = [n for n in self.input_names if n not in feeds]
        if missing:
            raise KeyError(
                f"PolicyRunner.infer: feeds missing required inputs {missing}; "
                f"got {sorted(feeds.keys())}, expected {self.input_names}"
            )

        prepared = {}
        for name in self.input_names:
            arr = np.asarray(feeds[name], dtype=np.float32)
            arr = np.clip(arr, -self.clip_observation, self.clip_observation)
            prepared[name] = arr

        outputs = self.session.run(self.output_names, prepared)
        actions = np.squeeze(outputs[0])
        return np.clip(actions, -self.clip_action, self.clip_action)
