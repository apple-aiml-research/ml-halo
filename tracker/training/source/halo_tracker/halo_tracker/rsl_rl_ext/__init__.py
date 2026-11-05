#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Extensions of the official ``rsl-rl-lib`` (>=5.5.1,<5.6) used by the HALO tracker.

Only what the HALO pipeline needs on top of upstream rsl_rl:

* :class:`HaloCNNModel`: ``CNNModel`` with a SimBa residual trunk, orthogonal init, and temporal ``(B, T, F)``
  observation groups encoded by a 1-D :class:`TemporalCNN` (PreRL teachers, distilled student, PostRL actor/critics).
* :class:`MultiTeacherDistillation`: on-policy distillation from N per-motion routed PreRL teachers, with a DAgger
  rollout mixture.
* :class:`MultiCriticPPO`: PostRL PPO with one shared actor and N per-motion routed critics.

Configs reference these classes by qualified name, e.g. ``"halo_tracker.rsl_rl_ext:MultiCriticPPO"``; PreRL uses the
upstream ``PPO``. Distillation runs under the upstream ``DistillationRunner``, PPO and MultiCriticPPO under the upstream
``OnPolicyRunner``.
"""

from .models import HaloCNNModel
from .modules import ResidualMLP, TemporalCNN
from .multi_critic_ppo import MultiCriticPPO
from .multi_teacher_distillation import MultiTeacherDistillation

__all__ = [
    "HaloCNNModel",
    "MultiCriticPPO",
    "MultiTeacherDistillation",
    "ResidualMLP",
    "TemporalCNN",
]
