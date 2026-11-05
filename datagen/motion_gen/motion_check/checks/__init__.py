#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Check registry package."""

import importlib
import importlib.util

from motion_gen.motion_check.checks import base  # noqa: F401  (defines REGISTRY / run_checks)

for _mod in ("dynamics", "foot", "limits", "collision", "task_success"):
    _name = f"motion_gen.motion_check.checks.{_mod}"
    if importlib.util.find_spec(_name) is not None:
        importlib.import_module(_name)
