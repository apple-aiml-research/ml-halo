#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Enable `python -m halo_comm <cmd>` (mirrors the `halo_comm` console script)."""
from .cli import main

raise SystemExit(main())
