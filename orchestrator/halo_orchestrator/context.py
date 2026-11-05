#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""FrameRing: the recent ego frames sent with a VLM call (only the current turn carries images)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FrameRing:
    """The newest ego frame."""
    last: Optional[tuple[float, Any]] = None   # (stamp, image)

    def push(self, stamp: float, image: Any) -> None:
        if image is not None:
            self.last = (stamp, image)

    def labelled(self, now: float) -> list[tuple[str, Any]]:
        """[(label, image)] for the newest frame, or [] if none."""
        if self.last is None:
            return []
        stamp, img = self.last
        age = max(0.0, now - stamp)
        return [("ego frame: now" if age < 0.05 else f"ego frame: age {age:.1f}s", img)]
