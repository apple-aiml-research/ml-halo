#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""
History Buffer for Observation History.

Efficient circular buffer implementation for storing observation history.
"""

import numpy as np


class HistoryBuffer:
    """Optimized circular buffer for storing observation history.

    Matches IsaacLab's CircularBuffer behavior:
    - On first add, fills all history slots with the current observation
    - Subsequently updates with new observations in FIFO order

    Performance optimizations:
    - Uses pre-allocated numpy array (faster, contiguous memory)
    - get_history() returns a copy in chronological order via np.concatenate
    - Minimal copying operations
    """

    def __init__(self, max_len: int, feature_dim: int):
        """Initialize history buffer.

        Args:
            max_len: Maximum history length (number of timesteps)
            feature_dim: Dimension of each observation
        """
        self.feature_dim = feature_dim
        self.max_len = max_len
        self.head = 0  # Index for next write position
        self.num_pushes = 0  # Track number of pushes to handle initialization

        # Pre-allocate buffer as single contiguous numpy array
        self.buffer = np.zeros((max_len, feature_dim), dtype=np.float32)

    def add(self, observation: np.ndarray):
        """Add new observation to buffer.

        On first add, fills all history slots with the current observation
        to match IsaacLab's CircularBuffer behavior.

        Args:
            observation: Observation vector of shape (feature_dim,)
        """
        # On first push, fill all slots with current observation
        if self.num_pushes == 0:
            self.buffer[:] = observation  # Broadcast to all rows
            self.head = 0
        else:
            # Normal FIFO behavior: write to current position and advance
            self.buffer[self.head] = observation
            self.head = (self.head + 1) % self.max_len

        self.num_pushes += 1

    def get_history(self) -> np.ndarray:
        """Get history as array with shape (max_len, feature_dim).

        Returns data in chronological order (oldest to newest).

        Returns:
            History array of shape (max_len, feature_dim)
        """
        return np.concatenate([self.buffer[self.head:], self.buffer[:self.head]], axis=0)

    def get_latest(self) -> np.ndarray:
        """Get the most recently added observation.

        Returns:
            Latest observation of shape (feature_dim,)
        """
        return self.buffer[(self.head - 1) % self.max_len]
