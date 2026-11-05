#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""2D occupancy grid: collision checks and A* path planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class OccGrid:
    grid: np.ndarray  # bool, [n_y, n_x], True = occupied
    x_min: float
    y_min: float
    resolution: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape  # (n_y, n_x)

    def world_to_idx(self, xy: Sequence[float]) -> tuple[int, int]:
        ix = int((xy[0] - self.x_min) / self.resolution)
        iy = int((xy[1] - self.y_min) / self.resolution)
        return ix, iy

    def in_bounds(self, ix: int, iy: int) -> bool:
        ny, nx = self.grid.shape
        return 0 <= ix < nx and 0 <= iy < ny


def _rotated_box_cells(
    occ: OccGrid, center_xy: Sequence[float], half_xy: Sequence[float], yaw_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of grid indices `(ix, iy)` covered by the rotated AABB."""
    ny, nx = occ.shape
    cx, cy = float(center_xy[0]), float(center_xy[1])
    hx, hy = float(half_xy[0]), float(half_xy[1])
    res = occ.resolution

    diag = np.hypot(hx, hy)
    ix0 = max(0, int((cx - diag - occ.x_min) / res))
    ix1 = min(nx - 1, int((cx + diag - occ.x_min) / res) + 1)
    iy0 = max(0, int((cy - diag - occ.y_min) / res))
    iy1 = min(ny - 1, int((cy + diag - occ.y_min) / res) + 1)
    if ix0 > ix1 or iy0 > iy1:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    gi0 = np.arange(ix0, ix1 + 1)
    gi1 = np.arange(iy0, iy1 + 1)
    GX, GY = np.meshgrid(gi0, gi1)
    wx = occ.x_min + (GX + 0.5) * res - cx
    wy = occ.y_min + (GY + 0.5) * res - cy
    cosr, sinr = np.cos(yaw_rad), np.sin(yaw_rad)
    lx = cosr * wx + sinr * wy
    ly = -sinr * wx + cosr * wy
    inside = (np.abs(lx) <= hx) & (np.abs(ly) <= hy)
    return GX[inside].ravel(), GY[inside].ravel()


def is_circle_free(occ: OccGrid, center_xy: Sequence[float], radius: float) -> bool:
    """True iff no occupied cell is inside the circle of radius `radius`."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    res = occ.resolution
    ny, nx = occ.shape
    ix0 = max(0, int((cx - radius - occ.x_min) / res))
    ix1 = min(nx - 1, int((cx + radius - occ.x_min) / res) + 1)
    iy0 = max(0, int((cy - radius - occ.y_min) / res))
    iy1 = min(ny - 1, int((cy + radius - occ.y_min) / res) + 1)
    if ix0 > ix1 or iy0 > iy1:
        return False
    gi0 = np.arange(ix0, ix1 + 1)
    gi1 = np.arange(iy0, iy1 + 1)
    GX, GY = np.meshgrid(gi0, gi1)
    wx = occ.x_min + (GX + 0.5) * res - cx
    wy = occ.y_min + (GY + 0.5) * res - cy
    inside = (wx * wx + wy * wy) <= radius * radius
    return not occ.grid[GY[inside].ravel(), GX[inside].ravel()].any()


def is_segment_clear_with_bubble(
    occ: OccGrid,
    p0_xy: Sequence[float],
    p1_xy: Sequence[float],
    bubble_radius: float = 0.35,
    samples_per_meter: float = 10.0,
) -> bool:
    """True iff a disk of radius ``bubble_radius`` swept from p0 to p1 hits no occupied cell."""
    p0 = np.asarray(p0_xy, dtype=float)
    p1 = np.asarray(p1_xy, dtype=float)
    L = float(np.linalg.norm(p1 - p0))
    n_samples = max(2, int(L * samples_per_meter))
    for t in np.linspace(0.0, 1.0, n_samples):
        c = p0 + t * (p1 - p0)
        if not is_circle_free(occ, c, bubble_radius):
            return False
    return True


def dilate(occ: OccGrid, radius_m: float) -> OccGrid:
    """Copy of ``occ`` with every occupied cell grown by ``radius_m`` (square kernel)."""
    from scipy.ndimage import binary_dilation

    r_cells = max(1, int(round(radius_m / occ.resolution)))
    struct = np.ones((2 * r_cells + 1, 2 * r_cells + 1), dtype=bool)
    return OccGrid(
        grid=binary_dilation(occ.grid, structure=struct).copy(),
        x_min=occ.x_min,
        y_min=occ.y_min,
        resolution=occ.resolution,
    )


def plan_path_astar(
    occ: OccGrid, start_xy: Sequence[float], goal_xy: Sequence[float]
) -> list[tuple[float, float]] | None:
    """8-connected A* from start to goal cell centres; ``None`` if blocked or unreachable."""
    import heapq

    sx_idx, sy_idx = occ.world_to_idx(start_xy)
    gx_idx, gy_idx = occ.world_to_idx(goal_xy)
    if not (occ.in_bounds(sx_idx, sy_idx) and occ.in_bounds(gx_idx, gy_idx)):
        return None
    if occ.grid[sy_idx, sx_idx] or occ.grid[gy_idx, gx_idx]:
        return None  # start or goal sits inside an obstacle

    def h(ix: int, iy: int) -> float:
        return float(np.hypot(ix - gx_idx, iy - gy_idx))

    open_heap: list[tuple[float, float, int, int]] = [(h(sx_idx, sy_idx), 0.0, sx_idx, sy_idx)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost_so_far: dict[tuple[int, int], float] = {(sx_idx, sy_idx): 0.0}

    neighbors = [(dx, dy, 1.41421356 if dx and dy else 1.0) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]

    while open_heap:
        _, g, ix, iy = heapq.heappop(open_heap)
        if (ix, iy) == (gx_idx, gy_idx):
            cells = [(ix, iy)]
            while (ix, iy) in came_from:
                ix, iy = came_from[(ix, iy)]
                cells.append((ix, iy))
            cells.reverse()
            return [
                (float(occ.x_min + (cx + 0.5) * occ.resolution), float(occ.y_min + (cy + 0.5) * occ.resolution))
                for cx, cy in cells
            ]
        for dx, dy, w in neighbors:
            nix, niy = ix + dx, iy + dy
            if not occ.in_bounds(nix, niy) or occ.grid[niy, nix]:
                continue
            new_cost = g + w
            key = (nix, niy)
            if key not in cost_so_far or new_cost < cost_so_far[key]:
                cost_so_far[key] = new_cost
                heapq.heappush(open_heap, (new_cost + h(nix, niy), new_cost, nix, niy))
                came_from[key] = (ix, iy)

    return None  # goal unreachable


def simplify_path(path: Sequence[tuple[float, float]], n_waypoints: int = 10) -> list[tuple[float, float]]:
    """Downsample a dense A* path to ``n_waypoints`` evenly spaced points."""
    if len(path) <= n_waypoints:
        return list(path)
    indices = np.linspace(0, len(path) - 1, n_waypoints).astype(int)
    return [path[i] for i in indices]


def add_box(occ: OccGrid, center_xy: Sequence[float], half_xy: Sequence[float], yaw_rad: float = 0.0) -> None:
    """Mark a rotated AABB as occupied (in-place)."""
    ix, iy = _rotated_box_cells(occ, center_xy, half_xy, yaw_rad)
    if ix.size:
        occ.grid[iy, ix] = True
