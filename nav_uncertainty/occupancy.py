"""Grid geometry, log-odds occupancy mapping, the planner's cost grid and the likelihood field.

Arrays are indexed [ix, iy] (x first). Plot them with imshow(arr.T, origin="lower").
"""
import math

import numpy as np

INSCRIBED = 200.0       # cost of cells where the footprint would overlap a wall: allowed only to escape


class GridSpec:
    def __init__(self, x_min, y_min, x_max, y_max, res):
        self.x_min, self.y_min, self.res = float(x_min), float(y_min), float(res)
        self.nx = int(round((x_max - x_min) / res))
        self.ny = int(round((y_max - y_min) / res))

    @property
    def shape(self):
        return (self.nx, self.ny)

    def to_idx(self, xy):
        xy = np.asarray(xy, float)
        ix = np.floor((xy[..., 0] - self.x_min) / self.res).astype(int)
        iy = np.floor((xy[..., 1] - self.y_min) / self.res).astype(int)
        return ix, iy

    def inside(self, ix, iy):
        return (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)

    def to_xy(self, ix, iy):
        return np.stack([self.x_min + (np.asarray(ix) + 0.5) * self.res,
                         self.y_min + (np.asarray(iy) + 0.5) * self.res], axis=-1)

    def centres(self):
        return (self.x_min + (np.arange(self.nx) + 0.5) * self.res,
                self.y_min + (np.arange(self.ny) + 0.5) * self.res)


def shift(a, dx, dy, fill):
    """out[i, j] = a[i - dx, j - dy], padding with `fill`."""
    out = np.full_like(a, fill)
    nx, ny = a.shape
    xs_dst, xs_src = slice(max(dx, 0), nx + min(dx, 0)), slice(max(-dx, 0), nx + min(-dx, 0))
    ys_dst, ys_src = slice(max(dy, 0), ny + min(dy, 0)), slice(max(-dy, 0), ny + min(-dy, 0))
    out[xs_dst, ys_dst] = a[xs_src, ys_src]
    return out


def distance_to_occupied(occ, res, cap):
    """Chamfer distance (m) from each cell centre to the nearest occupied cell centre, capped at `cap`."""
    d = np.where(occ, 0.0, np.inf)
    steps = [(dx, dy, math.hypot(dx, dy) * res) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    for _ in range(int(math.ceil(cap / res)) + 1):
        for dx, dy, w in steps:
            d = np.minimum(d, shift(d, dx, dy, np.inf) + w)
    return np.minimum(d, cap)


def box_dilate(occ, kx, ky):
    out = np.zeros_like(occ)
    for dx in range(-kx, kx + 1):
        for dy in range(-ky, ky + 1):
            out |= shift(occ, dx, dy, False)
    return out


def cost_grid(occ, res, half_x, half_y, soft_radius=0.8, soft_weight=5.0):
    """Planner cost per cell: inf in walls, INSCRIBED where the box footprint (yaw fixed at 0) would
    touch a wall, and a soft penalty near walls so paths keep to the middle of corridors."""
    kx = int(math.ceil((half_x + res / 2) / res)) - 1
    ky = int(math.ceil((half_y + res / 2) / res)) - 1
    d = distance_to_occupied(occ, res, soft_radius)
    cost = 1.0 + soft_weight * np.clip((soft_radius - d) / soft_radius, 0.0, 1.0) ** 2
    cost[box_dilate(occ, kx, ky)] = INSCRIBED
    cost[occ] = np.inf
    return cost


class LikelihoodField:
    """Distance from a point to the nearest wall surface, for the particle filter's beam model."""

    def __init__(self, spec, occ, cap=0.5):
        self.spec, self.cap = spec, cap
        self.d = np.maximum(distance_to_occupied(occ, spec.res, cap) - spec.res / 2, 0.0)

    def __call__(self, xy):
        ix, iy = self.spec.to_idx(xy)
        ok = self.spec.inside(ix, iy)
        out = np.full(ix.shape, self.cap)
        out[ok] = self.d[ix[ok], iy[ok]]
        return out


class OccupancyMap:
    def __init__(self, spec, l_occ=0.85, l_free=-0.4, l_min=-2.0, l_max=3.5, occ_thresh=0.5):
        self.spec = spec
        self.L = np.zeros(spec.shape, dtype=np.float32)
        self.l_occ, self.l_free, self.l_min, self.l_max = l_occ, l_free, l_min, l_max
        self.occ_thresh = occ_thresh

    @classmethod
    def from_truth(cls, spec, occ):
        m = cls(spec)
        m.L[:] = np.where(occ, m.l_max, m.l_min)
        return m

    def occupied(self):
        return self.L > self.occ_thresh

    def known(self):
        return self.L != 0

    def update(self, pose_xy, angles, ranges, max_range):
        """Integrate one scan taken at pose_xy. Free space up to each return, occupied just past it."""
        spec, res = self.spec, self.spec.res
        valid = ~np.isnan(ranges)
        if not valid.any():
            return
        r, a = ranges[valid], angles[valid]
        dirs = np.column_stack([np.cos(a), np.sin(a)])
        hit = np.isfinite(r)
        free_len = np.where(hit, r - res / 2, max_range)

        s = np.arange(0.0, free_len.max(), res / 2)
        pts = pose_xy + s[None, :, None] * dirs[:, None, :]                 # (B, S, 2)
        ix, iy = spec.to_idx(pts[s[None, :] < free_len[:, None]])
        ok = spec.inside(ix, iy)
        free = np.unique(ix[ok] * spec.ny + iy[ok])

        # nudge the endpoint half a cell into the wall so the cell holding the surface ends up occupied
        ends = pose_xy + (r[hit] + res / 2)[:, None] * dirs[hit]
        hx, hy = spec.to_idx(ends)
        ok = spec.inside(hx, hy)
        occ = np.unique(hx[ok] * spec.ny + hy[ok])
        free = np.setdiff1d(free, occ, assume_unique=True)

        flat = self.L.reshape(-1)
        flat[free] += self.l_free
        flat[occ] += self.l_occ
        np.clip(self.L, self.l_min, self.l_max, out=self.L)
