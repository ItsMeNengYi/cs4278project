"""Monte Carlo localisation (particle filter) with a likelihood-field beam model.

Each particle is (x, y) plus an odometry scale factor per axis. Odometry error here is mostly a
systematic scale bias, not random noise; a filter that only adds random noise stays confident while
it drifts along featureless corridors. Carrying the scale lets the filter learn the bias wherever
the walls make motion observable, and keep correcting for it where they don't.
"""
import numpy as np


class ParticleFilter:
    def __init__(self, n, mean, sigma, rng, sigma_rel=0.1, sigma_abs=0.0007,
                 hit_sigma=0.1, z_rand=0.05, beams_used=36, scale_sigma=0.05, scale_walk=2e-4):
        self.rng = rng
        self.p = np.asarray(mean, float) + rng.normal(0.0, sigma, (n, 2))
        self.scale = 1.0 + rng.normal(0.0, scale_sigma, (n, 2))   # true = odom / scale, per axis
        self.w = np.full(n, 1.0 / n)
        self.sigma_rel, self.sigma_abs = sigma_rel, sigma_abs
        self.hit_sigma, self.z_rand, self.beams_used = hit_sigma, z_rand, beams_used
        self.scale_walk = scale_walk
        self.resamples = 0

    def predict(self, odom):
        """Motion update: undo each particle's scale guess, then add noise that grows with distance."""
        s = self.sigma_rel * np.linalg.norm(odom) + self.sigma_abs
        self.p += odom / self.scale + self.rng.normal(0.0, s, self.p.shape)
        self.scale += self.rng.normal(0.0, self.scale_walk, self.scale.shape)   # keeps diversity after resampling

    def update(self, angles, ranges, field):
        """Measurement update. field(xy) -> distance to the nearest wall surface."""
        ok = np.isfinite(ranges)                    # drop dropouts and max-range (no return) beams
        if ok.sum() < 5:
            return
        idx = np.flatnonzero(ok)
        idx = idx[np.linspace(0, len(idx) - 1, min(self.beams_used, len(idx))).astype(int)]
        r = ranges[idx]
        dirs = np.column_stack([np.cos(angles[idx]), np.sin(angles[idx])])

        ends = self.p[:, None, :] + r[None, :, None] * dirs[None, :, :]       # (N, B, 2)
        d = field(ends.reshape(-1, 2)).reshape(len(self.p), len(idx))
        like = (1 - self.z_rand) * np.exp(-0.5 * (d / self.hit_sigma) ** 2) + self.z_rand
        logw = np.log(self.w) + np.log(like).sum(axis=1)
        logw -= logw.max()
        w = np.exp(logw)
        self.w = w / w.sum()

        if 1.0 / np.sum(self.w ** 2) < len(self.p) / 2:
            self._resample()

    def _resample(self):
        """Low-variance (systematic) resampling."""
        n = len(self.p)
        positions = (self.rng.random() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), positions), n - 1)
        self.p = self.p[idx]
        self.scale = self.scale[idx]
        self.w = np.full(n, 1.0 / n)
        self.resamples += 1

    def estimate(self):
        mean = self.w @ self.p
        spread = np.sqrt(self.w @ np.sum((self.p - mean) ** 2, axis=1))
        return mean, spread

    def scale_estimate(self):
        return self.w @ self.scale
