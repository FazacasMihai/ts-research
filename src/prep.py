"""Turn (gappy) series into PyTorch tensors for the neural models.

Input is what `missing.drop_points` returns: lists `values[i]` of shape
(n_channels, k_i) and `times[i]` of shape (k_i,) in [0, 1].

All neural models get the same `SeriesData`:
    x        (n, L, c)  values, normalised per channel with train statistics
    t        (n, L)     times, scaled so that one original step = 1
    lengths  (n,)       number of real (not padded) points of each series
    y        (n,)       class index

Series shorter than L are padded by repeating their last point (same value,
same time). The models read their output at position lengths - 1, so the
padding never changes a prediction; for a Neural CDE a constant path after
the end also leaves the hidden state unchanged.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SeriesData:
    x: torch.Tensor
    t: torch.Tensor
    lengths: torch.Tensor
    y: torch.Tensor

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.long)
        return SeriesData(self.x[idx], self.t[idx], self.lengths[idx], self.y[idx])

    def to(self, device):
        return SeriesData(self.x.to(device), self.t.to(device),
                          self.lengths.to(device), self.y.to(device))


class Normaliser:
    """Per-channel mean / std, computed on the training series only."""

    def fit(self, values):
        allpoints = np.concatenate([np.asarray(v, dtype=float) for v in values], axis=1)
        self.mean = allpoints.mean(axis=1, keepdims=True)
        self.std = allpoints.std(axis=1, keepdims=True) + 1e-8
        return self

    def transform(self, values):
        return [(np.asarray(v, dtype=float) - self.mean) / self.std for v in values]


def to_tensors(values, times, y, time_scale):
    """Pad to a common length and stack into a SeriesData.

    time_scale: number of steps of the original grid (original length - 1), so
    that a gap of one original step becomes 1.0 in `t`.
    """
    lengths = np.array([v.shape[1] for v in values])
    L, c, n = lengths.max(), values[0].shape[0], len(values)
    x = np.empty((n, L, c), dtype=np.float32)
    t = np.empty((n, L), dtype=np.float32)
    for i, (v, ti) in enumerate(zip(values, times)):
        k = v.shape[1]
        x[i, :k] = v.T
        x[i, k:] = v[:, -1]
        t[i, :k] = ti * time_scale
        t[i, k:] = ti[-1] * time_scale
    return SeriesData(torch.from_numpy(x), torch.from_numpy(t),
                      torch.from_numpy(lengths), torch.as_tensor(y, dtype=torch.long))


def time_gaps(t):
    """Time since the previous point; for the first point, time since t = 0."""
    return torch.diff(t, dim=1, prepend=torch.zeros_like(t[:, :1]))
