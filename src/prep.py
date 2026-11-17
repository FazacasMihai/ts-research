"""Turn series (with gaps) into PyTorch tensors for the neural models."""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SeriesData:
    """x: values (n, L, c), t: times (n, L), lengths: real points per series, y: labels."""
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
    """Scale each channel to mean 0, std 1, using the training data only."""

    def fit(self, values):
        allpoints = np.concatenate([np.asarray(v, dtype=float) for v in values], axis=1)
        self.mean = allpoints.mean(axis=1, keepdims=True)
        self.std = allpoints.std(axis=1, keepdims=True) + 1e-8
        return self

    def transform(self, values):
        return [(np.asarray(v, dtype=float) - self.mean) / self.std for v in values]


def to_tensors(values, times, y, time_scale):
    """Stack series into one SeriesData.

    Times are multiplied by time_scale so that one original step = 1.
    Shorter series are padded by repeating their last point.
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
    """Time since the previous point (for the first point: since t = 0)."""
    return torch.diff(t, dim=1, prepend=torch.zeros_like(t[:, :1]))
