"""Remove random time points from series, keeping the real times.

A dataset is a 3D array (n_series, n_channels, n_timepoints) or a list of
(n_channels, n_timepoints) arrays. Times are scaled to [0, 1].
"""
import numpy as np


def drop_points(X, frac, seed=0):
    """Remove a fraction `frac` of the points of every series at random.

    The same seed always removes the same points.
    Returns two lists: values[i] (n_channels, k_i) and times[i] (k_i,).
    """
    if not 0.0 <= frac < 1.0:
        raise ValueError("frac must be in [0, 1)")
    rng = np.random.default_rng(seed)
    values, times = [], []
    for series in X:
        series = np.asarray(series, dtype=float)
        T = series.shape[1]
        k = max(2, T - int(round(frac * T)))
        keep = np.sort(rng.choice(T, size=k, replace=False))
        values.append(series[:, keep])
        times.append(keep / (T - 1))
    return values, times


def to_even_grid(values, times, n_points):
    """Interpolate each series back onto n_points equally spaced times in [0, 1]."""
    if np.isscalar(n_points):
        n_points = [n_points] * len(values)
    out = []
    for v, t, n in zip(values, times, n_points):
        grid = np.linspace(0.0, 1.0, int(n))
        out.append(np.stack([np.interp(grid, t, channel) for channel in v]))
    return out


def series_lengths(X):
    """Number of points in each series."""
    return [np.asarray(series).shape[1] for series in X]
