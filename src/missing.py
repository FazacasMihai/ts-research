"""Experiment 2 helper: remove random time points but keep the real timestamps.

A dataset is a collection of series, each of shape (n_channels, n_timepoints)
(aeon format). It can be a 3D array (all series the same length) or a list of
2D arrays (series of different lengths).

`drop_points` returns, for every series, the values that were kept and their
times. Time is the original position of the point scaled to [0, 1], so a
series with no points removed has evenly spaced times 0, 1/(T-1), ..., 1.

How each model uses the result:
    signatures, Neural CDE -> values + times directly (uneven steps are fine)
    DMD                    -> `to_even_grid` first (it assumes equal steps)
    LSTM / GRU             -> values + a "time since previous point" channel
"""
import numpy as np


def drop_points(X, frac, seed=0):
    """Remove a fraction `frac` of the time points of every series at random.

    Each series keeps k = T - round(frac * T) points (at least 2), chosen
    independently per series. With the same `seed` the same points are removed,
    so every model sees exactly the same gaps.

    Returns (values, times): two lists with one entry per series,
    values[i] of shape (n_channels, k_i) and times[i] of shape (k_i,).
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
    """Linear interpolation of each series back onto `n_points` evenly spaced times.

    The grid covers [0, 1], the time span of the original series, so with
    n_points = original length the step size is the same as before removal
    (this keeps DMD frequencies comparable across missing rates).
    `n_points` can be one number for all series or a list with one per series.
    Before the first / after the last kept point the value is held constant.
    """
    if np.isscalar(n_points):
        n_points = [n_points] * len(values)
    out = []
    for v, t, n in zip(values, times, n_points):
        grid = np.linspace(0.0, 1.0, int(n))
        out.append(np.stack([np.interp(grid, t, channel) for channel in v]))
    return out


def series_lengths(X):
    """Number of time points of every series (works for arrays and lists)."""
    return [np.asarray(series).shape[1] for series in X]
