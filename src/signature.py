"""Path signatures in plain NumPy (no extra library needed).

A time series is treated as a piecewise-linear path. The signature is built
segment by segment with Chen's identity, up to depth 3.

Shapes:
    path: (n_points, d) array, one row per time step, one column per channel.
    returns: 1D array of length d + d^2 + d^3 (for depth 3).
"""
import numpy as np


def _segment_signature(delta, depth):
    """Signature of one straight segment with increment `delta` (shape d)."""
    levels = [delta]
    if depth >= 2:
        levels.append(np.einsum("i,j->ij", delta, delta) / 2.0)
    if depth >= 3:
        levels.append(np.einsum("i,j,k->ijk", delta, delta, delta) / 6.0)
    return levels


def _chen(a, b, depth):
    """Combine signatures of two consecutive pieces (Chen's identity)."""
    out = [a[0] + b[0]]
    if depth >= 2:
        out.append(a[1] + b[1] + np.einsum("i,j->ij", a[0], b[0]))
    if depth >= 3:
        out.append(
            a[2] + b[2]
            + np.einsum("ij,k->ijk", a[1], b[0])
            + np.einsum("i,jk->ijk", a[0], b[1])
        )
    return out


def signature(path, depth=3):
    """Truncated signature of a path, flattened into one feature vector."""
    if not 1 <= depth <= 3:
        raise ValueError("depth must be 1, 2 or 3")
    path = np.asarray(path, dtype=float)
    deltas = np.diff(path, axis=0)
    d = path.shape[1]
    sig = [np.zeros(d), np.zeros((d, d)), np.zeros((d, d, d))][:depth]
    for delta in deltas:
        sig = _chen(sig, _segment_signature(delta, depth), depth)
    return np.concatenate([level.ravel() for level in sig])


def add_time(path, times=None):
    """Time augmentation: add time as an extra channel.

    `times` can be uneven (irregular sampling). Default: evenly spaced in [0, 1].
    """
    path = np.asarray(path, dtype=float)
    if times is None:
        times = np.linspace(0.0, 1.0, len(path))
    return np.column_stack([np.asarray(times, dtype=float), path])


def add_basepoint(path):
    """Start the path at zero, so the signature also sees the starting value."""
    path = np.asarray(path, dtype=float)
    return np.vstack([np.zeros((1, path.shape[1])), path])
