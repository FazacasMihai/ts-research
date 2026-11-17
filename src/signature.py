"""Path signatures in NumPy, up to depth 3.

The series is treated as a path of straight segments. Segments are joined
with Chen's identity. Input: (n_points, d). Output: d + d^2 + d^3 numbers.
"""
import numpy as np


def _segment_signature(delta, depth):
    """Signature of one straight segment."""
    levels = [delta]
    if depth >= 2:
        levels.append(np.einsum("i,j->ij", delta, delta) / 2.0)
    if depth >= 3:
        levels.append(np.einsum("i,j,k->ijk", delta, delta, delta) / 6.0)
    return levels


def _chen(a, b, depth):
    """Join the signatures of two pieces that follow each other."""
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
    """Signature of a path, as one flat vector."""
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
    """Add time as the first channel. Times may be uneven; default: even in [0, 1]."""
    path = np.asarray(path, dtype=float)
    if times is None:
        times = np.linspace(0.0, 1.0, len(path))
    return np.column_stack([np.asarray(times, dtype=float), path])


def add_basepoint(path):
    """Start the path at zero, so the signature also sees the starting value."""
    path = np.asarray(path, dtype=float)
    return np.vstack([np.zeros((1, path.shape[1])), path])
