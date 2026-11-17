"""Feature extractors: each turns one series (n_points, d) into a vector."""
import numpy as np

from .signature import add_basepoint, add_time, signature


def summary_stats(series):
    """Mean, std, min, max and total change of each channel."""
    s = np.asarray(series, dtype=float)
    return np.concatenate([s.mean(0), s.std(0), s.min(0), s.max(0), s[-1] - s[0]])


def signature_features(series, depth=3, times=None):
    """Signature of the series, with time added as a channel."""
    return signature(add_basepoint(add_time(series, times)), depth=depth)


def dmd_features(series, rank=4, delays=5):
    """DMD features: frequency and growth rate of the main modes.

    Needs equally spaced points.
    """
    s = np.asarray(series, dtype=float)
    n, d = s.shape
    H = np.vstack([s[i : n - delays + i + 1].T for i in range(delays)])
    X, Y = H[:, :-1], H[:, 1:]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(rank, int((S > 1e-10 * S[0]).sum()))
    U, S, Vt = U[:, :r], S[:r], Vt[:r]
    A_tilde = U.T @ Y @ Vt.T / S
    eig = np.linalg.eigvals(A_tilde)
    eig = eig[np.argsort(-np.abs(eig))]
    log_eig = np.log(eig.astype(complex) + 1e-12)
    feats = np.column_stack([log_eig.real, np.abs(log_eig.imag)]).ravel()
    out = np.zeros(2 * rank)
    out[: feats.size] = feats
    return out


def extract(X, fn, times=None, **kwargs):
    """Apply a feature function to every series.

    X: series of shape (n_channels, n_timepoints). If `times` is given, each
    series gets its own times (only for signature_features).
    """
    if times is None:
        return np.stack([fn(np.asarray(x).T, **kwargs) for x in X])
    return np.stack([fn(np.asarray(x).T, times=t, **kwargs) for x, t in zip(X, times)])
