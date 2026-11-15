"""Feature extractors: each turns one series (n_points, d) into a vector."""
import numpy as np

from .signature import add_basepoint, add_time, signature


def summary_stats(series):
    """Simple baseline: mean, std, min, max and total change per channel."""
    s = np.asarray(series, dtype=float)
    return np.concatenate([s.mean(0), s.std(0), s.min(0), s.max(0), s[-1] - s[0]])


def signature_features(series, depth=3, times=None):
    """Signature of the time-augmented, base-pointed path."""
    return signature(add_basepoint(add_time(series, times)), depth=depth)


def dmd_features(series, rank=4, delays=5):
    """Koopman/DMD features: rhythms (frequencies) and trends (growth rates).

    The series is stacked with delayed copies of itself (a Hankel matrix),
    then exact DMD finds the best linear rule x_{k+1} ~ A x_k. The eigenvalues
    of A give, for each mode, a frequency and a growth rate.
    Assumes evenly spaced samples: interpolate irregular data first.
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

    X: aeon format, a 3D array (n_cases, n_channels, n_timepoints) or a list
       of (n_channels, n_timepoints_i) arrays when lengths differ.
    times: optional, one array of timestamps per series (e.g. from
       `missing.drop_points`). Passed to `fn` as `times=`; only use it with
       functions that accept it (signature_features).
    """
    if times is None:
        return np.stack([fn(np.asarray(x).T, **kwargs) for x in X])
    return np.stack([fn(np.asarray(x).T, times=t, **kwargs) for x, t in zip(X, times)])
