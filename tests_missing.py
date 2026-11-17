"""Checks for src/missing.py."""
import numpy as np

from src.features import extract, signature_features
from src.missing import drop_points, to_even_grid

rng = np.random.default_rng(42)
X = rng.normal(size=(5, 3, 20))  # 5 series, 3 channels, 20 time points

# Nothing removed: same data, even times.
v, t = drop_points(X, 0.0)
assert all(np.array_equal(a, b) for a, b in zip(v, X))
assert np.allclose(t[0], np.linspace(0, 1, 20))

# 30% removed: 14 points left, with their original values and times.
v, t = drop_points(X, 0.3, seed=0)
for series, values, times in zip(X, v, t):
    assert values.shape == (3, 14) and times.shape == (14,)
    assert np.all(np.diff(times) > 0) and times[0] >= 0 and times[-1] <= 1
    idx = np.rint(times * 19).astype(int)
    assert np.array_equal(values, series[:, idx])

# Same seed, same gaps. Other seed, other gaps.
v2, t2 = drop_points(X, 0.3, seed=0)
v3, t3 = drop_points(X, 0.3, seed=1)
assert all(np.array_equal(a, b) for a, b in zip(t, t2))
assert not all(np.array_equal(a, b) for a, b in zip(t, t3))

# Series of different lengths work too.
ragged = [rng.normal(size=(2, 10)), rng.normal(size=(2, 30))]
v, t = drop_points(ragged, 0.5, seed=0)
assert [x.shape[1] for x in v] == [5, 15]

# Interpolation gives back a straight line exactly.
line = np.linspace(0, 1, 20)[None, None, :] * np.ones((1, 2, 1))
v, t = drop_points(line, 0.5, seed=0)
back = to_even_grid(v, t, 20)[0]
inside = (np.linspace(0, 1, 20) >= t[0][0]) & (np.linspace(0, 1, 20) <= t[0][-1])
assert np.allclose(back[:, inside], line[0][:, inside])

# extract gives each series its own times.
v, t = drop_points(X, 0.5, seed=0)
F = extract(v, signature_features, times=t, depth=2)
assert F.shape == (5, 4 + 16)  # 4 channels (with time): 4 + 4^2
assert np.allclose(F[0], signature_features(v[0].T, depth=2, times=t[0]))

print("missing-data checks passed")
