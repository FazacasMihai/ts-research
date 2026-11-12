"""Quick checks that the NumPy signature matches known properties."""
import numpy as np

from src.signature import signature

# Level 1 = total increment; level 2 antisymmetric part = Levy area.
path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])  # right, then up
s = signature(path, depth=2)
lvl1, lvl2 = s[:2], s[2:].reshape(2, 2)
assert np.allclose(lvl1, [1, 1])
assert np.isclose(0.5 * (lvl2[0, 1] - lvl2[1, 0]), 0.5)  # Levy area +1/2
# Same path traced at a different speed -> same signature.
fine = np.array([[0, 0], [0.5, 0], [1, 0], [1, 0.25], [1, 1]], dtype=float)
assert np.allclose(signature(path, 3), signature(fine, 3))
print("signature checks passed")
