"""Quick checks for the neural-model code (src/prep.py, src/models.py, src/train.py)."""
import numpy as np
import torch

from src.missing import drop_points
from src.models import MODELS, build_model, count_params
from src.prep import Normaliser, time_gaps, to_tensors
from src.train import evaluate, train_model

rng = np.random.default_rng(0)

# Padding: series of different lengths are padded by repeating the last point.
values = [rng.normal(size=(3, 5)), rng.normal(size=(3, 8))]
times = [np.linspace(0, 1, 5), np.linspace(0, 1, 8)]
d = to_tensors(values, times, [0, 1], time_scale=7)
assert d.x.shape == (2, 8, 3) and d.t.shape == (2, 8)
assert d.lengths.tolist() == [5, 8]
assert torch.allclose(d.x[0, 5:], d.x[0, 4].expand(3, 3))
# One original step = 1.0 after scaling; padding adds zero gaps.
assert torch.allclose(time_gaps(d.t)[1, 1:], torch.ones(7))
assert torch.all(time_gaps(d.t)[0, 5:] == 0)

# Normaliser uses train statistics: train data gets mean 0, std 1 per channel.
X = rng.normal(loc=5, scale=3, size=(20, 2, 30))
norm = Normaliser().fit(list(X))
Z = np.concatenate(norm.transform(list(X)), axis=1)
assert np.allclose(Z.mean(1), 0, atol=1e-8) and np.allclose(Z.std(1), 1, atol=1e-6)

for name in MODELS:
    torch.manual_seed(0)
    model = build_model(name, n_channels=3, n_classes=2, hidden=16)
    assert count_params(model) > 0

    # Shape of the output.
    scores = model(d.x, d.t, d.lengths)
    assert scores.shape == (2, 2)

    # Padding never changes a prediction: the short series alone gives the same scores.
    alone = to_tensors(values[:1], times[:1], [0], time_scale=7)
    assert torch.allclose(model(alone.x, alone.t, alone.lengths)[0], scores[0], atol=1e-5)

    # Sanity check: the model can learn 10 random series by heart.
    Xs = rng.normal(size=(10, 3, 20))
    ys = np.arange(10) % 2
    v, t = drop_points(Xs, 0.3, seed=0)
    small = to_tensors(v, t, ys, time_scale=19)
    model = build_model(name, n_channels=3, n_classes=2, hidden=32)
    train_model(model, small, small, epochs=300, patience=300, lr=1e-2, batch_size=10)
    _, pred = evaluate(model, small)
    assert (pred == ys).all(), f"{name} could not fit 10 examples"

print("neural-model checks passed")
