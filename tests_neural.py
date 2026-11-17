"""Checks for src/prep.py, src/models.py and src/train.py."""
import numpy as np
import torch

from src.missing import drop_points
from src.models import MODELS, build_model, count_params
from src.prep import Normaliser, time_gaps, to_tensors
from src.train import evaluate, train_model

rng = np.random.default_rng(0)

# Short series are padded with their last point.
values = [rng.normal(size=(3, 5)), rng.normal(size=(3, 8))]
times = [np.linspace(0, 1, 5), np.linspace(0, 1, 8)]
d = to_tensors(values, times, [0, 1], time_scale=7)
assert d.x.shape == (2, 8, 3) and d.t.shape == (2, 8)
assert d.lengths.tolist() == [5, 8]
assert torch.allclose(d.x[0, 5:], d.x[0, 4].expand(3, 3))
# One original step = 1; padding adds gaps of 0.
assert torch.allclose(time_gaps(d.t)[1, 1:], torch.ones(7))
assert torch.all(time_gaps(d.t)[0, 5:] == 0)

# After normalising: mean 0, std 1 per channel.
X = rng.normal(loc=5, scale=3, size=(20, 2, 30))
norm = Normaliser().fit(list(X))
Z = np.concatenate(norm.transform(list(X)), axis=1)
assert np.allclose(Z.mean(1), 0, atol=1e-8) and np.allclose(Z.std(1), 1, atol=1e-6)

for name in MODELS:
    torch.manual_seed(0)
    model = build_model(name, n_channels=3, n_classes=2, hidden=16)
    assert count_params(model) > 0

    # Output shape.
    scores = model(d.x, d.t, d.lengths)
    assert scores.shape == (2, 2)

    # Padding does not change the result.
    alone = to_tensors(values[:1], times[:1], [0], time_scale=7)
    assert torch.allclose(model(alone.x, alone.t, alone.lengths)[0], scores[0], atol=1e-5)

    # The model can learn 10 examples by heart.
    Xs = rng.normal(size=(10, 3, 20))
    ys = np.arange(10) % 2
    v, t = drop_points(Xs, 0.3, seed=0)
    small = to_tensors(v, t, ys, time_scale=19)
    model = build_model(name, n_channels=3, n_classes=2, hidden=32)
    train_model(model, small, small, epochs=300, patience=300, lr=1e-2, batch_size=10)
    _, pred = evaluate(model, small)
    assert (pred == ys).all(), f"{name} could not fit 10 examples"

# CfC: different time gaps give a different output.
torch.manual_seed(0)
cfc = build_model("CfC", n_channels=3, n_classes=2, hidden=16)
x = torch.randn(4, 6, 3)
lengths = torch.full((4,), 6)
even = torch.arange(6.0).expand(4, 6)
uneven = torch.tensor([0.0, 1, 4, 5, 9, 10]).expand(4, 6)
assert not torch.allclose(cfc(x, even, lengths), cfc(x, uneven, lengths))

# CfC: a batch gives the same output as one series at a time.
torch.manual_seed(0)
cfc = build_model("CfC", n_channels=3, n_classes=2, hidden=16)
x = torch.randn(16, 6, 3)
t = torch.cumsum(torch.rand(16, 6) * 3, dim=1)
lengths = torch.full((16,), 6)
together = cfc(x, t, lengths)
one_by_one = torch.cat([cfc(x[i:i + 1], t[i:i + 1], lengths[i:i + 1]) for i in range(16)])
assert torch.allclose(together, one_by_one, atol=1e-5)

# Neural CDE: different time gaps give a different output.
torch.manual_seed(0)
cde = build_model("NeuralCDE", n_channels=3, n_classes=2, hidden=16)
x = torch.randn(4, 6, 3)
lengths = torch.full((4,), 6)
assert not torch.allclose(cde(x, even, lengths), cde(x, uneven, lengths))

# Neural CDE: a path that does not move leaves the state unchanged.
flat_x = x[:, :1].expand(4, 6, 3)
flat_t = torch.zeros(4, 6)
start_only = cde(x[:, :1], flat_t[:, :1], torch.ones(4, dtype=torch.long))
assert torch.allclose(cde(flat_x, flat_t, lengths), start_only, atol=1e-5)

# LTC: uses the time gaps, no NaN, and a batch = one series at a time.
torch.manual_seed(0)
ltc = build_model("LTC", n_channels=3, n_classes=2, hidden=16)
x = torch.randn(4, 6, 3)
lengths = torch.full((4,), 6)
assert not torch.allclose(ltc(x, even, lengths), ltc(x, uneven, lengths))
assert torch.isfinite(ltc(x, even, lengths)).all()
x = torch.randn(16, 6, 3)
t = torch.cumsum(torch.rand(16, 6) * 3 + 1, dim=1)
lengths = torch.full((16,), 6)
together = ltc(x, t, lengths)
one_by_one = torch.cat([ltc(x[i:i + 1], t[i:i + 1], lengths[i:i + 1]) for i in range(16)])
assert torch.allclose(together, one_by_one, atol=1e-5)

print("neural-model checks passed")
