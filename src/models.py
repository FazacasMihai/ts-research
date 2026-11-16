"""Neural classifiers. All share one interface:

    model(x, t, lengths) -> class scores of shape (n, n_classes)

with x, t, lengths as in `prep.SeriesData`. This way the training loop does
not need to know which model it is training.
"""
import torch
from torch import nn

from .prep import time_gaps


def last_valid(seq, lengths):
    """Output at the last real (not padded) step: seq (n, L, h) -> (n, h)."""
    idx = (lengths - 1).clamp(min=0)
    return seq[torch.arange(seq.shape[0], device=seq.device), idx]


class RNNClassifier(nn.Module):
    """Discrete-time baseline: GRU or LSTM.

    The network only sees "step 1, step 2, ...", so the time since the previous
    point is given as an extra input channel. It has to learn what that gap
    means; continuous-time models use it in their equations instead.
    """

    def __init__(self, n_channels, n_classes, hidden=64, cell="gru"):
        super().__init__()
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn(n_channels + 1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, t, lengths):
        inputs = torch.cat([x, time_gaps(t).unsqueeze(-1)], dim=-1)
        out, _ = self.rnn(inputs)
        return self.head(last_valid(out, lengths))


def build_model(name, n_channels, n_classes, hidden=64):
    if name == "GRU":
        return RNNClassifier(n_channels, n_classes, hidden, cell="gru")
    if name == "LSTM":
        return RNNClassifier(n_channels, n_classes, hidden, cell="lstm")
    raise ValueError(f"unknown model {name}")


MODELS = ["GRU", "LSTM"]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
