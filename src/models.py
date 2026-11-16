"""Neural classifiers. All share one interface:

    model(x, t, lengths) -> class scores of shape (n, n_classes)

with x, t, lengths as in `prep.SeriesData`. This way the training loop does
not need to know which model it is training.
"""
import torch
from ncps.torch import CfCCell
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


class CfCClassifier(nn.Module):
    """Closed-form continuous-time network (CfC, Hasani et al. 2022).

    Gets only the values; the time since the previous point goes into the
    cell's equations (`ts`), which interpolate between two states depending on
    how much time has passed: h = (1 - s) * f1 + s * f2 with
    s = sigmoid(a(x, h) * ts + b(x, h)).

    We loop over the ncps *cell* ourselves instead of using the ncps `CfC`
    wrapper: the wrapper squeezes the time gaps to shape (batch,), and the cell
    then multiplies them with a (batch, hidden) tensor, which lines the gaps up
    with hidden units instead of series (it fails, or is silently wrong when
    batch size == hidden size). Here the gaps have shape (batch, 1).

    backbone_units = hidden by default (the library uses 128), to keep the
    parameter count close to the GRU / LSTM baselines.
    """

    def __init__(self, n_channels, n_classes, hidden=64, backbone_units=None):
        super().__init__()
        self.hidden = hidden
        self.cell = CfCCell(n_channels, hidden, mode="default",
                            backbone_units=backbone_units or hidden, backbone_layers=1)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, t, lengths):
        gaps = time_gaps(t)
        h = x.new_zeros(x.shape[0], self.hidden)
        states = []
        for step in range(x.shape[1]):
            h, _ = self.cell(x[:, step], h, gaps[:, step:step + 1])
            states.append(h)
        return self.head(last_valid(torch.stack(states, dim=1), lengths))


def build_model(name, n_channels, n_classes, hidden=64):
    if name == "GRU":
        return RNNClassifier(n_channels, n_classes, hidden, cell="gru")
    if name == "LSTM":
        return RNNClassifier(n_channels, n_classes, hidden, cell="lstm")
    if name == "CfC":
        return CfCClassifier(n_channels, n_classes, hidden)
    raise ValueError(f"unknown model {name}")


MODELS = ["GRU", "LSTM", "CfC"]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
