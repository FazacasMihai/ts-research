"""Neural classifiers.

Every model is called as model(x, t, lengths) and returns class scores
of shape (n, n_classes).
"""
import torch
from ncps.torch import CfCCell, LTCCell
from ncps.wirings import FullyConnected
from torch import nn

from .prep import time_gaps


def last_valid(seq, lengths):
    """Pick the output at the last real (not padded) step."""
    idx = (lengths - 1).clamp(min=0)
    return seq[torch.arange(seq.shape[0], device=seq.device), idx]


class RNNClassifier(nn.Module):
    """GRU or LSTM. The time gap is given as an extra input channel."""

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
    """CfC (Hasani et al. 2022). The time gap goes into the cell's equations.

    We loop over the ncps cell ourselves: the ncps CfC wrapper mixes up the
    shape of the time gaps when a batch has more than one series.
    backbone_units = hidden keeps the size close to GRU / LSTM.
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


class LTCClassifier(nn.Module):
    """LTC (Hasani et al. 2021). Each step runs a small ODE for the time gap.

    Same cell loop as CfC, for the same reason. A gap of 0 gives NaN in the
    cell, so gaps are at least min_gap = 1 (this only changes the first point).
    Padded steps keep the old state.
    """

    def __init__(self, n_channels, n_classes, hidden=64, ode_unfolds=6, min_gap=1.0):
        super().__init__()
        self.hidden, self.min_gap = hidden, min_gap
        self.cell = LTCCell(FullyConnected(hidden), n_channels, ode_unfolds=ode_unfolds,
                            implicit_param_constraints=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, t, lengths):
        gaps = time_gaps(t).clamp(min=self.min_gap)
        h = x.new_zeros(x.shape[0], self.hidden)
        for step in range(x.shape[1]):
            _, new_h = self.cell(x[:, step], h, gaps[:, step:step + 1])
            real = (step < lengths).unsqueeze(-1)
            h = torch.where(real, new_h, h)
        return self.head(h)


class CDEFunc(nn.Module):
    """f(z) in dz = f(z) dX: a small network giving a (hidden x channels) matrix."""

    def __init__(self, in_channels, hidden):
        super().__init__()
        self.in_channels, self.hidden = in_channels, hidden
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden * in_channels))

    def forward(self, s, z):
        return torch.tanh(self.net(z)).view(z.shape[0], self.hidden, self.in_channels)


class NeuralCDEClassifier(nn.Module):
    """Neural CDE (Kidger et al. 2020): dz = f(z) dX.

    X joins the points (with time as an extra channel) by straight lines.
    Each segment gets ceil(gap / max_step) RK4 steps, so a long gap (missing
    points) gets as many solver steps as the points it replaces.
    max_step=None gives one step per segment. We don't use torchcde.cdeint,
    because at a data point it takes the slope of the previous segment,
    so padding would change the result.
    """

    def __init__(self, n_channels, n_classes, hidden=64, max_step=1.0):
        super().__init__()
        self.max_step = max_step
        self.initial = nn.Linear(n_channels + 1, hidden)
        self.func = CDEFunc(n_channels + 1, hidden)
        self.head = nn.Linear(hidden, n_classes)

    def rk4(self, z, dX):
        """One RK4 step along a straight piece with increment dX."""
        dX = dX.unsqueeze(-1)

        def g(z):
            return (self.func(None, z) @ dX).squeeze(-1)

        k1 = g(z)
        k2 = g(z + k1 / 2)
        k3 = g(z + k2 / 2)
        k4 = g(z + k3)
        return z + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def forward(self, x, t, lengths):
        path = torch.cat([t.unsqueeze(-1), x], dim=-1)
        z = self.initial(path[:, 0])
        deltas = torch.diff(path, dim=1)
        if self.max_step is None:
            n_steps = torch.ones_like(t[:, 1:])
        else:
            n_steps = torch.ceil(time_gaps(t)[:, 1:] / self.max_step - 1e-6).clamp(min=1)
        for dX, n in zip(deltas.unbind(dim=1), n_steps.unbind(dim=1)):
            # Every series takes its own number of steps; finished ones wait.
            piece = dX / n.unsqueeze(-1)
            for k in range(int(n.max())):
                active = (k < n).unsqueeze(-1)
                z = torch.where(active, self.rk4(z, piece), z)
        return self.head(z)


def build_model(name, n_channels, n_classes, hidden=64):
    if name == "GRU":
        return RNNClassifier(n_channels, n_classes, hidden, cell="gru")
    if name == "LSTM":
        return RNNClassifier(n_channels, n_classes, hidden, cell="lstm")
    if name == "CfC":
        return CfCClassifier(n_channels, n_classes, hidden)
    if name == "NeuralCDE":
        return NeuralCDEClassifier(n_channels, n_classes, hidden)
    if name == "LTC":
        return LTCClassifier(n_channels, n_classes, hidden)
    raise ValueError(f"unknown model {name}")


MODELS = ["GRU", "LSTM", "CfC", "NeuralCDE", "LTC"]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
