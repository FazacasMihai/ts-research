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


class CDEFunc(nn.Module):
    """Vector field f(z): hidden state -> (hidden x input channels) matrix."""

    def __init__(self, in_channels, hidden):
        super().__init__()
        self.in_channels, self.hidden = in_channels, hidden
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden * in_channels))

    def forward(self, s, z):
        # tanh keeps the vector field bounded, as in Kidger et al. (2020).
        return torch.tanh(self.net(z)).view(z.shape[0], self.hidden, self.in_channels)


class NeuralCDEClassifier(nn.Module):
    """Neural controlled differential equation (Kidger et al. 2020).

    The observations, with time as an extra channel, are joined into a
    continuous path X (linear interpolation). The hidden state follows
        dz = f(z) dX,    z(0) = W X(0),
    solved with RK4, one step per segment between two observations, and the
    class scores are read from z at the end. Because time is a channel, a long
    gap between two points is a long move of the path in the time direction,
    and the model sees it.

    With linear interpolation, on the segment from observation i to i+1 the
    path moves at a constant rate dX_i = X_{i+1} - X_i (parametrised by
    observation index), so the CDE there is the ODE dz/ds = f(z) dX_i. We
    write that RK4 step out ourselves. It is the same computation as
    torchcde.cdeint with LinearInterpolation and rk4 / step 1, except at the
    segment ends: torchcde takes the slope at an observation from the segment
    *before* it, so the solver's last stage of every step uses the wrong
    segment, and the first padded step still moves z. Here each step uses
    only its own segment: on padding (repeated last point) dX = 0, z does
    not change, and padding never alters a prediction.
    """

    def __init__(self, n_channels, n_classes, hidden=64):
        super().__init__()
        self.initial = nn.Linear(n_channels + 1, hidden)
        self.func = CDEFunc(n_channels + 1, hidden)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, t, lengths):
        path = torch.cat([t.unsqueeze(-1), x], dim=-1)  # (n, L, channels + 1)
        z = self.initial(path[:, 0])
        for dX in torch.diff(path, dim=1).unbind(dim=1):
            dX = dX.unsqueeze(-1)  # (n, channels + 1, 1)

            def g(z):
                return (self.func(None, z) @ dX).squeeze(-1)

            k1 = g(z)
            k2 = g(z + k1 / 2)
            k3 = g(z + k2 / 2)
            k4 = g(z + k3)
            z = z + (k1 + 2 * k2 + 2 * k3 + k4) / 6
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
    raise ValueError(f"unknown model {name}")


MODELS = ["GRU", "LSTM", "CfC", "NeuralCDE"]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
