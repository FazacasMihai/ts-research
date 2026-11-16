"""One training loop for every neural model (same budget for all).

Adam, cross-entropy, mini-batches. After every epoch the loss on a
validation set (taken from the train split, never the test split) is checked;
training stops when it has not improved for `patience` epochs, and the
weights from the best epoch are kept.
"""
import copy
import time

import numpy as np
import torch
from torch import nn


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate(model, data, batch_size=256):
    """Mean cross-entropy and predicted classes on a SeriesData."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total, preds = 0.0, []
    for start in range(0, len(data), batch_size):
        b = data.subset(range(start, min(start + batch_size, len(data))))
        scores = model(b.x, b.t, b.lengths)
        total += loss_fn(scores, b.y).item()
        preds.append(scores.argmax(dim=1))
    return total / len(data), torch.cat(preds).cpu().numpy()


def train_model(model, train, val, epochs=500, patience=50, lr=1e-3,
                batch_size=32, seed=0, verbose=False):
    """Train in place; returns a dict with epochs run, best val loss, seconds."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)

    best_loss, best_state, best_epoch = float("inf"), None, 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=gen)
        for i in range(0, len(train), batch_size):
            b = train.subset(order[i:i + batch_size])
            opt.zero_grad()
            loss = loss_fn(model(b.x, b.t, b.lengths), b.y)
            loss.backward()
            opt.step()

        val_loss, _ = evaluate(model, val)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"    epoch {epoch:3d}  val loss {val_loss:.4f}")
        if epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return {"epochs": epoch, "best_epoch": best_epoch, "val_loss": best_loss,
            "seconds": time.perf_counter() - start}
