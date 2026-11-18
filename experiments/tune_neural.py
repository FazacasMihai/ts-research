"""Equal hyperparameter search for the neural models, on validation only.

Every model gets the same small grid (learning rate x hidden size). Runs on
clean data (0% missing); the test split is never used. The setting with the
lowest mean validation loss is saved per dataset and model, and exp03 can
load it with --params. Finished runs are skipped, so a stopped run can be
started again.

Run from the project root:

    python -m experiments.tune_neural --datasets BasicMotions --models GRU
    python -m experiments.tune_neural            # full search (use a GPU / Colab)
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from experiments.exp03_neural import DATASETS, prepare
from src.data import load_uea
from src.models import MODELS, build_model, count_params
from src.train import evaluate, set_seed, train_model

LRS = [1e-3, 3e-3, 1e-2]
HIDDENS = [32, 64]
FIELDS = ["dataset", "method", "lr", "hidden", "seed", "n_params", "val_loss",
          "val_accuracy", "seconds", "epochs", "best_epoch", "device"]


def best_settings(df):
    """Lowest mean validation loss per dataset and model."""
    mean = df.groupby(["dataset", "method", "lr", "hidden"], as_index=False)["val_loss"].mean()
    best = mean.loc[mean.groupby(["dataset", "method"])["val_loss"].idxmin()]
    out = {}
    for row in best.itertuples():
        out.setdefault(row.dataset, {})[row.method] = {"lr": float(row.lr),
                                                       "hidden": int(row.hidden)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--lrs", nargs="+", type=float, default=LRS)
    parser.add_argument("--hiddens", nargs="+", type=int, default=HIDDENS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--out", default="results/tune_neural.csv")
    parser.add_argument("--best", default="results/tune_neural_best.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(zip(old.dataset, old.method, old.lr, old.hidden, old.seed))
        print(f"{len(done)} runs already in {out}, skipping them")

    with out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            writer.writeheader()

        for name in args.datasets:
            X_train, y_train, X_test, y_test = load_uea(name)
            labels = LabelEncoder().fit(y_train)
            y_train, y_test = labels.transform(y_train), labels.transform(y_test)
            n_channels, n_classes = X_train[0].shape[0], len(labels.classes_)
            print(f"\n{name}: {n_channels} channels, {n_classes} classes")

            for seed in args.seeds:
                fit, val, _ = prepare(X_train, y_train, X_test, y_test, 0.0, seed)
                fit, val = fit.to(device), val.to(device)
                for method, lr, hidden in itertools.product(args.models, args.lrs,
                                                            args.hiddens):
                    if (name, method, lr, hidden, seed) in done:
                        continue
                    set_seed(seed)
                    model = build_model(method, n_channels, n_classes, hidden).to(device)
                    info = train_model(model, fit, val, epochs=args.epochs,
                                       patience=args.patience, lr=lr, seed=seed)
                    _, pred = evaluate(model, val)
                    row = {"dataset": name, "method": method, "lr": lr, "hidden": hidden,
                           "seed": seed, "n_params": count_params(model),
                           "val_loss": info["val_loss"],
                           "val_accuracy": accuracy_score(val.y.cpu().numpy(), pred),
                           "seconds": info["seconds"], "epochs": info["epochs"],
                           "best_epoch": info["best_epoch"], "device": device}
                    writer.writerow(row)
                    f.flush()
                    print(f"  seed={seed} {method:<9} lr={lr:<6g} hidden={hidden:<3} "
                          f"val_loss={row['val_loss']:.3f}  "
                          f"val_acc={row['val_accuracy']:.3f}  "
                          f"epochs={info['epochs']:>3} (best {info['best_epoch']})  "
                          f"time={info['seconds']:.1f}s")

    df = pd.read_csv(out)
    best = best_settings(df)
    Path(args.best).write_text(json.dumps(best, indent=2))
    print("\nChosen settings (lowest mean validation loss):")
    for name, models in best.items():
        for method, s in models.items():
            print(f"  {name:<26} {method:<9} lr={s['lr']:g} hidden={s['hidden']}")
    print(f"\nAll runs in {out}, chosen settings in {args.best}")


if __name__ == "__main__":
    main()
