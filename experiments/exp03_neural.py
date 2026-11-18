"""Experiments 1 + 2, neural models: clean data and missing time points.

Uses the same datasets, gaps and seeds as exp02. 20% of the train split is
kept aside for early stopping; the test split is used once at the end.
One CSV row per run. --params loads the settings chosen by tune_neural.

Run from the project root:

    python -m experiments.exp03_neural --datasets BasicMotions --fracs 0 --seeds 0
    python -m experiments.exp03_neural            # full grid (use a GPU / Colab)
    python -m experiments.exp03_neural --params results/tune_neural_best.json
"""
import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data import load_uea
from src.missing import drop_points, series_lengths
from src.models import MODELS, build_model, count_params
from src.prep import Normaliser, to_tensors
from src.train import evaluate, set_seed, train_model

DATASETS = ["BasicMotions", "Libras", "NATOPS", "ArticularyWordRecognition"]
FRACS = [0.0, 0.3, 0.5, 0.7]
SEEDS = [0, 1, 2]
FIELDS = ["dataset", "missing_frac", "seed", "method", "lr", "hidden", "n_params",
          "accuracy", "macro_f1", "seconds", "epochs", "best_epoch", "device"]


def prepare(X_train, y_train, X_test, y_test, frac, seed):
    """Remove points, normalise, and split off a validation set."""
    time_scale = max(series_lengths(X_train) + series_lengths(X_test)) - 1
    v_train, t_train = drop_points(X_train, frac, seed=(seed, 0))
    v_test, t_test = drop_points(X_test, frac, seed=(seed, 1))

    norm = Normaliser().fit(v_train)
    train = to_tensors(norm.transform(v_train), t_train, y_train, time_scale)
    test = to_tensors(norm.transform(v_test), t_test, y_test, time_scale)

    idx_fit, idx_val = train_test_split(range(len(train)), test_size=0.2,
                                        stratify=y_train, random_state=seed)
    return train.subset(idx_fit), train.subset(idx_val), test


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--fracs", nargs="+", type=float, default=FRACS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--params", help="JSON from tune_neural (overrides --hidden, --lr)")
    # Small datasets give only 1-2 updates per epoch, so allow many epochs.
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--out", default="results/exp03_neural.csv")
    args = parser.parse_args()

    params = json.loads(Path(args.params).read_text()) if args.params else {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for name in args.datasets:
            X_train, y_train, X_test, y_test = load_uea(name)
            labels = LabelEncoder().fit(y_train)
            y_train, y_test = labels.transform(y_train), labels.transform(y_test)
            n_channels, n_classes = X_train[0].shape[0], len(labels.classes_)
            print(f"\n{name}: {len(X_train)} train / {len(X_test)} test, "
                  f"{n_channels} channels, {n_classes} classes")

            for frac in args.fracs:
                seeds = args.seeds[:1] if frac == 0 else args.seeds
                for seed in seeds:
                    fit, val, test = prepare(X_train, y_train, X_test, y_test, frac, seed)
                    fit, val, test = fit.to(device), val.to(device), test.to(device)
                    for method in args.models:
                        chosen = params.get(name, {}).get(method, {})
                        if params and not chosen:
                            print(f"  no tuned settings for {name} {method}, using defaults")
                        lr = chosen.get("lr", args.lr)
                        hidden = chosen.get("hidden", args.hidden)
                        set_seed(seed)
                        model = build_model(method, n_channels, n_classes, hidden).to(device)
                        info = train_model(model, fit, val, epochs=args.epochs,
                                           patience=args.patience, lr=lr, seed=seed)
                        _, pred = evaluate(model, test)
                        y_true = test.y.cpu().numpy()
                        row = {"dataset": name, "missing_frac": frac, "seed": seed,
                               "method": method, "lr": lr, "hidden": hidden,
                               "n_params": count_params(model),
                               "accuracy": accuracy_score(y_true, pred),
                               "macro_f1": f1_score(y_true, pred, average="macro"),
                               "seconds": info["seconds"], "epochs": info["epochs"],
                               "best_epoch": info["best_epoch"], "device": device}
                        writer.writerow(row)
                        f.flush()
                        print(f"  missing={frac:.0%} seed={seed} {method:<6} "
                              f"params={row['n_params']:>6}  acc={row['accuracy']:.3f}  "
                              f"macroF1={row['macro_f1']:.3f}  "
                              f"epochs={info['epochs']:>3} (best {info['best_epoch']})  "
                              f"time={info['seconds']:.1f}s")

    df = pd.read_csv(out)
    table = df.pivot_table(index=["dataset", "method"], columns="missing_frac",
                           values="accuracy", aggfunc="mean")
    print("\nMean accuracy (columns = fraction of points removed):")
    print(table.round(3).to_string())
    print(f"\nAll runs saved to {out}")


if __name__ == "__main__":
    main()
