"""Experiment 2, feature methods: accuracy when time points are missing.

For each dataset, missing fraction and seed: remove points, compute
features, train logistic regression, score on the test split. One CSV row
per run.

Run from the project root:

    python -m experiments.exp02_missing_data
    python -m experiments.exp02_missing_data --datasets BasicMotions --seeds 0
"""
import argparse
import csv
import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_uea
from src.features import dmd_features, extract, signature_features, summary_stats
from src.missing import drop_points, series_lengths, to_even_grid

DATASETS = ["BasicMotions", "Libras", "NATOPS", "ArticularyWordRecognition"]
FRACS = [0.0, 0.3, 0.5, 0.7]
SEEDS = [0, 1, 2]
FIELDS = ["dataset", "missing_frac", "seed", "method", "n_features",
          "accuracy", "macro_f1", "seconds"]


def features(method, values, times, lengths):
    """Feature matrix for one method. DMD first goes back to an even grid."""
    if method == "Summary statistics":
        return extract(values, summary_stats)
    if method == "Signature depth 2":
        return extract(values, signature_features, times=times, depth=2)
    if method == "Signature depth 3":
        return extract(values, signature_features, times=times, depth=3)
    if method == "DMD (Koopman)":
        return extract(to_even_grid(values, times, lengths), dmd_features)
    raise ValueError(method)


METHODS = ["Summary statistics", "Signature depth 2", "Signature depth 3", "DMD (Koopman)"]


def run_one(method, train, test, y_train, y_test):
    start = time.perf_counter()
    F_train = features(method, *train)
    F_test = features(method, *test)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    clf.fit(F_train, y_train)
    pred = clf.predict(F_test)
    return {
        "n_features": F_train.shape[1],
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "seconds": time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--fracs", nargs="+", type=float, default=FRACS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--out", default="results/exp02_missing_features.csv")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for name in args.datasets:
            X_train, y_train, X_test, y_test = load_uea(name)
            len_train, len_test = series_lengths(X_train), series_lengths(X_test)
            print(f"\n{name}: {len(X_train)} train / {len(X_test)} test series, "
                  f"{X_train[0].shape[0]} channels, up to {max(len_train)} points")

            for frac in args.fracs:
                # Nothing removed: every seed gives the same data, so run once.
                seeds = args.seeds[:1] if frac == 0 else args.seeds
                for seed in seeds:
                    # Different (but fixed) gaps for train and test.
                    train = (*drop_points(X_train, frac, seed=(seed, 0)), len_train)
                    test = (*drop_points(X_test, frac, seed=(seed, 1)), len_test)
                    for method in args.methods:
                        row = {"dataset": name, "missing_frac": frac, "seed": seed,
                               "method": method,
                               **run_one(method, train, test, y_train, y_test)}
                        writer.writerow(row)
                        f.flush()
                        print(f"  missing={frac:.0%} seed={seed} {method:<20} "
                              f"features={row['n_features']:>5}  "
                              f"acc={row['accuracy']:.3f}  "
                              f"macroF1={row['macro_f1']:.3f}  "
                              f"time={row['seconds']:.1f}s")

    df = pd.read_csv(out)
    table = df.pivot_table(index=["dataset", "method"], columns="missing_frac",
                           values="accuracy", aggfunc="mean")
    print("\nMean accuracy (columns = fraction of points removed):")
    print(table.round(3).to_string())
    print(f"\nAll runs saved to {out}")


if __name__ == "__main__":
    main()
