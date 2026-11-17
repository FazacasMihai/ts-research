"""First test: feature methods + logistic regression on BasicMotions.

Run from the project root:

    python -m experiments.exp01_first_test
"""
import time

from aeon.datasets import load_basic_motions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import dmd_features, extract, signature_features, summary_stats


def evaluate(name, fn, X_train, y_train, X_test, y_test, **kwargs):
    start = time.perf_counter()
    F_train = extract(X_train, fn, **kwargs)
    F_test = extract(X_test, fn, **kwargs)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    clf.fit(F_train, y_train)
    pred = clf.predict(F_test)
    secs = time.perf_counter() - start
    print(
        f"{name:<22} features={F_train.shape[1]:>4}  "
        f"acc={accuracy_score(y_test, pred):.3f}  "
        f"macroF1={f1_score(y_test, pred, average='macro'):.3f}  "
        f"time={secs:.2f}s"
    )


def main():
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    print(f"BasicMotions: train {X_train.shape}, test {X_test.shape}\n")
    evaluate("Summary statistics", summary_stats, X_train, y_train, X_test, y_test)
    evaluate("Signature depth 2", signature_features, X_train, y_train, X_test, y_test, depth=2)
    evaluate("Signature depth 3", signature_features, X_train, y_train, X_test, y_test, depth=3)
    evaluate("DMD (Koopman)", dmd_features, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
