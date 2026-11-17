"""Load UEA datasets (downloaded once into data/uea/)."""
from pathlib import Path

from aeon.datasets import load_classification

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "uea"


def load_uea(name):
    """Return X_train, y_train, X_test, y_test with the official split."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_classification(name, split="train", extract_path=str(DATA_DIR))
    X_test, y_test = load_classification(name, split="test", extract_path=str(DATA_DIR))
    return X_train, y_train, X_test, y_test
