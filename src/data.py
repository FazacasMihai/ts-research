"""Loading the public UEA datasets (downloaded once, then read from data/uea/)."""
from pathlib import Path

from aeon.datasets import load_classification

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "uea"


def load_uea(name):
    """Fixed train/test split of a UEA dataset, as published in the archive.

    Returns X_train, y_train, X_test, y_test. X is a 3D array
    (n_cases, n_channels, n_timepoints), or a list of 2D arrays if the
    series have different lengths. The first call downloads the dataset.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_classification(name, split="train", extract_path=str(DATA_DIR))
    X_test, y_test = load_classification(name, split="test", extract_path=str(DATA_DIR))
    return X_train, y_train, X_test, y_test
