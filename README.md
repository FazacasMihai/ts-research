# Continuous-time models vs. mathematical features for time series classification

Research project code. See `CLAUDE.md` for the full plan and decisions.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install numpy scipy scikit-learn aeon
```

The neural-network libraries in `requirements.txt` are only needed from experiment 1 on.

## Run the tests and experiments

Always run from the project root (so `src` can be imported):

```bash
python tests_signature.py
python tests_missing.py
python -m experiments.exp01_first_test
python -m experiments.exp02_missing_data
```

`exp02_missing_data` removes 0/30/50/70% of the time points of every series and
writes one row per run to `results/exp02_missing_features.csv`. Datasets are
downloaded once into `data/uea/`. For a quick run:

```bash
python -m experiments.exp02_missing_data --datasets BasicMotions --seeds 0
```
