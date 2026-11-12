# Continuous-time models vs. mathematical features for time series classification

Research project code. See `CLAUDE.md` for the full plan and decisions.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install numpy scipy scikit-learn aeon
```

The neural-network libraries in `requirements.txt` are only needed from experiment 1 on.

## Run the first test

```bash
python tests_signature.py
python -m experiments.exp01_first_test
```
