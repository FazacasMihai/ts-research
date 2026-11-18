# Continuous-time models vs. mathematical features for time series classification

Research project code.

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
python tests_neural.py
python -m experiments.exp01_first_test
python -m experiments.exp02_missing_data
python -m experiments.tune_neural
python -m experiments.exp03_neural --params results/tune_neural_best.json
```

`tune_neural` gives every neural model the same small grid (learning rate 1e-3, 3e-3,
1e-2; hidden size 32, 64) on clean data and the validation split only. It writes every
run to `results/tune_neural.csv` and the setting with the lowest validation loss per
dataset and model to `results/tune_neural_best.json`. Finished runs are skipped, so a
stopped search can be started again. Quick check:

```bash
python -m experiments.tune_neural --datasets BasicMotions --models GRU
```

`exp03_neural` trains the neural models (GRU, LSTM, CfC, Neural CDE, LTC) on the same
datasets and the same gaps as `exp02`, and writes to `results/exp03_neural.csv`.
The full grid is slow on a CPU; run it on a GPU (e.g. Colab). Quick check:

```bash
python -m experiments.exp03_neural --datasets BasicMotions --fracs 0 --seeds 0
```

`exp02_missing_data` removes 0/30/50/70% of the time points of every series and
writes one row per run to `results/exp02_missing_features.csv`. Datasets are
downloaded once into `data/uea/`. For a quick run:

```bash
python -m experiments.exp02_missing_data --datasets BasicMotions --seeds 0
```

## Paper

`paper/main.tex` (amsart) includes the chapters in `paper/sections/`; the bibliography is
`paper/references.bib`. Red `[TODO: ...]` notes mark what is still to be written.
Compile with `latexmk -pdf main.tex` inside `paper/`, or upload the folder to Overleaf.
