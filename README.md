# Extended experiments for *Efficient Algorithms for Detecting Relevant, Necessary and Useful Features*

This repository contains the whole experimentation for the paper "Efficient 
Algorithms for Detecting Relevant, Necessary and Useful Features" by T.
Capdevielle and S. Cifuentes.

| File | What it is |
|---|---|
| `experiments.py` | Corrected, refactored re-implementation of the full usefulness-score pipeline (its own `Node`/`Tree`/`compute_score`, dataset loaders, trainer) **plus** all experiment code: the paper's score figures, importance comparison, profiling, and binning. Heavily commented. |
| `experiments_notebook.ipynb` | Interactive notebook: a thin, narrated layer that imports the module and shows every table/figure inline. Best for reading and exploring. |
| `experiments_driver.py` | The same experiments as a headless command-line driver, for one-shot reproducible runs. Writes tables (CSV) and figures (PNG+PDF) to `results_ext/`. |
| `requirements.txt` | Dependencies (all already used by the notebook, plus `lime`). |

There are two equivalent front-ends over the same tested engine: open
`experiments_notebook.ipynb` (edit `DATASET`/`N_MODELS`, *Run All*) for an
interactive read, or run the driver below for a headless one-shot.

---

## What it runs

### (main) The paper's usefulness-score figures
`run_score_experiment()` trains `n_models` trees per bin count, aggregates each
feature's usefulness score into Q1/median/Q3 and the mean accuracy, and
`plot_scores()` draws the exact Fig. 4/5 layout. This is the originalexperiment
of the paper.

### (A) Compare the usefulness score with other importance methods
`run_comparison_experiment()` computes, on the **same** trees, the usefulness
score alongside:

* **MDI / Gini** (`clf.feature_importances_`, free),
* **Permutation importance** (`sklearn.inspection`, on held-out data),
* **SHAP** (TreeSHAP — the paper’s baseline),
* **LIME** (local surrogates, aggregated to a global score),
* **Integrated Gradients** (needs a differentiable model, so it is run on a small
  **MLP surrogate** — itself a talking point: the usefulness score needs no surrogate).

### (B) Runtime and memory analysis
`scaling_experiment()` + `method_runtime_comparison()` measure how the usefulness
algorithm **scales with #bins and tree size** (wall-clock via `time.perf_counter`,
peak memory via the stdlib `tracemalloc`; the tree-size panel overlays a fitted
power law so empirical ≈ theoretical complexity), and the **cost of every method**
on the same model.

### (C) Binning-strategy sensitivity
`run_binning_experiment()` re-runs the usefulness experiment under
`uniform` / `quantile` / `kmeans` discretisation and reports, per strategy and #bins:
model accuracy, agreement with ground truth, and **cross-strategy ranking stability**
(how much the induced ranking moves when you change the binning).

---

## How to run (reproducible)

```bash
pip install -r requirements.txt

# Offline, quick check (Bike Sharing is a local CSV — no internet needed):
python experiments_driver.py --datasets bike_sharing --n-models 3 --no-ig

# Full run, paper settings (20 models). California & Adult download their public
# sources on first use:
python experiments_driver.py \
    --datasets california bike_sharing adult_income --n-models 20
```

Every run is seeded (`--seed`, default 42) and writes a `run_manifest.json`
recording the exact settings. Tables are CSVs and every figure is saved as 
`.png`. Use `--help` for all options; you canrun a subset with e.g. 
`--experiments scores comparison`, or `--no-ig`/`--no-lime`to skip the 
slow/optional methods.

You can also drive it from a notebook cell:
```python
import sys; sys.path.insert(0, "extensions")
import experiments as ux
ux.set_all_seeds(42); ux.set_plot_style()
cfg = ux.Config(n_models=20)
comp = ux.run_comparison_experiment("bike_sharing", cfg, bins=6)
comp["gt"]            # ground-truth agreement per method
```

## Notes

* **Datasets that need network.** California Housing (`sklearn`) and Adult Income
  (UCI) are downloaded on demand. Bike Sharing runs fully offline from the CSV
  already in `datasets/`.
* **Integrated Gradients** is computed on an MLP surrogate (a decision tree is not
  differentiable). The surrogate’s test accuracy is reported so its fidelity is
  visible.
* **Reproducibility of discretization.** `KBinsDiscretizer` is built with
  `subsample=None` so `quantile`/`kmeans` bin edges are deterministic.
