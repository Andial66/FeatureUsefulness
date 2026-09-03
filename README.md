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
of the paper. Because this figure is noisier than the rest (each box already
pools 3 tree sizes, so a handful of reps swings the quartiles a lot), the
driver runs it at its own, higher rep count — `--scores-n-models` (default
200) — independent of `--n-models`. `plot_scores_grid()` additionally
combines the per-bin-count panels (3, 4, 5, 6, 8) into one figure,
`{dataset}_usefulness_scores_grid.png`: 3 same-sized panels on the first row,
the remaining 2 centered on the second, under one title
"Usefulness score - {Dataset}".

### (A) Compare the usefulness score with other importance methods
`run_comparison_experiment()` computes, on the **same** trees, the usefulness
score alongside:

* **Permutation importance** (`sklearn.inspection`, on held-out data),
* **SHAP** (TreeSHAP — the paper’s baseline),
* **LIME** (local surrogates, aggregated to a global score).

The methods' *rankings* are then compared three ways: pairwise Spearman/Kendall
correlation (every position weighted equally), top-k overlap with the usefulness
ranking (paper's Table 1, extended to every method), and pairwise **Rank-Biased
Overlap** (`rank_biased_overlap()` — top-weighted, so two methods agreeing on the
most important features counts for more than agreeing further down the list).

### (B) Runtime analysis
`scaling_experiment()` measures how the usefulness algorithm **scales with #bins**
(wall-clock via `time.perf_counter`). `method_runtime_comparison()` measures the
**cost of every method** on the same model. `method_scaling_experiment()` compares
**every method's runtime as tree size grows** (one line per method, with a fitted
power law for usefulness), each point repeated 10x so the measurement noise is
shown as a mean ± std band — genuinely measured for every method, SHAP included;
its band is just visually thin because SHAP's run-to-run timing noise is small
in absolute terms. `plot_method_scaling()` renders this two ways — log-log
(`_runtime.png`) and linear (`_runtime_linear.png`, tree sizes below 5% of the
largest dropped so the log-spaced size grid doesn't crowd the small end) — plus
a coefficient-of-variation bar chart (`_variance.png`). The linear view is the
one that shows usefulness's actual crossover with the other methods at large
tree sizes, which the log-log view flattens into parallel-looking lines.

### (C) Binning-strategy sensitivity
`run_binning_experiment()` re-runs the usefulness experiment under
`uniform` / `quantile` / `kmeans` discretisation and reports, per strategy and #bins,
model accuracy plus **ranking stability** two ways: cross-strategy (fixed #bins)
and cross-#bins (fixed strategy), each measured with both Spearman rho and
Rank-Biased Overlap, so top-of-ranking stability is visible alongside overall
rank stability.

---

## How to run (reproducible)

```bash
pip install -r requirements.txt

# Offline, quick check (Bike Sharing is a local CSV — no internet needed):
python experiments_driver.py --datasets bike_sharing --n-models 3

# Full run, paper settings (20 models). California & Adult download their public
# sources on first use:
python experiments_driver.py \
    --datasets california bike_sharing adult_income --n-models 20
```

Every run is seeded (`--seed`, default 42) and writes a `run_manifest.json`
recording the exact settings. Tables are CSVs and every figure is saved as 
`.png`. Use `--help` for all options; you canrun a subset with e.g. 
`--experiments scores comparison`, or `--no-lime` to skip the optional method.

You can also drive it from a notebook cell:
```python
import sys; sys.path.insert(0, "extensions")
import experiments as ux
ux.set_all_seeds(42); ux.set_plot_style()
cfg = ux.Config(n_models=20)
comp = ux.run_comparison_experiment("bike_sharing", cfg, bins=6)
comp["corr_spearman"]  # pairwise Spearman rank correlation between methods
```

## Notes

* **Datasets that need network.** California Housing (`sklearn`) and Adult Income
  (UCI) are downloaded on demand. Bike Sharing runs fully offline from the CSV
  already in `datasets/`.
* **Reproducibility of discretization.** `KBinsDiscretizer` is built with
  `subsample=None` so `quantile`/`kmeans` bin edges are deterministic.
* **Plot typography.** `set_plot_style()` renders every figure's text through a
  real LaTeX install (Computer Modern), for a look that matches a LaTeX-typeset
  paper. This needs `latex`/`dvipng` on `PATH` (any TeX distribution, e.g.
  TeX Live or MiKTeX); if none is found it automatically falls back to
  matplotlib's bundled Computer-Modern-like fonts, so the code still runs.
