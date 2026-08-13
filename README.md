# Extended experiments for *Feature Relevancy, Necessity and Usefulness*

This folder is a **self-contained** replacement for the whole experimentation:
it reproduces the paper's original results **and** adds the three analyses the
reviewers asked for, from one module and one command. It **does not modify, and
does not import, any existing file** — the original `Experiments.ipynb` is kept
only as a reference and is not needed to run anything here.

| File | What it is |
|---|---|
| `experiments.py` | Corrected, refactored re-implementation of the full usefulness-score pipeline (its own `Node`/`Tree`/`compute_score`, dataset loaders, trainer) **plus** all experiment code: the paper's score figures, importance comparison, profiling, and binning. Heavily commented. |
| `experiments_notebook.ipynb` | Interactive notebook: a thin, narrated layer that imports the module and shows every table/figure inline. Best for reading and exploring. |
| `experiments_driver.py` | The same experiments as a headless command-line driver, for one-shot reproducible runs. Writes tables (CSV) and figures (PNG+PDF) to `results_ext/`. |
| `requirements.txt` | Dependencies (all already used by the notebook, plus `lime`). |

There are two equivalent front-ends over the same tested engine: open
`experiments_notebook.ipynb` (edit `DATASET`/`N_MODELS`, *Run All*) for an
interactive read, or run the driver below for a headless one-shot.

**Do I need both the notebook and this folder?** No. This folder reproduces
everything on its own. The `scores` experiment below regenerates the paper's
Fig. 4/5; the `comparison` experiment supersedes the notebook's SHAP comparison
(Table 1) and extends it to more methods. `experiments.py` mirrors the
notebook's `Node`/`Tree`/`compute_score` logic (so numbers stay comparable) and
only changes the parts tagged `# FIX:` in the source.

---

## 1. Errors found in the original code

| # | Severity | Location (in `Experiments.ipynb`) | Problem | Effect |
|---|---|---|---|---|
| **B1** | **High – wrong results** | Bike Sharing trainer: `features_to_domain["season"] = (0, 1)` and `["weathersit"] = (0, 4)` | The raw categorical features `season` and `weathersit` actually take the values **1..4** (verified from `hour.csv`), not `0..1` / `0..4`. | With the wrong domain, **`season`’s usefulness score collapses to 0 and it is ranked *last* (12/12)** — which is exactly what Figure 5 shows. After the fix (`season=(1,4)`) it scores ≈1.1e7 and is ranked **3rd**. The reported "season is least useful" is an artifact of this bug. |
| **B2** | **High – silent data loss** | Adult Income cell: `with open(f"results/california_bins_{bins}", ...)` | The Adult-Income results loop **saves to the California filename**, overwriting the California results. | Any run of that cell corrupts `results/california_bins_*`. Should be `results/adult_income_bins_{bins}`. |
| **B3** | Medium – latent | `dec_tree_to_my_tree_rec`: `threshold = clf.tree_.threshold[node]` | Reads the threshold from the **global** `clf` instead of the `dec_tree` argument being converted. | Works only because the global happens to be the same tree; converting any other tree silently produces wrong thresholds. Fixed to `dec_tree.tree_.threshold[node]`. |
| **B4** | Medium – latent | `compute_score`: `for feature in features:` | Iterates a **global** `features` list (and shadows the `feature` argument); the `num_features` argument is unused. | Breaks/incorrect if `features` ≠ `domains.keys()`. Fixed to iterate `domains`. |
| **B5** | Low – defensive (no-op in practice) | `compute_score`: `tree_0.condition(feature, 0)` then `range(domain_min+1, …)` | The smallest feature value is **hard-coded to 0**; generalised to start from the true `domain_min`. | **Correction to an earlier claim:** this is a *no-op for real sklearn trees* — a tree never splits below a feature's minimum, so conditioning on `0` vs the true minimum restricts the tree identically (verified: current == original code on both domain regimes). It changed no score; **B1 (the domain declaration) alone** causes the `season` change. Kept as defensive hygiene, covered by a unit test. |
| **B6** | Low – paper/code mismatch | discretisation `strategy=` | The **paper text says `uniform`**, but the released code uses `strategy='quantile'` for California and Bike Sharing (only Adult uses `uniform`). | The ranking is sensitive to this choice (see task C), so the mismatch matters. The new code makes the strategy an explicit parameter. |
| **B7** | Low – misleading tests | `Tests` cell | The self-tests print `tree.count()` (always 7) regardless of which tree they mean to check, so a regression could not fail them. | Rewritten as real `assert`s in `run_tree_tests()`. |
| **B8** | Medium – biases scores | binned-feature domains declared as `(0, n_bins-1)` | `KBinsDiscretizer` can leave a bin empty (skewed features under `uniform`) or merge bins (sparse features under `quantile`), so the ordinal output skips values; declaring the full `(0, n_bins-1)` then counts "phantom" values that never occur. | Over-counts the entity space and biases the score; under `quantile`, `capital-gain`/`capital-loss` collapse to a constant on Adult. Found via `check_domains.py` (California/`uniform`, Adult/`quantile`). Fixed by densifying each binned column to its realized bins (`_densify`); a no-op where bins were already full, so released results are unchanged. |

All seven of the tree-algebra reference counts from the notebook’s test cell still
pass against the corrected code (`run_tree_tests()`), so the fixes do not change
the core semantics — only the buggy edges.

---

## 2. What it runs

### (main) The paper's usefulness-score figures (Fig. 4 / 5)
`run_score_experiment()` trains `n_models` trees per bin count, aggregates each
feature's usefulness score into Q1/median/Q3 and the mean accuracy, and
`plot_scores()` draws the exact Fig. 4/5 layout. This is the corrected version of
the notebook's "Experiment 1" (e.g. on Bike Sharing, `season` is no longer forced
to last place by the domain bug).

### (A) Compare the usefulness score with other importance methods
`run_comparison_experiment()` computes, on the **same** trees, the usefulness
score alongside:

* **MDI / Gini** (`clf.feature_importances_`, free),
* **Permutation importance** (`sklearn.inspection`, on held-out data),
* **SHAP** (TreeSHAP — the paper’s baseline),
* **LIME** (local surrogates, aggregated to a global score),
* **Integrated Gradients** (needs a differentiable model, so it is run on a small
  **MLP surrogate** — itself a talking point: the usefulness score needs no surrogate).

Because the methods live on different scales, they are compared by the **rankings**
they induce, three ways:
* pairwise **Spearman/Kendall** rank-correlation between methods;
* **top-k overlap** of each method with the usefulness ranking — this is exactly
  the metric of the paper’s Table 1, now extended from SHAP-only to every method;
* agreement with the **domain ground-truth ranking** stated in Section 4 of the
  paper (Spearman ρ, top-1 hit, top-3 recall) — i.e. *which method best recovers
  known importance*.

### (B) Runtime and memory analysis
`scaling_experiment()` + `method_runtime_comparison()` measure how the usefulness
algorithm **scales with #bins and tree size** (wall-clock via `time.perf_counter`,
peak memory via the stdlib `tracemalloc`; the tree-size panel overlays a fitted
power law so empirical ≈ theoretical complexity), and the **cost of every method**
on the same model.

Because the raw runtime bar is unfair to the method in measurable ways, four extra
analyses make the comparison both fairer and more informative:

* **(a) `profile_scorer_speedup()`** — most of the prototype's time was
  `copy.deepcopy`, not the algorithm. A copy-free scorer (`Tree.clone`, sharing the
  immutable `domains`) is **~5x faster with byte-identical results**; it is now the
  default path in `compute_score`.
* **(b) `runtime_vs_dataset_size()`** — usefulness reads only the tree, so it is
  **data-free**: its runtime is flat in the number of instances, while
  SHAP / permutation / LIME grow (LIME crosses above it by ~40 rows).
* **(c) `time_to_stable_ranking()`** — the approximate methods are cheap only at a
  fixed small sample; this reports the wall-clock each needs before its ranking
  stops changing. The exact methods are stable by construction.
* **(d) `quality_cost_table()` + Pareto plot** — ground-truth agreement (quality)
  vs runtime (cost). *Caveat:* the quality axis can saturate when the paper's
  ground truth lists few features (as on Bike Sharing, where several methods tie),
  so read it together with the richer datasets.

Honest takeaway: usefulness is fast in absolute terms and, once the scorer is
optimised, competitive on wall-clock; its real distinction is being **exact,
surrogate-free and data-independent**, not "fastest".

### (C) Binning-strategy sensitivity
`run_binning_experiment()` re-runs the usefulness experiment under
`uniform` / `quantile` / `kmeans` discretisation and reports, per strategy and #bins:
model accuracy, agreement with ground truth, and **cross-strategy ranking stability**
(how much the induced ranking moves when you change the binning).

---

## 3. How to run (reproducible)

Two steps — install, then run one command. That command reproduces **everything**
(the paper's score figures + all three new analyses).

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
recording the exact settings. Tables are CSVs and every figure is saved as both
`.png` and `.pdf` (drop-in for the paper). Use `--help` for all options; you can
run a subset with e.g. `--experiments scores comparison`, or `--no-ig`/`--no-lime`
to skip the slow/optional methods.

You can also drive it from a notebook cell:
```python
import sys; sys.path.insert(0, "extensions")
import experiments as ux
ux.set_all_seeds(42); ux.set_plot_style()
cfg = ux.Config(n_models=20)
comp = ux.run_comparison_experiment("bike_sharing", cfg, bins=6)
comp["gt"]            # ground-truth agreement per method
```

## 4. Notes / caveats

* **Datasets that need network.** California Housing (`sklearn`) and Adult Income
  (UCI) are downloaded on demand. Bike Sharing runs fully offline from the CSV
  already in `datasets/`. The Adult loader falls back to sklearn’s OpenML copy if
  the UCI URL is unavailable.
* **Integrated Gradients** is computed on an MLP surrogate (a decision tree is not
  differentiable). The surrogate’s test accuracy is reported so its fidelity is
  visible. To use a real torch model + captum instead, swap the backend in
  `importance_integrated_gradients()`.
* **Reproducibility of discretisation.** `KBinsDiscretizer` is built with
  `subsample=None` so `quantile`/`kmeans` bin edges are deterministic.
