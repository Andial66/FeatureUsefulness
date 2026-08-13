#!/usr/bin/env python3
"""
experiments_driver.py
===========================

Reproducible command-line driver for the three extended experiments requested by
the reviewers of "Efficient Algorithms for Detecting Relevant, Necessary and
Useful Features".  It orchestrates :mod:`experiments` and writes every table
(as CSV) and figure (as PNG) under ``--out``.

Examples
--------
Run everything on the offline Bike Sharing dataset with the paper's 20 models::

    python experiments_driver.py --datasets bike_sharing

Quick smoke run (few models, skip the slow Integrated-Gradients surrogate)::

    python experiments_driver.py --datasets bike_sharing \
        --n-models 3 --no-ig --experiments comparison binning profiling

All datasets (California and Adult Income need outbound network access to their
public sources)::

    python experiments_driver.py \
        --datasets california bike_sharing adult_income

Every run is seeded (``--seed``) so results are reproducible.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import warnings

import pandas as pd

# Make the module importable whether run from the repo root or from extensions/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments as ux


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["bike_sharing"],
                   choices=["california", "bike_sharing", "adult_income"],
                   help="datasets to run (default: bike_sharing, the only offline one)")
    p.add_argument("--experiments", nargs="+",
                   default=["synthetic", "scores", "comparison", "profiling", "binning"],
                   choices=["synthetic", "scores", "comparison", "profiling", "binning"],
                   help="which experiment families to run ('scores' reproduces the paper's "
                        "original feature usefulness qualitative analysis; 'synthetic' is "
                        "the controlled-feature study and is dataset-independent, so it runs once)")
    p.add_argument("--n-models", type=int, default=20, help="trees per configuration (paper: 20)")
    p.add_argument("--bins", type=int, default=6, help="#bins for the comparison experiment")
    p.add_argument("--seed", type=int, default=42, help="master random seed")
    p.add_argument("--sample", type=int, default=200, help="rows for sampling-based methods")
    p.add_argument("--no-ig", action="store_true", help="skip Integrated Gradients (the slow one)")
    p.add_argument("--no-lime", action="store_true", help="skip LIME")
    p.add_argument("--out", default="results", help="output directory")
    return p.parse_args(argv)


def method_list(args) -> tuple:
    methods = ["usefulness", "mdi", "permutation", "shap", "lime", "integrated_gradients"]
    if args.no_lime:
        methods.remove("lime")
    if args.no_ig:
        methods.remove("integrated_gradients")
    return tuple(methods)


def main(argv=None) -> None:
    args = parse_args(argv)
    warnings.filterwarnings("ignore")           # keep the console readable
    os.makedirs(args.out, exist_ok=True)

    ux.set_all_seeds(args.seed)
    ux.set_plot_style()
    cfg = ux.Config(seed=args.seed, n_models=args.n_models, importance_sample=args.sample)
    methods = method_list(args)

    # A run manifest makes every figure traceable back to its exact settings.
    manifest = {"seed": args.seed, "n_models": args.n_models, "bins": args.bins,
                "methods": methods, "datasets": args.datasets, "experiments": args.experiments}
    with open(os.path.join(args.out, "run_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("Run manifest:", json.dumps(manifest))

    ux.run_tree_tests()                         # fail fast if the core regressed

    # ---- synthetic controlled-feature study (dataset-independent -> run once) ----
    if "synthetic" in args.experiments:
        print("\n[synthetic] controlled-feature study (known ground truth) ...")
        syn_methods = tuple(m for m in methods if m != "integrated_gradients")
        syn = ux.run_synthetic_experiment(cfg, seed=args.seed, methods=syn_methods)
        rare = ux.run_rare_relevance_experiment(cfg, seed=args.seed)
        stag = os.path.join(args.out, "synthetic")
        syn["clean"]["auc"].to_csv(f"{stag}_recovery_auc.csv")
        rare["table"].to_csv(f"{stag}_rare_mode.csv", index=False)
        ux.plot_synthetic_recovery(syn, f"{stag}_recovery")
        ux.plot_synthetic_recovery_auc(syn["clean"]["auc"], f"{stag}_recovery_auc")
        ux.plot_rare_relevance(rare, f"{stag}_rare_mode")
        print(f"    clean recovery: accuracy={syn['clean']['accuracy']:.3f}, "
              f"Spearman(emp,true)={syn['clean']['spearman_true_vs_empirical']:.3f}, "
              f"noise AUC={syn['clean']['auc']['auc_relevant_vs_noise'].min():.3f}")
        print(f"    redundancy: x0+x0_copy={syn['redundant']['redundancy']['sum']:.2f} "
              f"(true x0={syn['redundant']['redundancy']['true_x0']:.2f})")

    for ds in args.datasets:
        tag = os.path.join(args.out, ds)
        print(f"\n{'='*70}\nDataset: {ds}\n{'='*70}")

        # ---- paper's main experiment: usefulness scores ------
        if "scores" in args.experiments:
            print("[main] usefulness scores ...")
            scores = ux.run_score_experiment(ds, cfg, strategy="uniform")
            scores["table"].to_csv(f"{tag}_usefulness_scores.csv", index=False)
            ux.plot_scores(scores, ds, f"{tag}_usefulness_scores")
            print("    top feature by median score, per bin count:")
            for bins, d in scores["per_bins"].items():
                print(f"      {bins} bins: {d['labels'][-1]:20s} (mean acc {d['mean_accuracy']:.3f})")

        # ---- (A) importance-method comparison -----------------------------
        if "comparison" in args.experiments:
            print("[A] comparison of importance methods ...")
            comp = ux.run_comparison_experiment(ds, cfg, bins=args.bins,
                                                methods=methods, n_models=args.n_models)
            comp["topk"].to_csv(f"{tag}_topk_overlap.csv")
            comp["corr_spearman"].to_csv(f"{tag}_rank_correlation.csv")
            comp["gt"].to_csv(f"{tag}_ground_truth_agreement.csv")
            comp["mean_importance"].to_csv(f"{tag}_mean_importance.csv")
            ux.plot_method_importance_heatmap(comp, f"{tag}_importance_heatmap")
            ux.plot_rank_correlation_heatmap(comp, f"{tag}_rank_correlation")
            ux.plot_topk_intersection(comp, f"{tag}_topk_overlap")
            ux.plot_ground_truth_agreement(comp, f"{tag}_ground_truth_agreement")
            comp["gt_ci"].to_csv(f"{tag}_ground_truth_ci.csv")     # mean + 95% bootstrap CI
            print("    ground-truth agreement (Spearman ρ, mean [95% CI]):")
            for m in comp["gt_ci"].index:
                r = comp["gt_ci"].loc[m]
                print(f"      {m:22s} {r['mean']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
            w = comp["wilcoxon"].get("usefulness_vs_shap", {})
            if w:
                print(f"    paired Wilcoxon usefulness vs SHAP: p = {w.get('p'):.3g} "
                      f"(n={w.get('n')})  {w.get('note', '')}")

        # ---- (B) runtime & memory analysis --------------------------------
        if "profiling" in args.experiments:
            print("[B] runtime & memory profiling ...")
            scaling = ux.scaling_experiment(ds, cfg, strategy="uniform")
            scaling.to_csv(f"{tag}_scaling.csv", index=False)
            ux.plot_scaling(scaling, ds, f"{tag}_scaling")

            # runtime vs #features is dataset-independent (synthetic) -> compute once
            if ds == args.datasets[0]:
                nfeat = ux.runtime_vs_n_features(cfg)
                nfeat.to_csv(os.path.join(args.out, "runtime_vs_n_features.csv"), index=False)
                ux.plot_runtime_vs_n_features(nfeat, os.path.join(args.out, "runtime_vs_n_features"))

            model = ux.train_tree(ds, bins=args.bins, strategy="uniform",
                                  seed=args.seed, leaves=cfg.leaves_per_bin[ds] * args.bins)
            runtimes = ux.method_runtime_comparison(model, cfg, methods=methods)
            runtimes.to_csv(f"{tag}_method_runtimes.csv", index=False)
            ux.plot_method_runtimes(runtimes, ds, f"{tag}_method_runtimes")
            print("    per-method runtime (s):")
            print(runtimes.set_index("method")["time_s"].round(4).to_string())

            # (a) how much of the usefulness runtime was copy.deepcopy overhead
            speedup = pd.DataFrame([ux.profile_scorer_speedup(model)])
            speedup.to_csv(f"{tag}_scorer_speedup.csv", index=False)
            ux.plot_scorer_speedup(speedup, f"{tag}_scorer_speedup")
            print(f"    scorer speedup (deepcopy -> copy-free): {speedup['speedup'].iloc[0]:.1f}x")

            # (b) runtime vs number of instances (usefulness is data-free -> flat)
            size_methods = tuple(m for m in methods if m != "integrated_gradients")
            rt_size = ux.runtime_vs_dataset_size(model, cfg, methods=size_methods)
            rt_size.to_csv(f"{tag}_runtime_vs_size.csv", index=False)
            ux.plot_runtime_vs_dataset_size(rt_size, ds, f"{tag}_runtime_vs_size")

            # (c) wall-clock each approximate method needs to reach a stable ranking
            stable_methods = tuple(m for m in ("shap", "permutation", "lime") if m in methods)
            stable = ux.time_to_stable_ranking(model, cfg, methods=stable_methods)
            stable.to_csv(f"{tag}_time_to_stable.csv", index=False)
            ux.plot_time_to_stable(stable, ds, f"{tag}_time_to_stable")

            # (d) quality (ground-truth agreement) vs cost (runtime), Pareto view
            qc = ux.quality_cost_table(model, cfg, methods=methods)
            qc.to_csv(f"{tag}_quality_cost.csv", index=False)
            ux.plot_quality_cost_pareto(qc, ds, f"{tag}_quality_cost")

        # ---- (C) binning-strategy sensitivity -----------------------------
        if "binning" in args.experiments:
            print("[C] binning-strategy sensitivity ...")
            binning = ux.run_binning_experiment(ds, cfg, n_models=args.n_models)
            for key, frame in binning.items():
                frame.to_csv(f"{tag}_binning_{key}.csv", index=False)
            ux.plot_binning_sensitivity(binning, ds, f"{tag}_binning")
            print("    accuracy / ground-truth ρ by strategy and bins:")
            print(binning["summary"][["strategy", "bins", "accuracy", "spearman_gt"]]
                  .round(3).to_string(index=False))

    print(f"\nDone. Tables and figures written to '{args.out}/'.")


if __name__ == "__main__":
    main()
