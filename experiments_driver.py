#!/usr/bin/env python3
"""
experiments_driver.py
===========================

Reproducible command-line driver for the paper's experiments.
It orchestrates :mod:`experiments` and writes every table
(as CSV) and figure (as PNG) under ``--out``.

Examples
--------
Run everything on the offline Bike Sharing dataset with the paper's 20 models::

    python experiments_driver.py --datasets bike_sharing

Quick smoke run (few models)::

    python experiments_driver.py --datasets bike_sharing \
        --n-models 3 --experiments comparison binning profiling

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments as ux


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["bike_sharing"],
                   choices=["california", "bike_sharing", "adult_income"],
                   help="datasets to run (default: bike_sharing, the only offline one)")
    p.add_argument("--experiments", nargs="+",
                   default=["scores", "comparison", "profiling", "binning"],
                   choices=["scores", "comparison", "profiling", "binning"],
                   help="which experiment families to run ('scores' reproduces the paper's "
                        "original feature usefulness qualitative analysis)")
    p.add_argument("--n-models", type=int, default=20, help="trees per configuration (paper: 20)")
    p.add_argument("--bins", type=int, default=6, help="#bins for the comparison experiment")
    p.add_argument("--seed", type=int, default=42, help="master random seed")
    p.add_argument("--sample", type=int, default=200, help="rows for sampling-based methods")
    p.add_argument("--no-lime", action="store_true", help="skip LIME")
    p.add_argument("--out", default="results", help="output directory")
    return p.parse_args(argv)


def method_list(args) -> tuple:
    methods = ["usefulness", "permutation", "shap", "lime"]
    if args.no_lime:
        methods.remove("lime")
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
            comp["rbo"].to_csv(f"{tag}_rank_biased_overlap.csv")
            comp["mean_importance"].to_csv(f"{tag}_mean_importance.csv")
            ux.plot_method_importance_heatmap(comp, f"{tag}_importance_heatmap")
            ux.plot_rank_correlation_heatmap(comp, f"{tag}_rank_correlation")
            ux.plot_rank_biased_overlap_heatmap(comp, f"{tag}_rank_biased_overlap")
            ux.plot_topk_intersection(comp, f"{tag}_topk_overlap")
            print("    mean Spearman rank correlation with usefulness:")
            corr = comp["corr_spearman"]["usefulness"].drop("usefulness", errors="ignore")
            for m, r in corr.sort_values(ascending=False).items():
                print(f"      {m:22s} {r:.3f}")
            print(f"    mean Rank-Biased Overlap with usefulness (p={cfg.rbo_p}):")
            rbo = comp["rbo"]["usefulness"].drop("usefulness", errors="ignore")
            for m, r in rbo.sort_values(ascending=False).items():
                print(f"      {m:22s} {r:.3f}")

        # ---- (B) runtime analysis ------------------------------------------
        if "profiling" in args.experiments:
            print("[B] runtime profiling ...")
            scaling = ux.scaling_experiment(ds, cfg, strategy="uniform")
            scaling.to_csv(f"{tag}_scaling.csv", index=False)
            ux.plot_scaling(scaling, ds, f"{tag}_scaling")

            # runtime vs tree size, every method, repeated for variance
            method_scaling = ux.method_scaling_experiment(ds, cfg, strategy="uniform", methods=methods)
            method_scaling.to_csv(f"{tag}_method_scaling.csv", index=False)
            ux.plot_method_scaling(method_scaling, ds, f"{tag}_method_scaling")

            # focused head-to-head: usefulness vs SHAP (both structure-aware methods)
            if {"usefulness", "shap"}.issubset(method_scaling["method"].unique()):
                uf_shap = method_scaling[method_scaling["method"].isin(["usefulness", "shap"])]
                ux.plot_method_scaling(uf_shap, ds, f"{tag}_method_scaling_usefulness_vs_shap")

            model = ux.train_tree(ds, bins=args.bins, strategy="uniform",
                                  seed=args.seed, leaves=cfg.leaves_per_bin[ds] * args.bins)
            # how much of the usefulness runtime was copy.deepcopy overhead
            speedup = pd.DataFrame([ux.profile_scorer_speedup(model)])
            speedup.to_csv(f"{tag}_scorer_speedup.csv", index=False)
            ux.plot_scorer_speedup(speedup, f"{tag}_scorer_speedup")
            print(f"    scorer speedup (deepcopy -> copy-free): {speedup['speedup'].iloc[0]:.1f}x")

        # ---- (C) binning-strategy sensitivity -----------------------------
        if "binning" in args.experiments:
            print("[C] binning-strategy sensitivity ...")
            binning = ux.run_binning_experiment(ds, cfg, n_models=args.n_models)
            for key, frame in binning.items():
                frame.to_csv(f"{tag}_binning_{key}.csv", index=False)
            ux.plot_binning_sensitivity(binning, ds, f"{tag}_binning")
            print("    accuracy by strategy and bins:")
            print(binning["summary"][["strategy", "bins", "accuracy"]]
                  .round(3).to_string(index=False))
            print(f"    cross-strategy ranking agreement (mean, fixed bins), p={cfg.rbo_p}:")
            print(f"      Spearman rho = {binning['stability']['spearman'].mean():.3f}   "
                  f"RBO = {binning['stability']['rbo'].mean():.3f}")
            print(f"    cross-#bins ranking agreement (mean, fixed strategy), p={cfg.rbo_p}:")
            print(f"      Spearman rho = {binning['bins_stability']['spearman'].mean():.3f}   "
                  f"RBO = {binning['bins_stability']['rbo'].mean():.3f}")

    print(f"\nDone. Tables and figures written to '{args.out}/'.")


if __name__ == "__main__":
    main()
