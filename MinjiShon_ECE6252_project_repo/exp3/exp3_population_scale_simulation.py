#!/usr/bin/env python3
"""
Experiment 3: Population-scale social harm simulation.
Lightweight CPU-only script.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_exp2_rates(path: Path):
    data = json.loads(path.read_text())
    darai = data["darai"]["stress_cross_view"]
    cure = data["cure_tsr"]["stress"]["Darkening-5"]
    clean_darai = data["darai"]["clean_iid"]
    clean_cure = data["cure_tsr"]["clean"]["ChallengeFree"]
    return {
        "DARai_cross_view": {
            "fpr": float(darai["mean_fpr_macro"]),
            "fnr": float(darai["mean_fnr_macro"]),
            "accuracy": float(darai["accuracy"]),
            "balanced_accuracy": float(darai["balanced_accuracy"]),
            "macro_f1": float(darai["macro_f1"]),
        },
        "CURE_Darkening5": {
            "fpr": float(cure["mean_fpr_macro"]),
            "fnr": float(cure["mean_fnr_macro"]),
            "accuracy": float(cure["accuracy"]),
            "balanced_accuracy": float(cure["balanced_accuracy"]),
            "macro_f1": float(cure["macro_f1"]),
        },
        "DARai_clean_iid": {
            "fpr": float(clean_darai["mean_fpr_macro"]),
            "fnr": float(clean_darai["mean_fnr_macro"]),
            "accuracy": float(clean_darai["accuracy"]),
            "balanced_accuracy": float(clean_darai["balanced_accuracy"]),
            "macro_f1": float(clean_darai["macro_f1"]),
        },
        "CURE_ChallengeFree": {
            "fpr": float(clean_cure["mean_fpr_macro"]),
            "fnr": float(clean_cure["mean_fnr_macro"]),
            "accuracy": float(clean_cure["accuracy"]),
            "balanced_accuracy": float(clean_cure["balanced_accuracy"]),
            "macro_f1": float(clean_cure["macro_f1"]),
        },
    }


def exact_outcomes(population: int, true_targets: int, fpr: float, fnr: float):
    positives = population - true_targets
    tpr = 1.0 - fnr

    exp_tp = true_targets * tpr
    exp_fn = true_targets * fnr
    exp_fp = positives * fpr
    exp_tn = positives * (1.0 - fpr)

    total_flagged = exp_tp + exp_fp
    ppv = exp_tp / total_flagged if total_flagged > 0 else 0.0
    prob_flagged_innocent = exp_fp / total_flagged if total_flagged > 0 else 0.0

    return {
        "population": int(population),
        "true_targets": int(true_targets),
        "non_targets": int(positives),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tpr": float(tpr),
        "expected_tp": float(exp_tp),
        "expected_fn": float(exp_fn),
        "expected_fp": float(exp_fp),
        "expected_tn": float(exp_tn),
        "expected_total_flagged": float(total_flagged),
        "precision_ppv": float(ppv),
        "prob_flagged_is_false_positive": float(prob_flagged_innocent),
        "false_alerts_per_true_positive": float(exp_fp / exp_tp) if exp_tp > 0 else math.inf,
    }


def monte_carlo(population: int, true_targets: int, fpr: float, fnr: float, trials: int, seed: int):
    rng = np.random.default_rng(seed)
    positives = population - true_targets
    tpr = 1.0 - fnr

    tp = rng.binomial(true_targets, tpr, size=trials)
    fp = rng.binomial(positives, fpr, size=trials)
    flagged = tp + fp
    innocent_share = np.divide(fp, flagged, out=np.zeros_like(fp, dtype=float), where=flagged > 0)
    ppv = np.divide(tp, flagged, out=np.zeros_like(tp, dtype=float), where=flagged > 0)

    def q(x):
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "p05": float(np.quantile(x, 0.05)),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
        }

    return {
        "tp": q(tp),
        "fp": q(fp),
        "flagged": q(flagged),
        "innocent_share_among_flagged": q(innocent_share),
        "ppv": q(ppv),
    }


def plot_false_alerts_vs_population(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(7, 4.5))
    for system, g in df.groupby("system"):
        g = g.sort_values("population")
        plt.plot(g["population"], g["expected_fp"], marker="o", label=system)
    plt.xlabel("Population size")
    plt.ylabel("Expected false alerts")
    plt.title("Social cost grows with deployment scale")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_innocent_share_vs_population(df: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(7, 4.5))
    for system, g in df.groupby("system"):
        g = g.sort_values("population")
        plt.plot(g["population"], g["prob_flagged_is_false_positive"], marker="o", label=system)
    plt.xlabel("Population size")
    plt.ylabel("P(flagged person is innocent)")
    plt.title("Most flagged people are innocent at low prevalence")
    plt.xscale("log")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_social_cost_vs_accuracy(rates: dict, population: int, true_targets: int, out_path: Path):
    plt.figure(figsize=(7, 4.5))
    for name, d in rates.items():
        r = exact_outcomes(population, true_targets, d["fpr"], d["fnr"])
        plt.scatter(d["accuracy"], r["expected_fp"], s=80, label=name)
        plt.annotate(name, (d["accuracy"], r["expected_fp"]), textcoords="offset points", xytext=(5, 5), fontsize=8)

    sweep = np.linspace(0.001, 0.20, 200)
    for label, fnr in [("DARai-style FNR fixed", rates["DARai_cross_view"]["fnr"]),
                       ("CURE Darkening-5 FNR fixed", rates["CURE_Darkening5"]["fnr"] )]:
        acc = []
        fp = []
        for fpr in sweep:
            prevalence = true_targets / population
            accuracy = prevalence * (1 - fnr) + (1 - prevalence) * (1 - fpr)
            r = exact_outcomes(population, true_targets, fpr, fnr)
            acc.append(accuracy)
            fp.append(r["expected_fp"])
        plt.plot(acc, fp, alpha=0.8, label=label)

    plt.xlabel(f"Deployment accuracy at N={population:,}, targets={true_targets}")
    plt.ylabel("Expected false alerts")
    plt.title("Social cost vs. model accuracy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_rows(rates, populations, true_targets):
    rows = []
    for system, d in rates.items():
        for N in populations:
            r = exact_outcomes(N, true_targets, d["fpr"], d["fnr"])
            r["system"] = system
            r["reported_model_accuracy"] = d["accuracy"]
            r["reported_balanced_accuracy"] = d["balanced_accuracy"]
            r["reported_macro_f1"] = d["macro_f1"]
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp2-summary", default="/home/hice1/mshon6/scratch/ECE6252_project/exp2/results/exp2_summary.json")
    ap.add_argument("--outdir", default="/home/hice1/mshon6/scratch/ECE6252_project/exp3")
    ap.add_argument("--population", type=int, default=1_000_000)
    ap.add_argument("--true-targets", type=int, default=10)
    ap.add_argument("--populations", nargs="*", type=int, default=[10_000, 100_000, 1_000_000, 10_000_000])
    ap.add_argument("--trials", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rates = load_exp2_rates(Path(args.exp2_summary))

    main_rows = []
    mc = {}
    for system, d in rates.items():
        exact = exact_outcomes(args.population, args.true_targets, d["fpr"], d["fnr"])
        exact["system"] = system
        exact["reported_model_accuracy"] = d["accuracy"]
        exact["reported_balanced_accuracy"] = d["balanced_accuracy"]
        exact["reported_macro_f1"] = d["macro_f1"]
        main_rows.append(exact)
        mc[system] = monte_carlo(args.population, args.true_targets, d["fpr"], d["fnr"], args.trials, args.seed)

    main_df = pd.DataFrame(main_rows).sort_values("system")
    scale_df = build_rows(rates, args.populations, args.true_targets)

    main_df.to_csv(outdir / "exp3_main_scenarios.csv", index=False)
    scale_df.to_csv(outdir / "exp3_population_scaling.csv", index=False)
    with open(outdir / "exp3_monte_carlo.json", "w") as f:
        json.dump(mc, f, indent=2)
    with open(outdir / "exp3_rates_used.json", "w") as f:
        json.dump(rates, f, indent=2)

    plot_false_alerts_vs_population(scale_df, outdir / "false_alerts_vs_population.png")
    plot_innocent_share_vs_population(scale_df, outdir / "innocent_share_vs_population.png")
    plot_social_cost_vs_accuracy(rates, args.population, args.true_targets, outdir / "social_cost_vs_accuracy.png")

    lines = []
    lines.append("Experiment 3 summary")
    lines.append(f"Population={args.population:,}, true_targets={args.true_targets}")
    lines.append("")
    for _, row in main_df.iterrows():
        lines.append(f"[{row['system']}]")
        lines.append(f"reported model accuracy={row['reported_model_accuracy']:.4f}")
        lines.append(f"FPR={row['fpr']:.6f}, FNR={row['fnr']:.6f}")
        lines.append(f"expected false alerts={row['expected_fp']:.1f}")
        lines.append(f"expected true alerts={row['expected_tp']:.3f}")
        lines.append(f"P(flagged person is innocent)={row['prob_flagged_is_false_positive']:.6f}")
        lines.append(f"false alerts per true positive={row['false_alerts_per_true_positive']:.1f}")
        lines.append("")
    (outdir / "exp3_readme.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
