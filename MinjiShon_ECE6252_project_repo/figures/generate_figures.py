
import argparse
import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def read_json(zip_path, member):
    with zipfile.ZipFile(zip_path) as z:
        return json.load(z.open(member))


def read_csv(zip_path, member):
    with zipfile.ZipFile(zip_path) as z:
        return pd.read_csv(z.open(member))


def mean_ci(series):
    arr = np.asarray(series, dtype=float)
    mean = arr.mean()
    ci = 0.0 if len(arr) < 2 else 1.96 * arr.std(ddof=1) / math.sqrt(len(arr))
    return mean, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-zip", required=True, help="Path to Exp 1 runs.zip")
    ap.add_argument("--exp2-zip", required=True, help="Path to Exp 2 exp2.zip")
    ap.add_argument("--exp2-condition-csv", required=True, help="Path to exp2_condition_summary.csv")
    ap.add_argument("--exp3-zip", required=True, help="Path to Exp 3 exp3.zip")
    ap.add_argument("--outdir", required=True, help="Directory for output figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    # -------------------------
    # Exp 1: DARai reliability gap
    # -------------------------
    exp1 = {
        "IID cam1": read_json(args.runs_zip, "iid_cam1/metrics_overall.json"),
        "Cross-view cam1→cam2": read_json(args.runs_zip, "cross_view_cam1_to_cam2/metrics_overall.json"),
        "Cross-view cam2→cam1": read_json(args.runs_zip, "cross_view_cam2_to_cam1/metrics_overall.json"),
    }
    exp1_pc = {
        "IID cam1": read_csv(args.runs_zip, "iid_cam1/per_class_metrics.csv"),
        "Cross-view cam1→cam2": read_csv(args.runs_zip, "cross_view_cam1_to_cam2/per_class_metrics.csv"),
        "Cross-view cam2→cam1": read_csv(args.runs_zip, "cross_view_cam2_to_cam1/per_class_metrics.csv"),
    }

    rows = []
    for cond, metrics in exp1.items():
        rows.extend([
            {"condition": cond, "metric": "Accuracy", "value": metrics["test_accuracy"]},
            {"condition": cond, "metric": "Macro-F1", "value": metrics["test_macro_f1"]},
            {"condition": cond, "metric": "Balanced accuracy", "value": metrics["test_balanced_accuracy"]},
            {"condition": cond, "metric": "Mean FNR", "value": float(exp1_pc[cond]["fnr"].mean())},
            {"condition": cond, "metric": "Mean FPR", "value": float(exp1_pc[cond]["fpr"].mean())},
        ])
    df1 = pd.DataFrame(rows)

    plt.figure(figsize=(10.5, 5.8))
    ax = sns.barplot(data=df1, x="metric", y="value", hue="condition")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Figure 1. DARai reliability gap under camera shift")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(outdir / "figure1_darai_reliability_gap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Exp 2: CURE failure concentration
    # -------------------------
    cure_pc = {
        "ChallengeFree": read_csv(args.exp2_zip, "cure_runs/baseline_cf/ChallengeFree/per_class_metrics.csv"),
        "Darkening-5": read_csv(args.exp2_zip, "cure_runs/baseline_cf/Darkening-5/per_class_metrics.csv"),
        "CodecError-5": read_csv(args.exp2_zip, "cure_runs/baseline_cf/CodecError-5/per_class_metrics.csv"),
    }
    line_rows = []
    for cond, sev, family in [
        ("ChallengeFree", 0, "Darkening"),
        ("Darkening-5", 5, "Darkening"),
        ("ChallengeFree", 0, "CodecError"),
        ("CodecError-5", 5, "CodecError"),
    ]:
        mean, ci = mean_ci(cure_pc[cond]["fnr"])
        line_rows.append({
            "family": family,
            "severity": sev,
            "mean_fnr": mean,
            "ci95": ci,
            "condition": cond,
        })
    df2 = pd.DataFrame(line_rows)

    plt.figure(figsize=(8.8, 5.6))
    for family, sub in df2.groupby("family"):
        sub = sub.sort_values("severity")
        plt.errorbar(
            sub["severity"], sub["mean_fnr"], yerr=sub["ci95"],
            marker="o", capsize=4, linewidth=2.5, label=family
        )
    plt.ylim(0, 1.05)
    plt.xlim(-0.2, 5.2)
    plt.xticks([0, 5], ["0\nChallengeFree", "5\nSevere"])
    plt.ylabel("Mean per-class FNR")
    plt.xlabel("Degradation severity endpoint available in outputs")
    plt.title("Figure 2. Failure concentration in CURE-TSR under severe degradation")
    plt.legend(title="Stress family")
    plt.tight_layout()
    plt.savefig(outdir / "figure2_cure_failure_concentration.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------
    # Exp 3: scale-harm paradox
    # -------------------------
    with zipfile.ZipFile(args.exp3_zip) as z:
        exp3_scaling = pd.read_csv(z.open("exp3_population_scaling.csv"))
    stress = exp3_scaling[exp3_scaling["system"].isin(["DARai_cross_view", "CURE_Darkening5"])].copy()
    stress["display"] = stress["system"].map({
        "DARai_cross_view": "DARai cross-view",
        "CURE_Darkening5": "CURE Darkening-5",
    })

    palette = sns.color_palette()[:2]
    fig, ax1 = plt.subplots(figsize=(9.4, 5.8))
    for color, (name, sub) in zip(palette, stress.groupby("display")):
        sub = sub.sort_values("population")
        ax1.plot(
            sub["population"], sub["expected_fp"],
            marker="o", linewidth=2.5, label=f"{name}: false positives", color=color
        )
    ax1.set_xscale("log")
    ax1.set_xlabel("Population screened (log scale)")
    ax1.set_ylabel("Expected false positives")
    ax1.ticklabel_format(axis="y", style="plain")

    ax2 = ax1.twinx()
    for color, (name, sub) in zip(palette, stress.groupby("display")):
        sub = sub.sort_values("population")
        ax2.plot(
            sub["population"], 1 - sub["prob_flagged_is_false_positive"],
            marker="s", linestyle="--", linewidth=2.0, label=f"{name}: PPV", color=color
        )
    ax2.set_ylabel("Precision / PPV among flagged people")
    ax2.set_ylim(0, max((1 - stress["prob_flagged_is_false_positive"]).max() * 1.15, 0.05))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    plt.title("Figure 3. Scale-harm paradox: false alerts explode as screening expands")
    plt.tight_layout()
    plt.savefig(outdir / "figure3_scale_harm_paradox.png", dpi=300, bbox_inches="tight")
    plt.close()

    # optional summary table
    exp2_condition = pd.read_csv(args.exp2_condition_csv)
    summary = {
        "exp1_conditions": list(exp1.keys()),
        "exp2_conditions": exp2_condition["condition"].tolist(),
        "exp3_systems": sorted(stress["display"].unique().tolist()),
    }
    with open(outdir / "figure_generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
