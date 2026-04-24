#!/usr/bin/env python3
"""Unified robustness audit for DARai and CURE-TSR.

Important limitation:
- A DARai activity-recognition checkpoint from Exp 1 cannot be validly applied to CURE-TSR traffic-sign classes.
- This script therefore loads TWO task-specific checkpoints:
    1) DARai checkpoint from Exp 1 for activity recognition.
    2) A separate CURE-TSR checkpoint for sign recognition.

Both branches use the same evaluation and reporting logic so the clean-vs-stress instability is directly comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from cure_tsr_dataset import CURETSRDataset, discover_cure_records, write_manifest, DEFAULT_CURE_ROOT

DEFAULT_DARAI_ROOT = "/home/hice1/mshon6/scratch/ECE6252_project/Dataset/DARai/RGB_pt1"
DEFAULT_CURE_ROOT = DEFAULT_CURE_ROOT


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GenericImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_col: str, path_col: str, class_to_idx: Dict[str, int], transform):
        self.df = df.reset_index(drop=True).copy()
        self.label_col = label_col
        self.path_col = path_col
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row[self.path_col]).convert("RGB")
        x = self.transform(img)
        y = self.class_to_idx[str(row[self.label_col])]
        meta = dict(row)
        return x, y, meta


def build_resnet18(num_classes: int, pretrained: bool = False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and "classes" in ckpt:
        classes = list(ckpt["classes"])
        state_dict = ckpt["state_dict"]
    else:
        raise ValueError(
            f"Checkpoint format not supported: {checkpoint_path}. Expected a dict with keys 'state_dict' and 'classes'."
        )
    model = build_resnet18(len(classes), pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, classes


def split_groups(groups: Sequence[str], val_frac: float, test_frac: float, seed: int):
    groups = list(groups)
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_test = max(1, int(round(n * test_frac))) if n >= 3 else 1
    n_val = max(1, int(round(n * val_frac))) if n - n_test >= 2 else max(0, n - n_test - 1)
    test_g = set(groups[:n_test])
    val_g = set(groups[n_test:n_test+n_val])
    train_g = set(groups[n_test+n_val:])
    if not train_g:
        moved = next(iter(val_g or test_g))
        train_g.add(moved)
        if moved in val_g:
            val_g.remove(moved)
        else:
            test_g.remove(moved)
    return train_g, val_g, test_g


def prepare_darai_clean_and_stress(manifest_csv: str, activities: Sequence[str], clean_camera: str, stress_camera: str, seed: int, val_frac: float, test_frac: float):
    df = pd.read_csv(manifest_csv)
    req_cols = {"local_path", "activity", "camera", "group_id"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"DARai manifest missing required columns: {sorted(missing)}")
    df = df[df["activity"].isin(list(activities))].copy()
    df = df[df["local_path"].map(lambda p: isinstance(p, str) and os.path.exists(p))].copy()
    common_groups = sorted(set(df[df["camera"] == clean_camera]["group_id"].unique()) & set(df[df["camera"] == stress_camera]["group_id"].unique()))
    if len(common_groups) < 3:
        raise RuntimeError(f"Not enough common group_id between {clean_camera} and {stress_camera}: {len(common_groups)}")
    train_g, val_g, test_g = split_groups(common_groups, val_frac=val_frac, test_frac=test_frac, seed=seed)
    clean_test = df[(df["camera"] == clean_camera) & (df["group_id"].isin(test_g))].copy()
    stress_test = df[(df["camera"] == stress_camera) & (df["group_id"].isin(test_g))].copy()
    return clean_test, stress_test


def run_eval(model, loader, criterion, device, label_names: Sequence[str], label_field: str):
    model.eval()
    total_loss, total_n = 0.0, 0
    ys, ps = [], []
    metas = []
    with torch.no_grad():
        for x, y, batch_meta in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)
            pred = logits.argmax(dim=1).cpu().numpy()
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(pred.tolist())
            if isinstance(batch_meta, dict):
                # default PyTorch collate creates dict of lists.
                batch_size = len(next(iter(batch_meta.values())))
                for i in range(batch_size):
                    metas.append({k: batch_meta[k][i] for k in batch_meta})
            else:
                metas.extend(list(batch_meta))

    ys = np.asarray(ys)
    ps = np.asarray(ps)
    cm = confusion_matrix(ys, ps, labels=np.arange(len(label_names)))
    pr, rc, f1, support = precision_recall_fscore_support(ys, ps, labels=np.arange(len(label_names)), zero_division=0)
    total = cm.sum()
    per_class_rows = []
    for i, label in enumerate(label_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        per_class_rows.append({
            label_field: label,
            "precision": float(pr[i]),
            "recall": float(rc[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "fpr": float(fpr),
            "fnr": float(fnr),
        })
    result = {
        "loss": total_loss / max(1, total_n),
        "accuracy": float((ys == ps).mean()),
        "macro_f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(ys, ps)),
        "mean_fnr_macro": float(np.mean([r["fnr"] for r in per_class_rows])),
        "mean_fpr_macro": float(np.mean([r["fpr"] for r in per_class_rows])),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class_rows,
        "pred_df": pd.DataFrame({
            "y_true": [label_names[i] for i in ys],
            "y_pred": [label_names[i] for i in ps],
        }).assign(**pd.DataFrame(metas) if metas else {}),
    }
    return result


def save_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_per_class_bar(df: pd.DataFrame, value_col: str, title: str, out_path: Path, label_field: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df[label_field], df[value_col])
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_loader(df: pd.DataFrame, label_col: str, path_col: str, classes: Sequence[str], img_size: int, batch_size: int, num_workers: int):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    ds = GenericImageDataset(df, label_col=label_col, path_col=path_col, class_to_idx=class_to_idx, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def evaluate_darai(args, outdir: Path, device: torch.device):
    model, classes = load_checkpoint_model(args.darai_checkpoint, device)
    clean_df, stress_df = prepare_darai_clean_and_stress(
        args.darai_manifest,
        activities=args.darai_activities,
        clean_camera=args.darai_clean_camera,
        stress_camera=args.darai_stress_camera,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    clean_loader = make_loader(clean_df, "activity", "local_path", classes, args.img_size, args.batch_size, args.num_workers)
    stress_loader = make_loader(stress_df, "activity", "local_path", classes, args.img_size, args.batch_size, args.num_workers)
    criterion = nn.CrossEntropyLoss()
    clean = run_eval(model, clean_loader, criterion, device, classes, label_field="activity")
    stress = run_eval(model, stress_loader, criterion, device, classes, label_field="activity")

    darai_dir = outdir / "darai"
    darai_dir.mkdir(parents=True, exist_ok=True)
    for name, result in [("clean_iid", clean), ("stress_cross_view", stress)]:
        per_class_df = pd.DataFrame(result["per_class"])
        per_class_df.to_csv(darai_dir / f"{name}_per_class.csv", index=False)
        result["pred_df"].to_csv(darai_dir / f"{name}_predictions.csv", index=False)
        save_confusion_matrix(np.asarray(result["confusion_matrix"]), classes, darai_dir / f"{name}_confusion_matrix.png")
        save_per_class_bar(per_class_df, "fnr", f"DARai {name} per-class FNR", darai_dir / f"{name}_per_class_fnr.png", "activity")
        save_per_class_bar(per_class_df, "fpr", f"DARai {name} per-class FPR", darai_dir / f"{name}_per_class_fpr.png", "activity")

    summary = {
        "clean_iid": {k: v for k, v in clean.items() if k not in {"per_class", "pred_df", "confusion_matrix"}},
        "stress_cross_view": {k: v for k, v in stress.items() if k not in {"per_class", "pred_df", "confusion_matrix"}},
    }
    summary["disparity"] = {
        "accuracy_drop": summary["clean_iid"]["accuracy"] - summary["stress_cross_view"]["accuracy"],
        "balanced_accuracy_drop": summary["clean_iid"]["balanced_accuracy"] - summary["stress_cross_view"]["balanced_accuracy"],
        "macro_f1_drop": summary["clean_iid"]["macro_f1"] - summary["stress_cross_view"]["macro_f1"],
        "mean_fnr_macro_increase": summary["stress_cross_view"]["mean_fnr_macro"] - summary["clean_iid"]["mean_fnr_macro"],
        "mean_fpr_macro_increase": summary["stress_cross_view"]["mean_fpr_macro"] - summary["clean_iid"]["mean_fpr_macro"],
    }
    return summary


def ensure_cure_manifest(args) -> str:
    if args.cure_manifest:
        return args.cure_manifest
    out_csv = str(Path(args.outdir) / "cure_manifest.csv")
    rows = discover_cure_records(
        root=args.cure_root,
        split_filter=[args.cure_split],
        label_mode=args.cure_label_mode,
        label_regex=args.cure_label_regex,
    )
    write_manifest(rows, out_csv)
    return out_csv


def evaluate_cure_condition(model, manifest_csv: str, classes: Sequence[str], split_name: str, condition_folder: str, img_size: int, batch_size: int, num_workers: int, device: torch.device):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = CURETSRDataset(
        manifest_csv=manifest_csv,
        split_filter=[split_name],
        condition_filter=[condition_folder],
        class_to_idx={c: i for i, c in enumerate(classes)},
        transform=tf,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    return run_eval(model, loader, criterion, device, classes, label_field="label")


def evaluate_cure(args, outdir: Path, device: torch.device):
    manifest_csv = ensure_cure_manifest(args)
    model, classes = load_checkpoint_model(args.cure_checkpoint, device)
    conditions = [args.cure_clean_condition] + list(args.cure_stress_conditions)
    cure_dir = outdir / "cure_tsr"
    cure_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for condition in conditions:
        result = evaluate_cure_condition(
            model=model,
            manifest_csv=manifest_csv,
            classes=classes,
            split_name=args.cure_split,
            condition_folder=condition,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        results[condition] = result
        per_class_df = pd.DataFrame(result["per_class"])
        per_class_df.to_csv(cure_dir / f"{condition}_per_class.csv", index=False)
        result["pred_df"].to_csv(cure_dir / f"{condition}_predictions.csv", index=False)
        save_confusion_matrix(np.asarray(result["confusion_matrix"]), classes, cure_dir / f"{condition}_confusion_matrix.png")
        save_per_class_bar(per_class_df, "fnr", f"CURE-TSR {condition} per-class FNR", cure_dir / f"{condition}_per_class_fnr.png", "label")
        save_per_class_bar(per_class_df, "fpr", f"CURE-TSR {condition} per-class FPR", cure_dir / f"{condition}_per_class_fpr.png", "label")

    clean = results[args.cure_clean_condition]
    stress_summary = {}
    for condition in args.cure_stress_conditions:
        stress = results[condition]
        stress_summary[condition] = {
            **{k: v for k, v in stress.items() if k not in {"per_class", "pred_df", "confusion_matrix"}},
            "accuracy_drop_vs_clean": clean["accuracy"] - stress["accuracy"],
            "balanced_accuracy_drop_vs_clean": clean["balanced_accuracy"] - stress["balanced_accuracy"],
            "macro_f1_drop_vs_clean": clean["macro_f1"] - stress["macro_f1"],
            "mean_fnr_macro_increase_vs_clean": stress["mean_fnr_macro"] - clean["mean_fnr_macro"],
            "mean_fpr_macro_increase_vs_clean": stress["mean_fpr_macro"] - clean["mean_fpr_macro"],
        }
    summary = {
        "clean": {args.cure_clean_condition: {k: v for k, v in clean.items() if k not in {"per_class", "pred_df", "confusion_matrix"}}},
        "stress": stress_summary,
        "manifest_csv": manifest_csv,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.20)

    # DARai branch
    ap.add_argument("--darai-manifest", default=f"{DEFAULT_DARAI_ROOT}/exp1_rgb_baseline/subset_rgb/manifest.csv")
    ap.add_argument("--darai-checkpoint", required=True)
    ap.add_argument("--darai-clean-camera", default="camera_1_fps_15")
    ap.add_argument("--darai-stress-camera", default="camera_2_fps_15")
    ap.add_argument("--darai-activities", nargs="+", default=["Making pancake", "Reading", "Working on a computer"])

    # CURE branch
    ap.add_argument("--cure-root", default=DEFAULT_CURE_ROOT)
    ap.add_argument("--cure-manifest", default=None)
    ap.add_argument("--cure-checkpoint", required=True)
    ap.add_argument("--cure-split", default="Real_Test")
    ap.add_argument("--cure-clean-condition", default="ChallengeFree")
    ap.add_argument("--cure-stress-conditions", nargs="+", default=["Darkening-5", "CodecError-5"])
    ap.add_argument("--cure-label-mode", choices=["auto", "condition_child", "filename_regex"], default="auto")
    ap.add_argument("--cure-label-regex", default=None)

    args = ap.parse_args()
    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate DARai
    darai_summary = evaluate_darai(args, outdir, device)

    # Evaluate CURE-TSR
    cure_summary = evaluate_cure(args, outdir, device)

    summary = {
        "device": str(device),
        "darai": darai_summary,
        "cure_tsr": cure_summary,
    }
    with open(outdir / "exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    rows = []
    rows.append({"dataset": "DARai", "condition": "clean_iid", **darai_summary["clean_iid"]})
    rows.append({"dataset": "DARai", "condition": "stress_cross_view", **darai_summary["stress_cross_view"]})
    rows.append({"dataset": "CURE-TSR", "condition": args.cure_clean_condition, **cure_summary["clean"][args.cure_clean_condition]})
    for cond, vals in cure_summary["stress"].items():
        rows.append({"dataset": "CURE-TSR", "condition": cond, **{k: v for k, v in vals.items() if not k.endswith("_vs_clean")}})
    pd.DataFrame(rows).to_csv(outdir / "exp2_condition_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
