#!/usr/bin/env python3
"""Train a simple CURE-TSR baseline on ChallengeFree and optionally evaluate on challenge conditions.
Grounded to manifest columns:
absolute_path, relative_path, split, condition_folder, challenge_type, level, label
"""
import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ManifestImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict, transform):
        self.df = df.reset_index(drop=True).copy()
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["absolute_path"]).convert("RGB")
        x = self.transform(img)
        y = self.label_to_idx[str(row["label"])]
        return x, y, row["absolute_path"]


def build_model(num_classes: int, pretrained: bool = True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device):
    train = optimizer is not None
    model.train(train)
    total_loss, total_n = 0.0, 0
    ys, ps, paths = [], [], []
    for x, y, p in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        ys.extend(y.detach().cpu().numpy().tolist())
        ps.extend(pred.tolist())
        paths.extend(list(p))
    return total_loss / max(1, total_n), np.array(ys), np.array(ps), paths


def evaluate_metrics(y_true, y_pred, class_names):
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    pr, rc, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0
    )
    rows = []
    total = cm.sum()
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        rows.append({
            'label': name,
            'precision': float(pr[i]),
            'recall': float(rc[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
            'fpr': float(fpr),
            'fnr': float(fnr),
        })
    return {
        'accuracy': acc,
        'macro_f1': float(macro_f1),
        'balanced_accuracy': float(bal_acc),
        'confusion_matrix': cm,
        'per_class': rows,
    }


def save_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_bar(df, value_col, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df['label'].astype(str), df[value_col])
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def build_subset(df: pd.DataFrame, conditions):
    out = df[df['condition_folder'].isin(conditions)].copy()
    out['label'] = out['label'].astype(str)
    out = out[out['absolute_path'].map(lambda p: isinstance(p, str) and os.path.exists(p))].copy()
    return out


def stratified_group_split(df: pd.DataFrame, val_frac: float, seed: int):
    # group by label, sample validation per label to keep all classes present when possible
    rng = random.Random(seed)
    val_indices = []
    for label, part in df.groupby('label'):
        idxs = list(part.index)
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_frac))) if len(idxs) >= 3 else 1 if len(idxs) >= 2 else 0
        val_indices.extend(idxs[:n_val])
    val_df = df.loc[sorted(set(val_indices))].copy()
    train_df = df.drop(index=sorted(set(val_indices))).copy()
    return train_df, val_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-manifest', required=True)
    ap.add_argument('--eval-manifest', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--train-conditions', nargs='+', default=['ChallengeFree'])
    ap.add_argument('--eval-conditions', nargs='+', default=['ChallengeFree', 'Darkening-5', 'CodecError-5'])
    ap.add_argument('--img-size', type=int, default=160)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-pretrained', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_manifest = pd.read_csv(args.train_manifest)
    eval_manifest = pd.read_csv(args.eval_manifest)
    train_base = build_subset(train_manifest, args.train_conditions)
    eval_base = build_subset(eval_manifest, args.eval_conditions)
    if len(train_base) == 0:
        raise ValueError('No training rows after filtering train manifest.')
    if len(eval_base) == 0:
        raise ValueError('No evaluation rows after filtering eval manifest.')

    common_labels = sorted(set(train_base['label'].unique()) & set(eval_base['label'].unique()))
    train_base = train_base[train_base['label'].isin(common_labels)].copy()
    eval_base = eval_base[eval_base['label'].isin(common_labels)].copy()
    class_names = sorted(common_labels)
    label_to_idx = {c: i for i, c in enumerate(class_names)}

    train_df, val_df = stratified_group_split(train_base, args.val_frac, args.seed)

    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ManifestImageDataset(train_df, label_to_idx, train_tf)
    val_ds = ManifestImageDataset(val_df, label_to_idx, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(len(class_names), pretrained=not args.no_pretrained).to(device)
    class_weights = compute_class_weights(np.array([label_to_idx[x] for x in train_df['label']]), len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state = None
    best_val = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_y, tr_p, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_y, va_p, _ = run_epoch(model, val_loader, criterion, None, device)
        tr_m = evaluate_metrics(tr_y, tr_p, class_names)
        va_m = evaluate_metrics(va_y, va_p, class_names)
        row = {
            'epoch': epoch,
            'train_loss': tr_loss,
            'val_loss': va_loss,
            'train_macro_f1': tr_m['macro_f1'],
            'val_macro_f1': va_m['macro_f1'],
            'train_balanced_accuracy': tr_m['balanced_accuracy'],
            'val_balanced_accuracy': va_m['balanced_accuracy'],
        }
        history.append(row)
        print(row)
        if va_m['balanced_accuracy'] > best_val:
            best_val = va_m['balanced_accuracy']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    pd.DataFrame(history).to_csv(outdir / 'history.csv', index=False)
    torch.save({'state_dict': model.state_dict(), 'classes': class_names}, outdir / 'best_model.pt')

    summary = {
        'train_conditions': args.train_conditions,
        'eval_conditions': args.eval_conditions,
        'classes': class_names,
        'n_train': int(len(train_df)),
        'n_val': int(len(val_df)),
        'n_eval_total': int(len(eval_base)),
        'per_condition': {},
    }

    for cond in args.eval_conditions:
        cond_df = eval_base[eval_base['condition_folder'] == cond].copy()
        if len(cond_df) == 0:
            continue
        cond_ds = ManifestImageDataset(cond_df, label_to_idx, eval_tf)
        cond_loader = DataLoader(cond_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        te_loss, te_y, te_p, te_paths = run_epoch(model, cond_loader, criterion, None, device)
        te_m = evaluate_metrics(te_y, te_p, class_names)
        cond_dir = outdir / cond
        cond_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(te_m['per_class']).to_csv(cond_dir / 'per_class_metrics.csv', index=False)
        pd.DataFrame({
            'absolute_path': te_paths,
            'y_true': [class_names[i] for i in te_y],
            'y_pred': [class_names[i] for i in te_p],
        }).to_csv(cond_dir / 'predictions.csv', index=False)
        save_confusion_matrix(te_m['confusion_matrix'], class_names, cond_dir / 'confusion_matrix.png')
        per_class_df = pd.DataFrame(te_m['per_class'])
        save_bar(per_class_df, 'f1', cond_dir / 'per_class_f1.png', f'{cond} per-class F1')
        save_bar(per_class_df, 'fpr', cond_dir / 'per_class_fpr.png', f'{cond} per-class FPR')
        save_bar(per_class_df, 'fnr', cond_dir / 'per_class_fnr.png', f'{cond} per-class FNR')
        summary['per_condition'][cond] = {
            'n': int(len(cond_df)),
            'loss': te_loss,
            'accuracy': te_m['accuracy'],
            'macro_f1': te_m['macro_f1'],
            'balanced_accuracy': te_m['balanced_accuracy'],
            'mean_fpr': float(np.mean([r['fpr'] for r in te_m['per_class']])),
            'mean_fnr': float(np.mean([r['fnr'] for r in te_m['per_class']])),
        }

    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
