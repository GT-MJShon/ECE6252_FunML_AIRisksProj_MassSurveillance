#!/usr/bin/env python3
"""DARai Exp1 RGB cross-view audit.
Grounded to extracted manifest with columns:
local_path, activity, camera, token1, token2, group_id, frame
No assumption about token semantics beyond grouping.
"""
import argparse
import json
import math
import os
import random
from collections import Counter
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


class ImageDataset(Dataset):
    def __init__(self, df, label_to_idx, transform):
        self.df = df.reset_index(drop=True).copy()
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['local_path']).convert('RGB')
        x = self.transform(img)
        y = self.label_to_idx[row['activity']]
        return x, y, row['local_path']


def build_model(num_classes: int, pretrained: bool = True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def split_groups(groups, val_frac, test_frac, seed):
    groups = list(groups)
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_test = max(1, int(round(n * test_frac))) if n >= 3 else 1
    n_val = max(1, int(round(n * val_frac))) if n - n_test >= 2 else max(0, n - n_test - 1)
    test_g = set(groups[:n_test])
    val_g = set(groups[n_test:n_test+n_val])
    train_g = set(groups[n_test+n_val:])
    if len(train_g) == 0:
        moved = next(iter(val_g or test_g))
        train_g.add(moved)
        if moved in val_g:
            val_g.remove(moved)
        else:
            test_g.remove(moved)
    return train_g, val_g, test_g


def prepare_splits(df, split_mode, train_camera=None, test_camera=None, val_frac=0.15, test_frac=0.2, seed=42):
    if split_mode == 'iid':
        if train_camera is None:
            raise ValueError('--train-camera required for iid')
        d = df[df['camera'] == train_camera].copy()
        groups = sorted(d['group_id'].unique())
        train_g, val_g, test_g = split_groups(groups, val_frac, test_frac, seed)
        train_df = d[d['group_id'].isin(train_g)]
        val_df = d[d['group_id'].isin(val_g)]
        test_df = d[d['group_id'].isin(test_g)]
        desc = {'mode': 'iid', 'camera': train_camera}
        return train_df, val_df, test_df, desc

    if split_mode == 'cross_view':
        if train_camera is None or test_camera is None:
            raise ValueError('--train-camera and --test-camera required for cross_view')
        common = set(df[df['camera'] == train_camera]['group_id'].unique()) & set(df[df['camera'] == test_camera]['group_id'].unique())
        if len(common) < 3:
            raise ValueError(f'Not enough common group_id between {train_camera} and {test_camera}: {len(common)}')
        train_g, val_g, test_g = split_groups(sorted(common), val_frac, test_frac, seed)
        train_df = df[(df['camera'] == train_camera) & (df['group_id'].isin(train_g))].copy()
        val_df = df[(df['camera'] == train_camera) & (df['group_id'].isin(val_g))].copy()
        test_df = df[(df['camera'] == test_camera) & (df['group_id'].isin(test_g))].copy()
        desc = {'mode': 'cross_view', 'train_camera': train_camera, 'test_camera': test_camera}
        return train_df, val_df, test_df, desc

    raise ValueError(f'Unknown split_mode: {split_mode}')


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
    pr, rc, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0)
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
            'activity': name,
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
    ax.bar(df['activity'], df[value_col])
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--split-mode', choices=['iid', 'cross_view'], default='cross_view')
    ap.add_argument('--train-camera', default='camera_1_fps_15')
    ap.add_argument('--test-camera', default='camera_2_fps_15')
    ap.add_argument('--activities', nargs='+', required=True)
    ap.add_argument('--min-common-groups', type=int, default=4)
    ap.add_argument('--img-size', type=int, default=160)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--test-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    df = df[df['activity'].isin(args.activities)].copy()
    df = df[df['local_path'].map(lambda p: isinstance(p, str) and os.path.exists(p))].copy()
    if len(df) == 0:
        raise ValueError('No rows left after filtering. Check manifest and activities.')

    # filter to common group_ids across cameras for cross_view
    if args.split_mode == 'cross_view':
        g1 = set(df[df['camera'] == args.train_camera]['group_id'].unique())
        g2 = set(df[df['camera'] == args.test_camera]['group_id'].unique())
        common = sorted(g1 & g2)
        if len(common) < args.min_common_groups:
            raise ValueError(f'Only {len(common)} common group_id across cameras. Lower --min-common-groups or extract more data.')
        df = df[df['group_id'].isin(common)].copy()

    class_names = sorted(df['activity'].unique())
    label_to_idx = {c: i for i, c in enumerate(class_names)}

    train_df, val_df, test_df, split_desc = prepare_splits(
        df, split_mode=args.split_mode, train_camera=args.train_camera,
        test_camera=args.test_camera, val_frac=args.val_frac,
        test_frac=args.test_frac, seed=args.seed,
    )

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError('Empty split detected.')

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

    train_ds = ImageDataset(train_df, label_to_idx, train_tf)
    val_ds = ImageDataset(val_df if len(val_df) else test_df.sample(min(len(test_df), max(1, len(test_df)//5)), random_state=args.seed), label_to_idx, eval_tf)
    test_ds = ImageDataset(test_df, label_to_idx, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(len(class_names), pretrained=True).to(device)
    class_weights = compute_class_weights(np.array([label_to_idx[x] for x in train_df['activity']]), len(class_names)).to(device)
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
        score = va_m['balanced_accuracy']
        if score > best_val:
            best_val = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_y, te_p, te_paths = run_epoch(model, test_loader, criterion, None, device)
    te_m = evaluate_metrics(te_y, te_p, class_names)

    pd.DataFrame(history).to_csv(outdir / 'history.csv', index=False)
    pd.DataFrame(te_m['per_class']).to_csv(outdir / 'per_class_metrics.csv', index=False)
    pd.DataFrame({
        'local_path': te_paths,
        'y_true': [class_names[i] for i in te_y],
        'y_pred': [class_names[i] for i in te_p],
    }).to_csv(outdir / 'test_predictions.csv', index=False)
    save_confusion_matrix(te_m['confusion_matrix'], class_names, outdir / 'confusion_matrix.png')
    per_class_df = pd.DataFrame(te_m['per_class'])
    save_bar(per_class_df, 'f1', outdir / 'per_class_f1.png', 'Per-class F1')
    save_bar(per_class_df, 'fpr', outdir / 'per_class_fpr.png', 'Per-class FPR')
    save_bar(per_class_df, 'fnr', outdir / 'per_class_fnr.png', 'Per-class FNR')

    payload = {
        'split': split_desc,
        'classes': class_names,
        'n_train': int(len(train_df)),
        'n_val': int(len(val_df)),
        'n_test': int(len(test_df)),
        'group_train': int(train_df['group_id'].nunique()),
        'group_val': int(val_df['group_id'].nunique()),
        'group_test': int(test_df['group_id'].nunique()),
        'test_loss': te_loss,
        'test_accuracy': te_m['accuracy'],
        'test_macro_f1': te_m['macro_f1'],
        'test_balanced_accuracy': te_m['balanced_accuracy'],
    }
    with open(outdir / 'metrics_overall.json', 'w') as f:
        json.dump(payload, f, indent=2)
    torch.save({'state_dict': model.state_dict(), 'classes': class_names}, outdir / 'best_model.pt')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
