#!/usr/bin/env python3
"""
Exp 1 extension: clip-based activity recognition on DARai RGB.
Two models selectable via --model:
  resnet18_clip_avg  -- ResNet-18 per-frame features, average-pooled over clip, then classified
  r3d18              -- R3D-18 (3D ResNet, torchvision) trained end-to-end on 16-frame clips

Input: same manifest CSV as darai_exp1_train.py
       columns: local_path, activity, camera, group_id, frame

Output (same format as darai_exp1_train.py for direct comparison):
  metrics_overall.json
  per_class_metrics.csv
  test_predictions.csv
  confusion_matrix.png
  per_class_f1.png / per_class_fpr.png / per_class_fnr.png
  history.csv
"""
import argparse
import json
import os
import random
from collections import defaultdict
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
from torchvision.models.video import r3d_18, R3D_18_Weights


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Clip building
# ---------------------------------------------------------------------------
def build_clips(df: pd.DataFrame, clip_len: int, stride: int):
    """
    Given a manifest DataFrame, group frames by (activity, camera, group_id),
    sort by frame number, and slide a window of clip_len with given stride.
    Returns a list of dicts: {activity, camera, group_id, frame_paths, label}
    """
    clips = []
    grouped = df.groupby(['activity', 'camera', 'group_id'])
    for (activity, camera, group_id), group in grouped:
        group_sorted = group.sort_values('frame').reset_index(drop=True)
        paths = group_sorted['local_path'].tolist()
        n = len(paths)
        if n < clip_len:
            # pad by repeating last frame
            paths = paths + [paths[-1]] * (clip_len - n)
            n = clip_len
        for start in range(0, n - clip_len + 1, stride):
            clips.append({
                'activity': activity,
                'camera': camera,
                'group_id': group_id,
                'frame_paths': paths[start:start + clip_len],
            })
    return clips


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class ClipAvgDataset(Dataset):
    """Loads clip_len frames individually; collates to (clip_len, C, H, W).
    Used by ResNet-18 clip-average model."""
    def __init__(self, clips, label_to_idx, transform):
        self.clips = clips
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        frames = torch.stack([
            self.transform(Image.open(p).convert('RGB'))
            for p in clip['frame_paths']
        ])  # (T, C, H, W)
        label = self.label_to_idx[clip['activity']]
        return frames, label, clip['group_id']


class R3DDataset(Dataset):
    """Loads clip_len frames and stacks to (C, T, H, W) for R3D-18."""
    def __init__(self, clips, label_to_idx, transform):
        self.clips = clips
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        frames = torch.stack([
            self.transform(Image.open(p).convert('RGB'))
            for p in clip['frame_paths']
        ])  # (T, C, H, W)
        video = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        label = self.label_to_idx[clip['activity']]
        return video, label, clip['group_id']


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ResNet18ClipAvg(nn.Module):
    """ResNet-18 backbone; average-pool frame features over clip dimension."""
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        # remove final FC to get feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.backbone(x)           # (B*T, feature_dim, 1, 1)
        feat = feat.view(B, T, -1)        # (B, T, feature_dim)
        feat = feat.mean(dim=1)           # (B, feature_dim) — temporal avg pool
        return self.fc(feat)


def build_r3d18(num_classes: int, pretrained: bool = True):
    weights = R3D_18_Weights.DEFAULT if pretrained else None
    model = r3d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Splits — identical logic to darai_exp1_train.py
# ---------------------------------------------------------------------------
def split_groups(groups, val_frac, test_frac, seed):
    groups = list(groups)
    rng = random.Random(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_test = max(1, int(round(n * test_frac))) if n >= 3 else 1
    n_val = max(1, int(round(n * val_frac))) if n - n_test >= 2 else max(0, n - n_test - 1)
    test_g = set(groups[:n_test])
    val_g = set(groups[n_test:n_test + n_val])
    train_g = set(groups[n_test + n_val:])
    if not train_g:
        moved = next(iter(val_g or test_g))
        train_g.add(moved)
        (val_g if moved in val_g else test_g).discard(moved)
    return train_g, val_g, test_g


def prepare_splits(df, split_mode, train_camera, test_camera, val_frac, test_frac, seed):
    if split_mode == 'iid':
        d = df[df['camera'] == train_camera].copy()
        groups = sorted(d['group_id'].unique())
        train_g, val_g, test_g = split_groups(groups, val_frac, test_frac, seed)
        return (d[d['group_id'].isin(train_g)],
                d[d['group_id'].isin(val_g)],
                d[d['group_id'].isin(test_g)],
                {'mode': 'iid', 'camera': train_camera})

    if split_mode == 'cross_view':
        common = sorted(
            set(df[df['camera'] == train_camera]['group_id'].unique()) &
            set(df[df['camera'] == test_camera]['group_id'].unique())
        )
        if len(common) < 3:
            raise ValueError(f'Not enough common group_ids: {len(common)}')
        train_g, val_g, test_g = split_groups(common, val_frac, test_frac, seed)
        return (df[(df['camera'] == train_camera) & (df['group_id'].isin(train_g))],
                df[(df['camera'] == train_camera) & (df['group_id'].isin(val_g))],
                df[(df['camera'] == test_camera) & (df['group_id'].isin(test_g))],
                {'mode': 'cross_view', 'train_camera': train_camera, 'test_camera': test_camera})

    raise ValueError(f'Unknown split_mode: {split_mode}')


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def compute_class_weights(clips, label_to_idx, num_classes):
    labels = [label_to_idx[c['activity']] for c in clips]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, model_type):
    training = optimizer is not None
    model.train(training)
    total_loss, total_n = 0.0, 0
    ys, ps, group_ids = [], [], []

    for batch in loader:
        x, y, gid = batch
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)
        ys.extend(y.cpu().numpy().tolist())
        ps.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        group_ids.extend(list(gid))

    return total_loss / max(1, total_n), np.array(ys), np.array(ps), group_ids


def evaluate_metrics(y_true, y_pred, class_names):
    acc = float((y_true == y_pred).mean())
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    pr, rc, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0)
    total = cm.sum()
    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        rows.append({
            'activity': name,
            'precision': float(pr[i]),
            'recall': float(rc[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
            'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        })
    return {'accuracy': acc, 'macro_f1': float(macro_f1),
            'balanced_accuracy': float(bal_acc),
            'confusion_matrix': cm, 'per_class': rows}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--model', choices=['resnet18_clip_avg', 'r3d18'], required=True)
    ap.add_argument('--split-mode', choices=['iid', 'cross_view'], default='cross_view')
    ap.add_argument('--train-camera', default='camera_1_fps_15')
    ap.add_argument('--test-camera', default='camera_2_fps_15')
    ap.add_argument('--activities', nargs='+',
                    default=['Making pancake', 'Reading', 'Working on a computer'])
    ap.add_argument('--clip-len', type=int, default=16)
    ap.add_argument('--clip-stride', type=int, default=8)
    ap.add_argument('--img-size', type=int, default=112)  # 112 is standard for R3D-18
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--test-frac', type=float, default=0.20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-pretrained', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- load manifest ----
    df = pd.read_csv(args.manifest)
    df = df[df['activity'].isin(args.activities)].copy()
    df = df[df['local_path'].map(lambda p: isinstance(p, str) and os.path.exists(p))].copy()
    if len(df) == 0:
        raise ValueError('No rows after filtering. Check --manifest and --activities.')

    if args.split_mode == 'cross_view':
        g1 = set(df[df['camera'] == args.train_camera]['group_id'].unique())
        g2 = set(df[df['camera'] == args.test_camera]['group_id'].unique())
        df = df[df['group_id'].isin(g1 & g2)].copy()

    class_names = sorted(df['activity'].unique())
    label_to_idx = {c: i for i, c in enumerate(class_names)}

    train_df, val_df, test_df, split_desc = prepare_splits(
        df, args.split_mode, args.train_camera, args.test_camera,
        args.val_frac, args.test_frac, args.seed)

    # ---- build clips ----
    train_clips = build_clips(train_df, args.clip_len, args.clip_stride)
    val_clips   = build_clips(val_df,   args.clip_len, args.clip_stride)
    test_clips  = build_clips(test_df,  args.clip_len, args.clip_stride)

    print(f"Clips — train: {len(train_clips)}, val: {len(val_clips)}, test: {len(test_clips)}")

    # ---- transforms ----
    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ---- datasets & loaders ----
    pretrained = not args.no_pretrained
    if args.model == 'resnet18_clip_avg':
        DS = ClipAvgDataset
        model = ResNet18ClipAvg(len(class_names), pretrained=pretrained)
    else:
        DS = R3DDataset
        model = build_r3d18(len(class_names), pretrained=pretrained)

    train_ds = DS(train_clips, label_to_idx, train_tf)
    val_ds   = DS(val_clips,   label_to_idx, eval_tf)
    test_ds  = DS(test_clips,  label_to_idx, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ---- training ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    class_weights = compute_class_weights(train_clips, label_to_idx, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    best_state, best_val = None, -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_y, tr_p, _ = run_epoch(model, train_loader, criterion, optimizer, device, args.model)
        va_loss, va_y, va_p, _ = run_epoch(model, val_loader,   criterion, None,      device, args.model)
        tr_m = evaluate_metrics(tr_y, tr_p, class_names)
        va_m = evaluate_metrics(va_y, va_p, class_names)
        scheduler.step()

        row = {
            'epoch': epoch,
            'train_loss': tr_loss, 'val_loss': va_loss,
            'train_macro_f1': tr_m['macro_f1'], 'val_macro_f1': va_m['macro_f1'],
            'train_balanced_accuracy': tr_m['balanced_accuracy'],
            'val_balanced_accuracy': va_m['balanced_accuracy'],
        }
        history.append(row)
        print(row)

        if va_m['balanced_accuracy'] > best_val:
            best_val = va_m['balanced_accuracy']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # ---- test evaluation ----
    te_loss, te_y, te_p, te_gids = run_epoch(model, test_loader, criterion, None, device, args.model)
    te_m = evaluate_metrics(te_y, te_p, class_names)

    # ---- save outputs ----
    pd.DataFrame(history).to_csv(outdir / 'history.csv', index=False)
    pd.DataFrame(te_m['per_class']).to_csv(outdir / 'per_class_metrics.csv', index=False)
    pd.DataFrame({
        'group_id': te_gids,
        'y_true': [class_names[i] for i in te_y],
        'y_pred': [class_names[i] for i in te_p],
    }).to_csv(outdir / 'test_predictions.csv', index=False)

    save_confusion_matrix(te_m['confusion_matrix'], class_names, outdir / 'confusion_matrix.png')
    per_class_df = pd.DataFrame(te_m['per_class'])
    save_bar(per_class_df, 'f1',  outdir / 'per_class_f1.png',  'Per-class F1')
    save_bar(per_class_df, 'fpr', outdir / 'per_class_fpr.png', 'Per-class FPR')
    save_bar(per_class_df, 'fnr', outdir / 'per_class_fnr.png', 'Per-class FNR')

    torch.save({'state_dict': model.state_dict(), 'classes': class_names},
               outdir / 'best_model.pt')

    payload = {
        'model': args.model,
        'split': split_desc,
        'clip_len': args.clip_len,
        'clip_stride': args.clip_stride,
        'classes': class_names,
        'n_train_clips': len(train_clips),
        'n_val_clips': len(val_clips),
        'n_test_clips': len(test_clips),
        'test_loss': te_loss,
        'test_accuracy': te_m['accuracy'],
        'test_macro_f1': te_m['macro_f1'],
        'test_balanced_accuracy': te_m['balanced_accuracy'],
    }
    with open(outdir / 'metrics_overall.json', 'w') as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
