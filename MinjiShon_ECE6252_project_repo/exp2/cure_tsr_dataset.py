#!/usr/bin/env python3
"""CURE-TSR manifest builder and dataset.

Grounding assumptions kept intentionally minimal:
- Root is expected to contain split folders such as Real_Test.
- A condition folder is any path component equal to ChallengeFree or matching <ChallengeType>-<Level>.
- Labels are NOT guessed from arbitrary path depth. By default, label_mode='auto' infers the label as the
  first directory immediately below the condition folder, and raises if files are directly under the condition folder.

Use --inspect first if you have not verified the exact structure.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_CURE_ROOT = "/storage/ice1/shared/d-pace_community/makerspace-datasets/AVs/CURE-TSR"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
COND_RE = re.compile(r"^(?P<challenge>.+)-(?P<level>[1-5])$")


@dataclass
class CURERecord:
    absolute_path: str
    relative_path: str
    split: str
    condition_folder: str
    challenge_type: str
    level: int
    label: str


def parse_condition_from_parts(parts: Sequence[str]) -> Optional[Tuple[int, str, str, int]]:
    for idx, part in enumerate(parts):
        if part == "ChallengeFree":
            return idx, part, "ChallengeFree", 0
        m = COND_RE.match(part)
        if m:
            return idx, part, m.group("challenge"), int(m.group("level"))
    return None


def infer_label_from_path(parts: Sequence[str], cond_idx: int, label_mode: str, label_regex: Optional[str], rel_path: str) -> str:
    if label_mode == "condition_child":
        if cond_idx + 2 > len(parts) - 1:
            raise ValueError(
                f"Cannot infer label from path because file appears directly under condition folder: {rel_path}"
            )
        return parts[cond_idx + 1]

    if label_mode == "filename_regex":
        if not label_regex:
            raise ValueError("label_mode=filename_regex requires --label-regex")
        m = re.search(label_regex, parts[-1])
        if not m:
            raise ValueError(f"Filename regex did not match: {rel_path}")
        if m.lastindex is None:
            raise ValueError("Filename regex must contain at least one capturing group")
        return m.group(1)

    if label_mode == "auto":
        # Fail closed if files are directly below the condition folder.
        if cond_idx + 2 > len(parts) - 1:
            raise ValueError(
                "Auto label inference failed: image file is directly below condition folder, so label is ambiguous. "
                f"Path: {rel_path}. Re-run with --label-mode filename_regex or provide a manifest."
            )
        return parts[cond_idx + 1]

    raise ValueError(f"Unknown label_mode: {label_mode}")


def discover_cure_records(
    root: str = DEFAULT_CURE_ROOT,
    split_filter: Optional[Sequence[str]] = None,
    challenge_type_filter: Optional[Sequence[str]] = None,
    level_filter: Optional[Sequence[int]] = None,
    condition_filter: Optional[Sequence[str]] = None,
    label_mode: str = "auto",
    label_regex: Optional[str] = None,
) -> List[CURERecord]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"CURE-TSR root does not exist: {root}")

    split_set = set(split_filter or [])
    challenge_set = set(challenge_type_filter or [])
    level_set = set(int(x) for x in (level_filter or []))
    condition_set = set(condition_filter or [])

    records: List[CURERecord] = []
    for path in root_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
            continue
        rel_parts = path.relative_to(root_path).parts
        cond = parse_condition_from_parts(rel_parts)
        if cond is None:
            continue
        cond_idx, cond_folder, challenge_type, level = cond
        split = rel_parts[cond_idx - 1] if cond_idx - 1 >= 0 else "UNKNOWN_SPLIT"

        if split_set and split not in split_set:
            continue
        if challenge_set and challenge_type not in challenge_set:
            continue
        if level_set and level not in level_set:
            continue
        if condition_set and cond_folder not in condition_set:
            continue

        rel_path = str(path.relative_to(root_path))
        label = infer_label_from_path(rel_parts, cond_idx, label_mode, label_regex, rel_path)
        records.append(
            CURERecord(
                absolute_path=str(path.resolve()),
                relative_path=rel_path,
                split=split,
                condition_folder=cond_folder,
                challenge_type=challenge_type,
                level=level,
                label=label,
            )
        )
    if not records:
        raise RuntimeError(
            "No CURE-TSR images were indexed with the current filters. Run with --inspect first and verify the path."
        )
    return records


class CURETSRDataset(Dataset):
    def __init__(
        self,
        *,
        root: str = DEFAULT_CURE_ROOT,
        manifest_csv: Optional[str] = None,
        split_filter: Optional[Sequence[str]] = None,
        challenge_type_filter: Optional[Sequence[str]] = None,
        level_filter: Optional[Sequence[int]] = None,
        condition_filter: Optional[Sequence[str]] = None,
        label_mode: str = "auto",
        label_regex: Optional[str] = None,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.transform = transform
        if manifest_csv:
            df = pd.read_csv(manifest_csv)
        else:
            rows = [r.__dict__ for r in discover_cure_records(
                root=root,
                split_filter=split_filter,
                challenge_type_filter=challenge_type_filter,
                level_filter=level_filter,
                condition_filter=condition_filter,
                label_mode=label_mode,
                label_regex=label_regex,
            )]
            df = pd.DataFrame(rows)

        if split_filter:
            df = df[df["split"].isin(split_filter)].copy()
        if challenge_type_filter:
            df = df[df["challenge_type"].isin(challenge_type_filter)].copy()
        if level_filter:
            df = df[df["level"].isin([int(x) for x in level_filter])].copy()
        if condition_filter:
            df = df[df["condition_folder"].isin(condition_filter)].copy()
        if df.empty:
            raise RuntimeError("CURETSRDataset is empty after filtering")

        self.df = df.reset_index(drop=True)
        if class_to_idx is None:
            classes = sorted(self.df["label"].astype(str).unique())
            class_to_idx = {c: i for i, c in enumerate(classes)}
        self.class_to_idx = dict(class_to_idx)
        missing = sorted(set(self.df["label"].astype(str)) - set(self.class_to_idx))
        if missing:
            raise ValueError(f"Labels in dataset not present in class_to_idx: {missing[:20]}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["absolute_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = self.class_to_idx[str(row["label"])]
        meta = {
            "absolute_path": row["absolute_path"],
            "relative_path": row["relative_path"],
            "split": row["split"],
            "condition_folder": row["condition_folder"],
            "challenge_type": row["challenge_type"],
            "level": int(row["level"]),
            "label": str(row["label"]),
        }
        return img, y, meta


def write_manifest(rows: List[CURERecord], out_csv: str) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["absolute_path", "relative_path", "split", "condition_folder", "challenge_type", "level", "label"])
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)


def inspect(root: str, limit: int = 40) -> None:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root)
    shown = 0
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            rel = str(path.relative_to(root_path))
            print(rel)
            shown += 1
            if shown >= limit:
                break
    if shown == 0:
        print("No image files found")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect")
    p.add_argument("--root", default=DEFAULT_CURE_ROOT)
    p.add_argument("--limit", type=int, default=40)

    p = sub.add_parser("build-manifest")
    p.add_argument("--root", default=DEFAULT_CURE_ROOT)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--split", action="append", default=[])
    p.add_argument("--condition", action="append", default=[])
    p.add_argument("--challenge-type", action="append", default=[])
    p.add_argument("--level", type=int, action="append", default=[])
    p.add_argument("--label-mode", choices=["auto", "condition_child", "filename_regex"], default="auto")
    p.add_argument("--label-regex", default=None)

    args = ap.parse_args()
    if args.cmd == "inspect":
        inspect(args.root, args.limit)
    elif args.cmd == "build-manifest":
        rows = discover_cure_records(
            root=args.root,
            split_filter=args.split,
            challenge_type_filter=args.challenge_type,
            level_filter=args.level,
            condition_filter=args.condition,
            label_mode=args.label_mode,
            label_regex=args.label_regex,
        )
        write_manifest(rows, args.out_csv)
        print(f"Wrote {len(rows)} rows to {args.out_csv}")
        df = pd.DataFrame([r.__dict__ for r in rows])
        print(df.groupby(["split", "condition_folder"]).size())
        print(f"Num labels: {df['label'].nunique()}")


if __name__ == "__main__":
    main()
