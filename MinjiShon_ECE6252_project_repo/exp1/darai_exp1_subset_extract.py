#!/usr/bin/env python3
"""Selective extraction + manifest builder for DARai RGB zip.
Grounded to observed zip layout:
.../RGB_compressed/<activity>/<camera>/<token1>_<token2>_<frame>.jpg
"""
import argparse
import csv
import os
import re
import shutil
from collections import defaultdict, Counter
from pathlib import Path
from zipfile import ZipFile

IMG_RE = re.compile(r".*RGB_compressed/([^/]+)/([^/]+)/([0-9]+)_([0-9]+)_([0-9]+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def natural_key(s: str):
    import re as _re
    return [int(x) if x.isdigit() else x.lower() for x in _re.split(r'(\d+)', s)]


def parse_member(name: str):
    m = IMG_RE.match(name)
    if not m:
        return None
    activity, camera, token1, token2, frame, ext = m.groups()
    return {
        'zip_path': name,
        'activity': activity,
        'camera': camera,
        'token1': token1,
        'token2': token2,
        'group_id': f'{token1}_{token2}',
        'frame': int(frame),
        'ext': ext.lower(),
        'filename': os.path.basename(name),
    }


def inspect(zip_path: str, topn: int = 40):
    activities = Counter()
    cameras = Counter()
    pair_counts = Counter()
    token1_counts = defaultdict(set)
    token2_counts = defaultdict(set)
    total = 0
    with ZipFile(zip_path) as zf:
        for name in zf.namelist():
            rec = parse_member(name)
            if rec is None:
                continue
            total += 1
            activities[rec['activity']] += 1
            cameras[rec['camera']] += 1
            pair_counts[(rec['activity'], rec['camera'])] += 1
            token1_counts[(rec['activity'], rec['camera'])].add(rec['token1'])
            token2_counts[(rec['activity'], rec['camera'])].add(rec['token2'])
    print(f'Parsed image files: {total}')
    print('\nActivities:')
    for k, v in activities.most_common():
        print(f'{v:8d}  {k}')
    print('\nCameras:')
    for k, v in cameras.most_common():
        print(f'{v:8d}  {k}')
    print(f'\nTop {topn} activity-camera pairs:')
    for (activity, camera), n in pair_counts.most_common(topn):
        print(f'{n:8d} | {activity} | {camera} | token1={len(token1_counts[(activity, camera)])} | token2={len(token2_counts[(activity, camera)])}')


def collect_records(zip_path: str, activities=None, cameras=None, token1_keep=None, token2_keep=None):
    activities = set(activities or [])
    cameras = set(cameras or [])
    token1_keep = set(token1_keep or [])
    token2_keep = set(token2_keep or [])
    out = []
    with ZipFile(zip_path) as zf:
        for name in zf.namelist():
            rec = parse_member(name)
            if rec is None:
                continue
            if activities and rec['activity'] not in activities:
                continue
            if cameras and rec['camera'] not in cameras:
                continue
            if token1_keep and rec['token1'] not in token1_keep:
                continue
            if token2_keep and rec['token2'] not in token2_keep:
                continue
            out.append(rec)
    return out


def downsample_records(records, frame_step: int):
    if frame_step <= 1:
        return records
    groups = defaultdict(list)
    for r in records:
        key = (r['activity'], r['camera'], r['group_id'])
        groups[key].append(r)
    kept = []
    for key, items in groups.items():
        items = sorted(items, key=lambda x: x['frame'])
        kept.extend(items[::frame_step])
    kept.sort(key=lambda r: natural_key(r['zip_path']))
    return kept


def dryrun(zip_path: str, activities, cameras, token1_keep, token2_keep, frame_step: int):
    with ZipFile(zip_path) as zf:
        info = {zi.filename: zi.file_size for zi in zf.infolist()}
    records = collect_records(zip_path, activities, cameras, token1_keep, token2_keep)
    kept = downsample_records(records, frame_step)
    size_gb = sum(info[r['zip_path']] for r in kept) / (1024 ** 3)
    print(f'Matched files before downsampling: {len(records)}')
    print(f'Kept files after downsampling:   {len(kept)}')
    print(f'Estimated uncompressed size:     {size_gb:.2f} GB')
    print('\nCounts by activity-camera:')
    cnt = Counter((r['activity'], r['camera']) for r in kept)
    for (a, c), n in cnt.most_common():
        print(f'{n:8d} | {a} | {c}')
    print('\nFirst 40 kept files:')
    for r in kept[:40]:
        print(r['zip_path'])


def extract(zip_path: str, out_root: str, activities, cameras, token1_keep, token2_keep, frame_step: int, manifest_path: str):
    records = collect_records(zip_path, activities, cameras, token1_keep, token2_keep)
    kept = downsample_records(records, frame_step)
    os.makedirs(out_root, exist_ok=True)
    with ZipFile(zip_path) as zf:
        for i, r in enumerate(kept, 1):
            rel = Path(*Path(r['zip_path']).parts[-4:])  # RGB_compressed/activity/camera/file
            dst = Path(out_root) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(r['zip_path']) as src, open(dst, 'wb') as out:
                shutil.copyfileobj(src, out)
            r['local_path'] = str(dst)
            if i % 1000 == 0:
                print(f'Extracted {i}/{len(kept)}')
    write_manifest(kept, manifest_path)
    print(f'Wrote manifest: {manifest_path}')


def write_manifest(records, manifest_path: str):
    fields = ['local_path', 'zip_path', 'activity', 'camera', 'token1', 'token2', 'group_id', 'frame']
    with open(manifest_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(records, key=lambda x: natural_key(x.get('local_path', x['zip_path']))):
            row = {k: r.get(k, '') for k in fields}
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('inspect')
    p.add_argument('--zip', required=True)
    p.add_argument('--topn', type=int, default=40)

    p = sub.add_parser('dryrun')
    p.add_argument('--zip', required=True)
    p.add_argument('--activity', action='append', default=[])
    p.add_argument('--camera', action='append', default=[])
    p.add_argument('--token1', action='append', default=[])
    p.add_argument('--token2', action='append', default=[])
    p.add_argument('--frame-step', type=int, default=3)

    p = sub.add_parser('extract')
    p.add_argument('--zip', required=True)
    p.add_argument('--out-root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--activity', action='append', default=[])
    p.add_argument('--camera', action='append', default=[])
    p.add_argument('--token1', action='append', default=[])
    p.add_argument('--token2', action='append', default=[])
    p.add_argument('--frame-step', type=int, default=3)

    args = ap.parse_args()
    if args.cmd == 'inspect':
        inspect(args.zip, args.topn)
    elif args.cmd == 'dryrun':
        dryrun(args.zip, args.activity, args.camera, args.token1, args.token2, args.frame_step)
    elif args.cmd == 'extract':
        extract(args.zip, args.out_root, args.activity, args.camera, args.token1, args.token2, args.frame_step, args.manifest)


if __name__ == '__main__':
    main()
