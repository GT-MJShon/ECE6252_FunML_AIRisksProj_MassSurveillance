#!/usr/bin/env bash
set -euo pipefail

# Absolute paths grounded to your PACE layout.
ZIP="${ZIP:-/home/hice1/mshon6/scratch/ECE6252_project/Dataset/DARai/RGB_pt1/RGB_pt1_compressed.zip}"
OUT_ROOT="${OUT_ROOT:-/home/hice1/mshon6/scratch/ECE6252_project/Dataset/DARai/RGB_pt1/exp2_subset_rgb}"
MANIFEST="${MANIFEST:-$OUT_ROOT/manifest.csv}"

# Default subset tuned to stay small. Override from env if needed.
ACTIVITIES="${ACTIVITIES:-Making pancake|Reading|Working on a computer}"
CAMERAS="${CAMERAS:-camera_1_fps_15|camera_2_fps_15}"
SUBJECTS="${SUBJECTS:-01|02|03|04}"
FRAME_STEP="${FRAME_STEP:-3}"

mkdir -p "$OUT_ROOT"

python - <<'PY'
import csv
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

ZIP = os.environ['ZIP']
OUT_ROOT = os.environ['OUT_ROOT']
MANIFEST = os.environ['MANIFEST']
activities = set(os.environ['ACTIVITIES'].split('|'))
cameras = set(os.environ['CAMERAS'].split('|'))
subjects = set(os.environ['SUBJECTS'].split('|'))
frame_step = int(os.environ['FRAME_STEP'])

rx = re.compile(r".*RGB_compressed/([^/]+)/([^/]+)/([0-9]+)_([0-9]+)_([0-9]+)\.(jpg|jpeg|png)$", re.IGNORECASE)
groups = defaultdict(list)

with ZipFile(ZIP) as zf:
    for name in zf.namelist():
        m = rx.match(name)
        if not m:
            continue
        activity, camera, subject, session, frame, ext = m.groups()
        if activity not in activities or camera not in cameras or subject not in subjects:
            continue
        groups[(activity, camera, subject, session)].append((int(frame), name, subject, session))

    kept = []
    for key, items in groups.items():
        items = sorted(items, key=lambda x: x[0])
        kept.extend(items[::frame_step])

    print(f"Matched groups: {len(groups)}")
    print(f"Kept files after frame downsampling: {len(kept)}")

    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (frame, member, subject, session) in enumerate(kept, 1):
        rel = Path(*Path(member).parts[-4:])
        dst = Path(OUT_ROOT) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, open(dst, 'wb') as out:
            shutil.copyfileobj(src, out)
        activity, camera = rel.parts[1], rel.parts[2]
        rows.append({
            'local_path': str(dst),
            'zip_path': member,
            'activity': activity,
            'camera': camera,
            'token1': subject,
            'token2': session,
            'group_id': f'{subject}_{session}',
            'frame': frame,
        })
        if i % 1000 == 0:
            print(f"Extracted {i}/{len(kept)}")

with open(MANIFEST, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['local_path','zip_path','activity','camera','token1','token2','group_id','frame'])
    w.writeheader()
    w.writerows(rows)
print(f"Wrote manifest: {MANIFEST}")
PY
