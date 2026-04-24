#!/bin/bash
#SBATCH --job-name=darai_r3d18
#SBATCH --partition=ice-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.log

MANIFEST=/home/hice1/mshon6/scratch/ECE6252_project/Dataset/DARai/RGB_pt1/exp1_rgb_baseline/subset_rgb/manifest.csv
RUNS=/home/hice1/mshon6/scratch/ECE6252_project/Dataset/DARai/RGB_pt1/exp1_rgb_baseline/runs
SCRIPT=/home/hice1/mshon6/scratch/ECE6252_project/repo/exp1/darai_exp1_clip_train.py
VENV=/home/hice1/mshon6/scratch/ECE6252_project/Dataset/DARai/RGB_pt1/exp1_rgb_baseline/.venv

source "$VENV/bin/activate"

echo "=== [1/2] r3d18 IID ==="
python "$SCRIPT" \
  --manifest "$MANIFEST" \
  --outdir "$RUNS/r3d18_iid" \
  --model r3d18 \
  --split-mode iid \
  --train-camera camera_1_fps_15 \
  --activities "Making pancake" "Reading" "Working on a computer" \
  --clip-len 16 --clip-stride 8 \
  --epochs 10 --batch-size 8

echo "=== [2/2] r3d18 cross-view ==="
python "$SCRIPT" \
  --manifest "$MANIFEST" \
  --outdir "$RUNS/r3d18_cross_view" \
  --model r3d18 \
  --split-mode cross_view \
  --train-camera camera_1_fps_15 \
  --test-camera camera_2_fps_15 \
  --activities "Making pancake" "Reading" "Working on a computer" \
  --clip-len 16 --clip-stride 8 \
  --epochs 10 --batch-size 8

echo "=== All done ==="
