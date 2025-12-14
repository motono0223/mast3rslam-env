#!/bin/bash

# エラーハンドリング
set -e

# Conda環境の読み込み (念のため)
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mast3r

DATA_DIR="/app/data/my_scene"
OUTPUT_DIR="/app/output/my_scene"

echo "=== Pipeline Started ==="

# ---------------------------------------------------------
# Step 1: MASt3Rによる推論 (カメラ姿勢と粗な点群の取得)
# ---------------------------------------------------------
echo ">>> Running MASt3R..."
cd /app/mast3r

# 注意: MASt3Rの具体的な実行コマンドは使用するスクリプトに依存します。
# 以下はmast3rのdemo.pyを想定した例です。
# 実際には、複数の画像からGlobal Alignmentを行うスクリプトを実行します。
python demo.py --input_dir $DATA_DIR --output_dir $OUTPUT_DIR/mast3r_out

# ---------------------------------------------------------
# Step 2: データ変換 (MASt3R output -> 3DGS input)
# ---------------------------------------------------------
echo ">>> Converting MASt3R output to 3DGS format..."

# ここが重要です。3DGSは通常、COLMAP形式 (images, cameras.txt, images.txt, points3D.txt) を期待します。
# MASt3Rの出力（poses, pointmaps）をこの形式に変換するPythonスクリプトが必要です。
# 仮に 'convert_mast3r_to_colmap.py' というスクリプトがあるとして実行します。
# python /app/scripts/convert_mast3r_to_colmap.py --input $OUTPUT_DIR/mast3r_out --output $OUTPUT_DIR/colmap_style

echo "Note: Ensure you have a converter script to bridge MASt3R and 3DGS."

# ---------------------------------------------------------
# Step 3: 3D Gaussian Splatting Training
# ---------------------------------------------------------
echo ">>> Running 3D Gaussian Splatting Training..."
cd /app/gaussian-splatting

# 変換されたデータを使って学習開始
# source_pathにはCOLMAP形式のフォルダを指定します
python train.py -s $OUTPUT_DIR/colmap_style -m $OUTPUT_DIR/gaussian_out --iterations 7000

echo "=== Pipeline Finished ==="
echo "Results are saved in $OUTPUT_DIR/gaussian_out"
