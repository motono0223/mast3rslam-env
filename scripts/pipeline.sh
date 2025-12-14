#!/bin/bash

# =================================================================
# End-to-End Pipeline: Image -> MASt3R -> COLMAP Format -> 3DGS
# =================================================================

# エラーが発生したら即停止
set -e

# コンテナ内のConda環境設定
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mast3r

# --- 設定項目 ---
# 入力画像があるディレクトリ (ホストの ./data/my_scene に対応)
INPUT_IMAGES_DIR="/app/data/my_scene/images" 

# 出力ディレクトリ
BASE_OUTPUT_DIR="/app/output/my_scene"
MAST3R_OUT_DIR="${BASE_OUTPUT_DIR}/mast3r_out"
COLMAP_OUT_DIR="${BASE_OUTPUT_DIR}/colmap_style"
GAUSSIAN_OUT_DIR="${BASE_OUTPUT_DIR}/gaussian_out"

# スクリプトのパス
CONVERTER_SCRIPT="/app/scripts/convert_mast3r_to_colmap.py"
MAST3R_WRAPPER_SCRIPT="${BASE_OUTPUT_DIR}/run_mast3r_wrapper.py"

echo "=== Pipeline Started ==="
echo "Input: $INPUT_IMAGES_DIR"
echo "Output: $BASE_OUTPUT_DIR"

# 出力ディレクトリの作成
mkdir -p "$MAST3R_OUT_DIR"
mkdir -p "$COLMAP_OUT_DIR"

# =================================================================
# Step 1: MASt3R Wrapper Script Generation & Execution
# =================================================================
echo ">>> [Step 1] Generating and Running MASt3R Inference..."

# MASt3Rをヘッドレス(GUIなし)で実行し、.ptファイルを保存するPythonスクリプトをその場で作成します。
# これにより、MASt3Rのソースコードを直接編集する必要がなくなります。

cat << EOF > "$MAST3R_WRAPPER_SCRIPT"
import os
import torch
import glob
from PIL import Image
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_nn_matching

# 画像読み込み
image_paths = sorted(glob.glob(os.path.join("$INPUT_IMAGES_DIR", "*")))
if len(image_paths) == 0:
    raise ValueError("No images found in $INPUT_IMAGES_DIR")

print(f"Found {len(image_paths)} images. Loading model...")

# モデルロード (MASt3R Large)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AsymmetricMASt3R.from_pretrained("naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdt_metric").to(device)

# 推論実行 (Global Alignment)
# 注: mast3rのAPIはdust3rに似ています。ここでは簡易的なGlobalAlignmentを使用します。
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerIterative

# 画像ロード (サイズは512にリサイズ推奨、メモリに応じて調整)
imgs = load_images(image_paths, size=512)
pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)

print("Running Inference...")
output = inference(pairs, model, device, batch_size=1, verbose=True)

print("Running Global Alignment...")
scene = global_aligner(output, device=device, mode=GlobalAlignerIterative)
scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=0.01)

# 結果の抽出
print("Extracting Data...")
imgs_np = scene.imgs
poses = scene.get_poses() # c2w (N, 4, 4)
intrinsics = scene.get_intrinsics() # (N, 3, 3)
pts3d = scene.get_pts3d() # List of (H, W, 3)

# 保存用辞書の作成
output_data = {
    'imgs': imgs_np,
    'poses': poses.detach().cpu(),
    'intrinsics': intrinsics.detach().cpu(),
    'pts3d': [p.detach().cpu() for p in pts3d]
}

save_path = os.path.join("$MAST3R_OUT_DIR", "reconstruction.pt")
torch.save(output_data, save_path)
print(f"Saved reconstruction data to {save_path}")
EOF

# 作成したラッパーを実行
cd /app/mast3r
python "$MAST3R_WRAPPER_SCRIPT"

# =================================================================
# Step 2: Convert to COLMAP Format
# =================================================================
echo ">>> [Step 2] Converting MASt3R output to COLMAP format..."

python "$CONVERTER_SCRIPT" \
    --input "${MAST3R_OUT_DIR}/reconstruction.pt" \
    --output "$COLMAP_OUT_DIR"

echo "Conversion finished. Check $COLMAP_OUT_DIR"

# =================================================================
# Step 3: 3D Gaussian Splatting Training
# =================================================================
echo ">>> [Step 3] Running 3D Gaussian Splatting Training..."

# 3DGSのディレクトリへ移動
cd /app/gaussian-splatting

# 学習実行
# -s: source path (COLMAP形式のフォルダ)
# -m: model path (出力先)
# --iterations: 反復回数 (テスト用に7000回に設定していますが、高品質なら30000回推奨)

python train.py \
    -s "$COLMAP_OUT_DIR" \
    -m "$GAUSSIAN_OUT_DIR" \
    --iterations 7000

echo "=== Pipeline Finished Successfully ==="
echo "Final 3DGS model is located at: $GAUSSIAN_OUT_DIR"
