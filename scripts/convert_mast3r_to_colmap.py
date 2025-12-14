import os
import argparse
import shutil
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

def save_cameras_txt(path, cameras):
    """
    cameras.txt を作成 (COLMAP形式)
    Format: CAMERA_ID MODEL WIDTH HEIGHT params...
    ここでは簡易化のため PINHOLE モデルを使用
    """
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera.\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        
        for cam_id, cam in cameras.items():
            # cam: {'w': int, 'h': int, 'fx': float, 'fy': float, 'cx': float, 'cy': float}
            f.write(f"{cam_id} PINHOLE {cam['w']} {cam['h']} {cam['fx']} {cam['fy']} {cam['cx']} {cam['cy']}\n")

def save_images_txt(path, images):
    """
    images.txt を作成 (COLMAP形式)
    Format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    注意: COLMAPは World-to-Camera (w2c) 座標系を使用
    """
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image.\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        
        for img_id, img in images.items():
            # w2c の回転(Quaternion)と並進ベクトルを取得
            q = img['q'] # [qw, qx, qy, qz]
            t = img['t'] # [tx, ty, tz]
            name = img['name']
            cam_id = img['camera_id']
            
            f.write(f"{img_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {cam_id} {name}\n")
            f.write("\n") # 2行目は空(Points2Dの対応点)にしておく。3DGSの初期化には不要な場合が多い。

def save_points3D_txt(path, points):
    """
    points3D.txt を作成 (COLMAP形式)
    Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
    MASt3Rの密な点群をサブサンプリングして保存
    """
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point.\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write(f"# Number of points: {len(points)}\n")
        
        for i, pt in enumerate(points):
            # pt: {'xyz': [x, y, z], 'rgb': [r, g, b]}
            xyz = pt['xyz']
            rgb = pt['rgb']
            pt_id = i + 1
            # Error=0, Track=[] で埋める
            f.write(f"{pt_id} {xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} 0 \n")

def get_w2c_from_c2w(c2w_matrix):
    """
    Camera-to-World 行列 (4x4) を World-to-Camera (R, t) に変換
    """
    w2c_matrix = np.linalg.inv(c2w_matrix)
    R = w2c_matrix[:3, :3]
    t = w2c_matrix[:3, 3]
    return R, t

def main(args):
    print(f"Loading MASt3R output from: {args.input}")
    
    # torch.loadでデータを読み込む (CPUにマップ)
    # 想定: MASt3R/DUSt3Rの出力辞書。以下のキーを持つことを想定。
    # 'images': リスト (numpy or tensor) - 元画像
    # 'poses': (N, 4, 4) numpy or tensor - c2w matrices
    # 'intrinsics': (N, 3, 3) numpy or tensor
    # 'pts3d': リスト or (N, H, W, 3) - 点群マップ
    
    data = torch.load(args.input, map_location='cpu')
    
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)
    
    sparse_dir = os.path.join(output_path, "sparse", "0")
    images_dir = os.path.join(output_path, "images")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # データを解析用に整形
    # ※ MASt3Rの実装によってキー名が微妙に異なる場合があるため、適宜調整してください。
    # ここでは一般的なDUSt3R/MASt3RのGlobalAligner出力を想定しています。
    
    imgs = data['imgs'] if 'imgs' in data else data['images']
    poses = data['poses'] # c2w
    intrinsics = data['intrinsics']
    
    # 点群データの取得 (全画素分あると重すぎるのでサブサンプリングする)
    pts3d_list = data['pts3d'] if 'pts3d' in data else None
    
    colmap_cameras = {}
    colmap_images = {}
    colmap_points = []
    
    print("Processing Frames...")
    
    for idx in tqdm(range(len(imgs))):
        # 画像の保存
        img_name = f"{idx:05d}.jpg"
        # 画像がTensorならNumpyへ変換 (0-1 float -> 0-255 uint8)
        img_data = imgs[idx]
        if isinstance(img_data, torch.Tensor):
            img_data = img_data.detach().cpu().numpy()
        
        # 色空間の確認 (H,W,C か C,H,W か)
        if img_data.shape[0] == 3: 
             img_data = img_data.transpose(1, 2, 0)
             
        img_data = (img_data * 255).astype(np.uint8) if img_data.max() <= 1.0 else img_data.astype(np.uint8)
        
        # PILを使わずにcv2で保存 (opencv-pythonはDockerに入っている想定)
        import cv2
        # RGB -> BGR
        cv2.imwrite(os.path.join(images_dir, img_name), cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))
        
        H, W, _ = img_data.shape
        
        # カメラパラメータ (Intrinsics)
        # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        K = intrinsics[idx]
        if isinstance(K, torch.Tensor): K = K.detach().cpu().numpy()
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # カメラIDは1から始める
        cam_id = idx + 1
        colmap_cameras[cam_id] = {
            'w': W, 'h': H, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy
        }
        
        # ポーズ (Extrinsics)
        # MASt3Rの Pose は c2w なので w2c に変換
        c2w = poses[idx]
        if isinstance(c2w, torch.Tensor): c2w = c2w.detach().cpu().numpy()
        
        R, t = get_w2c_from_c2w(c2w)
        
        # Rotation Matrix -> Quaternion (scalar-last -> scalar-first for COLMAP: qw, qx, qy, qz)
        r = Rotation.from_matrix(R)
        qx, qy, qz, qw = r.as_quat() # scipy returns [x, y, z, w]
        
        colmap_images[idx + 1] = {
            'q': [qw, qx, qy, qz],
            't': t,
            'camera_id': cam_id,
            'name': img_name
        }

    print("Processing Point Cloud (Subsampling)...")
    # 点群の生成 (3DGSの初期化用)
    # MASt3Rの出力は各ピクセルに対応する dense な pointmap
    # これを間引いて sparse な points3D.txt を作る
    
    stride = 8 # 8ピクセルごとにサンプリング (適宜調整)
    
    for idx in range(len(imgs)):
        if pts3d_list is None: break
        
        pts = pts3d_list[idx] # (H, W, 3)
        if isinstance(pts, torch.Tensor): pts = pts.detach().cpu().numpy()
        
        img_data = imgs[idx]
        if isinstance(img_data, torch.Tensor): 
            img_data = img_data.detach().cpu().numpy()
            if img_data.shape[0] == 3: img_data = img_data.transpose(1, 2, 0)
        
        H, W, _ = pts.shape
        
        # グリッドサンプリング
        sub_pts = pts[::stride, ::stride, :].reshape(-1, 3)
        sub_rgb = img_data[::stride, ::stride, :].reshape(-1, 3)
        
        # 無効な点のフィルタリング (MASt3Rは無効点を0やinfにすることがあるので要確認)
        # ここでは簡易的なチェック
        valid_mask = np.isfinite(sub_pts).all(axis=1) & (np.linalg.norm(sub_pts, axis=1) > 0.01)
        
        sub_pts = sub_pts[valid_mask]
        sub_rgb = (sub_rgb[valid_mask] * 255).astype(np.uint8) if sub_rgb.max() <= 1.0 else sub_rgb[valid_mask].astype(np.uint8)

        for p, c in zip(sub_pts, sub_rgb):
            colmap_points.append({'xyz': p, 'rgb': c})
            
    # メモリ節約のため、点群が多すぎる場合はランダムサンプリング
    max_points = 100_000
    if len(colmap_points) > max_points:
        print(f"Downsampling points from {len(colmap_points)} to {max_points}...")
        indices = np.random.choice(len(colmap_points), max_points, replace=False)
        colmap_points = [colmap_points[i] for i in indices]

    print("Saving COLMAP text files...")
    save_cameras_txt(os.path.join(sparse_dir, "cameras.txt"), colmap_cameras)
    save_images_txt(os.path.join(sparse_dir, "images.txt"), colmap_images)
    save_points3D_txt(os.path.join(sparse_dir, "points3D.txt"), colmap_points)
    
    print(f"Conversion Complete! Output saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MASt3R/DUSt3R output to COLMAP format for 3DGS")
    parser.add_argument("--input", required=True, help="Path to MASt3R output (.pt file containing dictionary)")
    parser.add_argument("--output", required=True, help="Output directory for COLMAP style structure")
    args = parser.parse_args()
    
    main(args)
