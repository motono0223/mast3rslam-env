# ベースイメージ: CUDA 11.8 Devel (コンパイル環境必須)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# ターゲットGPUアーキテクチャ (使用するGPUに合わせて変更推奨)
# 例: RTX 3090/4090 -> 8.6;8.9  A100 -> 8.0
# ここでは汎用的に設定していますが、ビルド時間を短縮するために絞ることを推奨します
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Minicondaのインストール
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Python環境の作成 (Python 3.10)
RUN conda create -n mast3r python=3.10 -y
SHELL ["conda", "run", "-n", "mast3r", "/bin/bash", "-c"]

# PyTorchのインストール (CUDA 11.8対応)
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 3D Gaussian Splatting のセットアップ
WORKDIR /app
RUN git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
WORKDIR /app/gaussian-splatting
RUN pip install -r requirements.txt
# CUDAカーネルのコンパイル (ここが最もエラーが出やすい箇所です)
RUN pip install submodules/diff-gaussian-rasterization
RUN pip install submodules/simple-knn

# MASt3R (Dust3rベース) のセットアップ
WORKDIR /app
# MASt3Rの公式リポジトリ (または使用したい特定のSLAM実装フォーク)
RUN git clone --recursive https://github.com/naver/mast3r.git
WORKDIR /app/mast3r
# MASt3R特有の依存関係
RUN pip install -r requirements.txt
# 追加で必要になることが多いパッケージ
RUN pip install opencv-python matplotlib scipy trimesh h5py

# エントリーポイントの設定
WORKDIR /app
CMD ["/bin/bash"]
