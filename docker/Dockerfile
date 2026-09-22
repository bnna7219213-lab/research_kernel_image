# ============================================================
# T2Image Phase-A — Base Image
# Target: NVIDIA A100 80GB, CUDA 12.x, PyTorch 2.5+
# Usage: docker build -t registry/t2image:phase-a .
# ============================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs wget curl ffmpeg libgl1-mesa-glx libglib2.0-0 \
        python3.11 python3.11-dev python3-pip \
        ninja-build && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch / CUDA 12.4
RUN pip install --no-cache-dir \
        torch==2.5.1+cu124 \
        torchvision==0.20.1+cu124 \
        torchaudio==2.5.1+cu124 \
        -f https://download.pytorch.org/whl/torch_stable.html

# Diffusion / transformer stack
RUN pip install --no-cache-dir \
        diffusers==0.32.2 \
        transformers==4.46.3 \
        accelerate==1.1.1 \
        datasets==3.1.0 \
        peft==0.14.0 \
        invisible-watermark==0.2.0 \
        omegaconf==2.3.0 \
        safetensors==0.4.5 \
        xformers==0.0.28.post1 \
        controlnet-aux==0.0.9

# Evaluation stack
RUN pip install --no-cache-dir \
        torch-fidelity==0.3.0 \
        clip==0.2.0 \
        pickscore==0.2.1 \
        hpsv2==1.2.0 \
        clean-fid==0.1.35 \
        open-clip-torch==2.26.1

# Data pipeline
RUN pip install --no-cache-dir \
        img2dataset==1.45.0 \
        webdataset==0.2.106 \
        albumentations==1.4.22 \
        Pillow==10.4.0

# Tracking & utilities
RUN pip install --no-cache-dir \
        wandb==0.18.7 \
        mlflow==2.18.0 \
        typer==0.13.0 \
        loguru==0.7.2 \
        tqdm==4.67.1 \
        orjson==3.10.11 \
        h5py==3.12.1

WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -e /workspace || true

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["help"]
