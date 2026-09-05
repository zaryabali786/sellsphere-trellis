# RunPod Serverless Dockerfile for Microsoft TRELLIS (Image-to-3D GLB & Apple USDZ) - Release v1.0.3
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /content

# Environment settings
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV ATTN_BACKEND=xformers
ENV SPCONV_ALGO=native
ENV PYTHONPATH="/content/TRELLIS:${PYTHONPATH}"

# Install system dependencies & Blender shared libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    xz-utils \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxi6 \
    libxkbcommon0 \
    libsm6 \
    libxext6 \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Blender 4.2 LTS for lossless Apple USDZ export
RUN wget -q https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz -O /tmp/blender.tar.xz && \
    tar -xf /tmp/blender.tar.xz -C /opt/ && \
    ln -s /opt/blender-4.2.0-linux-x64/blender /usr/local/bin/blender && \
    rm /tmp/blender.tar.xz

# Install python dependencies from requirements.txt
COPY requirements.txt /content/requirements.txt
RUN pip install --no-cache-dir -r /content/requirements.txt

# Install official pre-built wheels for PyTorch 2.2.0 + CUDA 12.1
RUN pip install --no-cache-dir xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir spconv-cu120

# Install utils3d
RUN pip install --no-cache-dir git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Compile CUDA submodules for TRELLIS
RUN git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    cd /tmp/extensions/nvdiffrast && pip install --no-build-isolation . && \
    rm -rf /tmp/extensions/nvdiffrast

RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast && \
    cd /tmp/extensions/diffoctreerast && pip install --no-build-isolation . && \
    rm -rf /tmp/extensions/diffoctreerast

RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting && \
    cd /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization && pip install --no-build-isolation . && \
    rm -rf /tmp/extensions/mip-splatting

# Clone official Microsoft TRELLIS repository
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /content/TRELLIS

# Copy USDZ converter script and RunPod Serverless handler
COPY convert_glb_to_usdz.py /content/convert_glb_to_usdz.py
COPY rp_handler.py /content/rp_handler.py

WORKDIR /content

CMD ["python", "-u", "/content/rp_handler.py"]
