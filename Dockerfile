# RunPod Serverless Dockerfile for Microsoft TRELLIS (Image-to-3D GLB & USDZ)
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /content

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt /content/requirements.txt
RUN pip install --no-cache-dir -r /content/requirements.txt

# Clone official Microsoft TRELLIS repository
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /content/TRELLIS

# Copy RunPod Serverless handler
COPY rp_handler.py /content/rp_handler.py

WORKDIR /content

CMD ["python", "-u", "/content/rp_handler.py"]
