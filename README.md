# Microsoft TRELLIS - RunPod Serverless Worker

This worker generates high-fidelity **3D Models (`.glb` & `.usdz`)** with sharp edges, hard-surface geometry, and PBR textures from a single 2D image using Microsoft TRELLIS.

## How to Deploy on RunPod Serverless:

1. **Push to GitHub**:
   Create a repository named `sellsphere-trellis-worker` and push the contents of this folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for TRELLIS RunPod Serverless Worker"
   git remote add origin https://github.com/<your-username>/sellsphere-trellis-worker.git
   git push -u origin main
   ```

2. **Create RunPod Serverless Endpoint**:
   - Go to [RunPod Serverless](https://www.runpod.io/console/serverless).
   - Click **+ New Endpoint**.
   - Select your GitHub repo: `sellsphere-trellis-worker`.
   - **GPU Tier**: NVIDIA RTX 4090 (24GB VRAM) or A40.
   - **Active Workers (Min)**: `0` (Zero idle cost).
   - **Max Workers**: `5` or `10`.
   - Click **Deploy**.

3. **Add Endpoint to Backend**:
   In `backend-client/.env`:
   ```env
   RUNPOD_TRELLIS_ENDPOINT_ID=your_trellis_endpoint_id
   ```
