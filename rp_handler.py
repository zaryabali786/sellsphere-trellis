"""
RunPod Serverless Handler for Microsoft TRELLIS (High-Quality Image-to-3D with GLB & Apple USDZ)
"""
import os
import sys

# Configure backends before importing PyTorch
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"

import io
import time
import base64
import subprocess
import requests
import runpod
from PIL import Image

sys.path.append("/content/TRELLIS")

# Lazy-loaded pipeline cache
pipeline = None

def load_pipeline():
    global pipeline
    if pipeline is not None:
        return pipeline

    import torch
    print(f"Loading Microsoft TRELLIS 3D pipeline (CUDA available: {torch.cuda.is_available()})...")
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    if torch.cuda.is_available():
        pipeline.cuda()
    print("Microsoft TRELLIS pipeline successfully loaded onto GPU.")
    return pipeline

def download_image(url_or_data):
    if not url_or_data:
        return None
    if url_or_data.startswith("data:image"):
        _, data = url_or_data.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGBA")
    elif url_or_data.startswith("http://") or url_or_data.startswith("https://"):
        resp = requests.get(url_or_data, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGBA")
    elif os.path.exists(url_or_data):
        return Image.open(url_or_data).convert("RGBA")
    else:
        return Image.open(io.BytesIO(base64.b64decode(url_or_data))).convert("RGBA")

def file_to_base64_data_uri(filepath, mime_type="model/gltf-binary"):
    with open(filepath, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

def convert_glb_to_usdz(glb_path, usdz_path):
    """
    Converts GLB to Apple USDZ using Blender 4.2 LTS headless.
    Preserves all PBR materials, base colors, and textures without watermarks.
    """
    cmd = [
        "blender",
        "-b",
        "--python",
        "/content/convert_glb_to_usdz.py",
        "--",
        glb_path,
        usdz_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if res.returncode == 0 and os.path.exists(usdz_path) and os.path.getsize(usdz_path) > 100:
            print(f"Blender USDZ conversion success ({os.path.getsize(usdz_path)} bytes)")
            return True
        else:
            print(f"Blender returncode: {res.returncode}")
            if res.stdout:
                print(f"Blender stdout: {res.stdout[-400:]}")
            if res.stderr:
                print(f"Blender stderr: {res.stderr[-400:]}")
            return False
    except Exception as e:
        print(f"Blender execution error: {e}")
        return False

def handler(job):
    job_input = job.get("input", {})
    image_url = job_input.get("image") or job_input.get("image_url") or job_input.get("human_img")
    seed = int(job_input.get("seed", 42))
    ss_guidance_strength = float(job_input.get("ss_guidance_strength", 7.5))
    slat_guidance_strength = float(job_input.get("slat_guidance_strength", 3.0))

    if not image_url:
        return {"error": "Image input is required (URL or base64)", "status": "FAILED"}

    try:
        t_start = time.time()
        pipe = load_pipeline()
        img = download_image(image_url)

        job_id = str(job.get("id", "job"))
        output_dir = f"/tmp/trellis_output/{job_id}"
        os.makedirs(output_dir, exist_ok=True)
        glb_file = os.path.join(output_dir, "model.glb")
        usdz_file = os.path.join(output_dir, "model.usdz")

        print(f"Starting TRELLIS 3D inference (Image: {img.size}, seed: {seed})...")
        outputs = pipe.run(
            img,
            seed=seed,
            sparse_structure_sampler_params={"steps": 12, "cfg_strength": ss_guidance_strength},
            slat_sampler_params={"steps": 12, "cfg_strength": slat_guidance_strength},
            formats=["mesh", "gaussian"],
            preprocess_image=True
        )

        print("Synthesizing 3D mesh & baking 1024x1024 PBR textures...")
        from trellis.utils import postprocessing_utils
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=0.95,
            texture_size=1024
        )
        glb.export(glb_file)
        glb_size = os.path.getsize(glb_file)
        print(f"Textured GLB created successfully ({glb_size} bytes, {round(time.time() - t_start, 1)}s)")

        # Convert to USDZ for Apple AR Quick Look
        print("Converting GLB to Apple USDZ via Blender 4.2...")
        usdz_ok = convert_glb_to_usdz(glb_file, usdz_file)

        glb_uri = file_to_base64_data_uri(glb_file, "model/gltf-binary")
        usdz_uri = file_to_base64_data_uri(usdz_file, "model/vnd.usdz+zip") if usdz_ok and os.path.exists(usdz_file) else None

        print(f"Completed job {job_id}. Total time: {round(time.time() - t_start, 1)}s")

        return {
            "status": "COMPLETED",
            "glb_url": glb_uri,
            "usdz_url": usdz_uri,
            "seed": seed
        }

    except Exception as e:
        import traceback
        err_msg = f"Error in TRELLIS handler: {e}\n{traceback.format_exc()}"
        print(err_msg)
        return {"error": str(e), "status": "FAILED"}

# Warm up pipeline on worker startup
try:
    load_pipeline()
except Exception as e:
    print(f"Startup notice: pipeline warm-up deferred: {e}")

runpod.serverless.start({"handler": handler})
