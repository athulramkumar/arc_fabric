"""LTX-Video model worker - thin FastAPI server wrapping the LTX inference pipeline."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticModel

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

WORKER_DIR = Path(__file__).parent
PROJECT_ROOT = WORKER_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models" / "ltx_video"
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "ltx_video"

app = FastAPI(title="LTX-Video Worker")

ltx_infer = None
InferenceConfig = None
model_variant = None
pipeline_config_path = None


class GenerateRequest(PydanticModel):
    prompt: str
    height: int = 480
    width: int = 704
    num_frames: int = 97
    seed: int = 42
    session_id: Optional[str] = None
    output_dir: Optional[str] = None


class GenerateResponse(PydanticModel):
    status: str
    output_path: Optional[str] = None
    num_frames: int = 0
    generation_time_s: float = 0.0
    error: Optional[str] = None


def create_local_config(variant: str) -> str:
    """Create a config YAML with absolute paths to local weights."""
    import yaml

    if variant in ("ltx_2b", "ltx_video"):
        weights_subdir = "ltxv-2b-0.9.8-distilled"
        checkpoint_file = "ltxv-2b-0.9.8-distilled.safetensors"
    else:
        weights_subdir = "ltxv-13b-0.9.8-distilled"
        checkpoint_file = "ltxv-13b-0.9.8-distilled.safetensors"

    weights_path = WEIGHTS_DIR / weights_subdir

    config = {
        "pipeline_type": "multi-scale",
        "checkpoint_path": str(weights_path / checkpoint_file),
        "downscale_factor": 0.6666666,
        "spatial_upscaler_model_path": str(weights_path / "ltxv-spatial-upscaler-0.9.8.safetensors"),
        "stg_mode": "attention_values",
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.025,
        "text_encoder_model_name_or_path": str(weights_path),
        "precision": "bfloat16",
        "sampler": "from_checkpoint",
        "prompt_enhancement_words_threshold": 0,
        "stochastic_sampling": False,
        "first_pass": {
            "timesteps": [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725],
            "guidance_scale": 1,
            "stg_scale": 0,
            "rescaling_scale": 1,
            "skip_block_list": [42],
        },
        "second_pass": {
            "timesteps": [0.9094, 0.725, 0.4219],
            "guidance_scale": 1,
            "stg_scale": 0,
            "rescaling_scale": 1,
            "skip_block_list": [42],
        },
    }

    # Check if upscaler exists; fall back to single-pass if not
    upscaler_path = weights_path / "ltxv-spatial-upscaler-0.9.8.safetensors"
    if not upscaler_path.exists():
        logger.warning("Spatial upscaler not found, using single-pass pipeline")
        config["pipeline_type"] = "single"
        config.pop("spatial_upscaler_model_path", None)
        config.pop("downscale_factor", None)
        config.pop("second_pass", None)

    config_path = str(PROJECT_ROOT / "outputs" / f"ltx_{variant}_worker_config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


def load_model(variant: str):
    """Load the LTX pipeline."""
    global ltx_infer, InferenceConfig, pipeline_config_path

    logger.info(f"Loading LTX-Video ({variant}) on {torch.cuda.get_device_name(0)}")

    sys.path.insert(0, str(MODELS_DIR))
    from ltx_video.inference import infer as _infer, InferenceConfig as _IC

    ltx_infer = _infer
    InferenceConfig = _IC
    pipeline_config_path = create_local_config(variant)

    logger.info(f"LTX-Video ({variant}) ready with config: {pipeline_config_path}")


@app.get("/health")
async def health():
    return {
        "status": "ready" if ltx_infer is not None else "loading",
        "model": f"ltx_{model_variant}",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if ltx_infer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        output_dir = Path(req.output_dir or str(PROJECT_ROOT / "outputs" / "ltx"))
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = str(output_dir / f"ltx_{req.session_id or 'test'}_{int(time.time())}.mp4")

        start = time.time()

        config = InferenceConfig(
            pipeline_config=pipeline_config_path,
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            seed=req.seed,
            output_path=output_path,
        )
        ltx_infer(config=config)

        gen_time = time.time() - start

        # Find the actual output file (LTX may modify the filename)
        actual_output = output_path
        if not os.path.exists(output_path):
            mp4s = list(output_dir.glob("*.mp4"))
            if mp4s:
                actual_output = str(sorted(mp4s, key=os.path.getmtime)[-1])

        return GenerateResponse(
            status="success",
            output_path=actual_output,
            num_frames=req.num_frames,
            generation_time_s=gen_time,
        )

    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        return GenerateResponse(status="error", error=str(e))


@app.get("/status")
async def status():
    mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
    return {
        "model": f"ltx_{model_variant}",
        "loaded": ltx_infer is not None,
        "gpu_memory_gb": mem,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9102)
    parser.add_argument("--model-name", type=str, default="ltx_2b")
    args = parser.parse_args()

    model_variant = args.model_name
    logger.info(f"Starting LTX worker ({model_variant}) on port {args.port}")
    load_model(model_variant)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
