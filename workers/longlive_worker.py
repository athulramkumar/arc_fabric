"""LongLive model worker - persistent FastAPI server wrapping the LongLive inference pipeline."""

import argparse
import os
import sys
import time
import logging
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
MODELS_DIR = PROJECT_ROOT / "models" / "longlive"
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "longlive"

app = FastAPI(title="LongLive Worker")

pipeline = None
device = None


class GenerateRequest(PydanticModel):
    prompt: str
    num_frames: int = 30
    seed: int = 0
    session_id: Optional[str] = None
    output_dir: Optional[str] = None


class GenerateResponse(PydanticModel):
    status: str
    output_path: Optional[str] = None
    num_frames: int = 0
    generation_time_s: float = 0.0
    error: Optional[str] = None


def setup_symlinks():
    """Create symlinks that LongLive expects."""
    os.chdir(str(MODELS_DIR))

    wan_link = MODELS_DIR / "wan_models" / "Wan2.1-T2V-1.3B"
    wan_link.parent.mkdir(parents=True, exist_ok=True)
    if not wan_link.exists():
        wan_link.symlink_to(WEIGHTS_DIR / "Wan2.1-T2V-1.3B")

    ll_link = MODELS_DIR / "longlive_models"
    if not ll_link.exists():
        ll_link.symlink_to(WEIGHTS_DIR / "LongLive")


def load_model():
    """Load the LongLive pipeline with LoRA."""
    global pipeline, device

    device = torch.device("cuda")
    logger.info(f"Loading LongLive on {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Free memory: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB")

    setup_symlinks()

    sys.path.insert(0, str(MODELS_DIR))
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline

    config = OmegaConf.load(str(MODELS_DIR / "configs" / "longlive_inference.yaml"))
    config.distributed = False

    torch.set_grad_enabled(False)

    pipeline = CausalInferencePipeline(config, device=device)

    # Load generator checkpoint
    if config.generator_ckpt:
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        if "generator" in state_dict or "generator_ema" in state_dict:
            key = "generator_ema" if config.use_ema else "generator"
            raw = state_dict[key]
        elif "model" in state_dict:
            raw = state_dict["model"]
        else:
            raise ValueError("Generator state dict not found")
        pipeline.generator.load_state_dict(raw)

    # Load LoRA
    if getattr(config, "adapter", None):
        from utils.lora_utils import configure_lora_for_model
        import peft

        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=True,
        )
        if config.lora_ckpt:
            lora_ckpt = torch.load(config.lora_ckpt, map_location="cpu")
            if isinstance(lora_ckpt, dict) and "generator_lora" in lora_ckpt:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_ckpt["generator_lora"])
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_ckpt)
        pipeline.is_lora_enabled = True
        logger.info("LoRA adapter loaded")

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    logger.info("LongLive pipeline loaded successfully")


@app.get("/health")
async def health():
    return {
        "status": "ready" if pipeline is not None else "loading",
        "model": "longlive",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from utils.misc import set_seed
        from torchvision.io import write_video
        from einops import rearrange

        set_seed(req.seed)

        output_dir = Path(req.output_dir or str(PROJECT_ROOT / "outputs" / "longlive"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / "output.mp4")

        start = time.time()

        prompts = [req.prompt]
        sampled_noise = torch.randn(
            [1, req.num_frames, 16, 60, 104], device=device, dtype=torch.bfloat16,
        )

        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            low_memory=False,
            profile=False,
        )

        video_out = rearrange(video, "b t c h w -> b t h w c").cpu()
        video_out = (255.0 * video_out[0]).clamp(0, 255).to(torch.uint8)
        write_video(output_path, video_out, fps=16)

        pipeline.vae.model.clear_cache()

        gen_time = time.time() - start
        logger.info(f"Generated {req.num_frames} frames in {gen_time:.1f}s")

        return GenerateResponse(
            status="success",
            output_path=output_path,
            num_frames=req.num_frames,
            generation_time_s=gen_time,
        )

    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        return GenerateResponse(status="error", error=str(e))


@app.get("/status")
async def status():
    mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
    return {"model": "longlive", "loaded": pipeline is not None, "gpu_memory_gb": mem}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--model-name", type=str, default="longlive")
    args = parser.parse_args()

    logger.info(f"Starting LongLive worker on port {args.port}")
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
