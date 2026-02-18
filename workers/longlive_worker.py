"""LongLive model worker - thin FastAPI server wrapping the LongLive inference pipeline."""

import argparse
import os
import sys
import time
import logging
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticModel
from typing import Optional

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
WORKER_DIR = Path(__file__).parent
PROJECT_ROOT = WORKER_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models" / "longlive"
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "longlive"

app = FastAPI(title="LongLive Worker")

pipeline = None
text_encoder = None
vae = None
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
    """Load the LongLive pipeline."""
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

    # Load LoRA if configured
    if getattr(config, "adapter", None):
        from utils.lora_utils import configure_lora_for_model
        import peft
        pipeline.is_lora_enabled = False
        if configure_lora_for_model is not None:
            pipeline.generator.model = configure_lora_for_model(
                pipeline.generator.model,
                model_name="generator",
                lora_config=config.adapter,
                is_main_process=True,
            )
            if config.lora_ckpt:
                lora_state = torch.load(config.lora_ckpt, map_location="cpu")
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_state)
            pipeline.is_lora_enabled = True
            logger.info("LoRA adapter loaded")

    pipeline.generator = pipeline.generator.to(device).to(torch.bfloat16)
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
        from omegaconf import OmegaConf
        from utils.dataset import TextDataset
        from utils.misc import set_seed
        from torchvision.io import write_video
        from einops import rearrange

        set_seed(req.seed)

        output_dir = Path(req.output_dir or str(PROJECT_ROOT / "outputs" / "longlive"))
        output_dir.mkdir(parents=True, exist_ok=True)

        config = OmegaConf.load(str(MODELS_DIR / "configs" / "longlive_inference.yaml"))

        start = time.time()

        prompts = [req.prompt]
        context = pipeline.text_encoder(prompts)
        prompt_embeds = context["prompt_embeds"]

        negative = pipeline.text_encoder([""])
        negative_embeds = negative["prompt_embeds"]

        video = pipeline.generate(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            num_frames=req.num_frames,
            seed=req.seed,
        )

        gen_time = time.time() - start

        # Decode and save
        output_path = str(output_dir / f"longlive_{req.session_id or 'test'}_{int(time.time())}.mp4")

        if isinstance(video, torch.Tensor):
            video_np = video.cpu()
            if video_np.dim() == 4:
                video_np = rearrange(video_np, "c t h w -> t h w c")
            video_np = ((video_np + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            write_video(output_path, video_np, fps=16)

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
    return {
        "model": "longlive",
        "loaded": pipeline is not None,
        "gpu_memory_gb": mem,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--model-name", type=str, default="longlive")
    args = parser.parse_args()

    logger.info(f"Starting LongLive worker on port {args.port}")
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
