"""Wan 2.1 model worker - persistent FastAPI server wrapping the Wan generate pipeline."""

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
MODELS_DIR = PROJECT_ROOT / "models" / "wan21"
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "wan21"

app = FastAPI(title="Wan 2.1 Worker")

wan_pipe = None
device = None
_model_variant = "t2v-1.3B"


class GenerateRequest(PydanticModel):
    prompt: str
    num_frames: int = 33
    height: int = 480
    width: int = 832
    seed: int = 42
    session_id: Optional[str] = None
    output_dir: Optional[str] = None


class GenerateResponse(PydanticModel):
    status: str
    output_path: Optional[str] = None
    num_frames: int = 0
    generation_time_s: float = 0.0
    error: Optional[str] = None


def load_model(variant: str = "t2v-1.3B"):
    global wan_pipe, device, _model_variant
    _model_variant = variant

    device = torch.device("cuda")
    logger.info(f"Loading Wan 2.1 ({variant}) on {torch.cuda.get_device_name(0)}")
    logger.info(f"Free memory: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB")

    sys.path.insert(0, str(MODELS_DIR))
    os.chdir(str(MODELS_DIR))

    import wan
    from wan.configs import WAN_CONFIGS

    ckpt_map = {
        "t2v-1.3B": str(WEIGHTS_DIR / "Wan2.1-T2V-1.3B"),
        "t2v-14B": str(WEIGHTS_DIR / "Wan2.1-T2V-14B"),
    }

    cfg = WAN_CONFIGS[variant]
    wan_pipe = wan.WanT2V(
        config=cfg,
        checkpoint_dir=ckpt_map[variant],
        device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False,
    )
    logger.info(f"Wan 2.1 ({variant}) loaded successfully")


@app.get("/health")
async def health():
    return {
        "status": "ready" if wan_pipe is not None else "loading",
        "model": f"wan21_{_model_variant}",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if wan_pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        output_dir = Path(req.output_dir or str(PROJECT_ROOT / "outputs" / "wan21"))
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = str(output_dir / "output.mp4")

        start = time.time()

        from wan.utils.utils import cache_video

        video = wan_pipe.generate(
            req.prompt,
            size=(req.width, req.height),
            frame_num=req.num_frames,
            shift=5.0,
            sample_solver="unipc",
            sampling_steps=30,
            guide_scale=5.0,
            seed=req.seed,
            offload_model=False,
        )

        cache_video(
            tensor=video[None],
            save_file=output_path,
            fps=16, nrow=1, normalize=True, value_range=(-1, 1),
        )

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
    return {
        "model": f"wan21_{_model_variant}",
        "loaded": wan_pipe is not None,
        "gpu_memory_gb": mem,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--model-name", type=str, default="wan21_1_3b")
    args = parser.parse_args()

    variant = "t2v-14B" if "14b" in args.model_name else "t2v-1.3B"
    logger.info(f"Starting Wan 2.1 worker ({variant}) on port {args.port}")
    load_model(variant)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
