"""Hybrid Wan 2.1 worker - loads both 14B and 1.3B for hybrid schedule generation."""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, List

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

app = FastAPI(title="Hybrid Wan 2.1 Worker")

model_manager = None


class GenerateRequest(PydanticModel):
    prompt: str
    num_frames: int = 81
    height: int = 480
    width: int = 832
    seed: int = 42
    sampling_steps: int = 50
    guidance_scale: float = 5.0
    shift: float = 5.0
    schedule: Optional[List] = None  # e.g. [["14B", 15], ["1.3B", 35]]
    enable_caching: bool = False
    cache_start_step: int = 10
    cache_end_step: Optional[int] = 40
    cache_interval: int = 3
    session_id: Optional[str] = None
    output_dir: Optional[str] = None


class GenerateResponse(PydanticModel):
    status: str
    output_path: Optional[str] = None
    num_frames: int = 0
    generation_time_s: float = 0.0
    schedule_summary: Optional[str] = None
    cache_statistics: Optional[dict] = None
    error: Optional[str] = None


def _patch_config_paths():
    """
    Patch api.config checkpoint paths before any other api modules are imported.

    config.py computes CHECKPOINT_DIR_14B / _1_3B relative to its own BASE_DIR
    (models/wan21), but our weights live under PROJECT_ROOT/weights/wan21.
    We must patch the module-level constants *before* model_manager is imported
    because model_manager copies them via ``from .config import …``.
    """
    sys.path.insert(0, str(MODELS_DIR))

    import api.config as api_cfg

    api_cfg.CHECKPOINT_DIR_14B = str(WEIGHTS_DIR / "Wan2.1-T2V-14B")
    api_cfg.CHECKPOINT_DIR_1_3B = str(WEIGHTS_DIR / "Wan2.1-T2V-1.3B")

    logger.info(f"Patched CHECKPOINT_DIR_14B -> {api_cfg.CHECKPOINT_DIR_14B}")
    logger.info(f"Patched CHECKPOINT_DIR_1_3B -> {api_cfg.CHECKPOINT_DIR_1_3B}")


def load_models():
    global model_manager

    os.chdir(str(MODELS_DIR))

    # Patch checkpoint paths before model_manager copies them at import time
    _patch_config_paths()

    from api.model_manager import ModelManager
    from api.config import CHECKPOINT_DIR_14B, CHECKPOINT_DIR_1_3B

    logger.info(f"Loading both Wan 2.1 models on {torch.cuda.get_device_name(0)}")
    logger.info(f"14B checkpoint: {CHECKPOINT_DIR_14B}")
    logger.info(f"1.3B checkpoint: {CHECKPOINT_DIR_1_3B}")

    model_manager = ModelManager(device_id=0)
    model_manager.load_models()

    logger.info("Both models loaded successfully")


@app.get("/health")
async def health():
    return {
        "status": "ready" if model_manager and model_manager.is_loaded else "loading",
        "model": "hybrid_wan21",
        "loaded_models": model_manager.loaded_models if model_manager else [],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "gpu_memory_allocated_gb": (
            torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        ),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        from api.generator import generate_video
        from api.utils.scheduling import parse_schedule, get_schedule_summary

        output_dir = Path(req.output_dir or str(PROJECT_ROOT / "outputs" / "hybrid_wan"))
        output_dir.mkdir(parents=True, exist_ok=True)

        import api.config as api_cfg
        original_output_dir = api_cfg.OUTPUT_DIR
        api_cfg.OUTPUT_DIR = str(output_dir)

        start = time.time()

        result = generate_video(
            model_manager=model_manager,
            prompt=req.prompt,
            model_type="hybrid",
            schedule=req.schedule,
            width=req.width,
            height=req.height,
            frame_count=req.num_frames,
            fps=16,
            sampling_steps=req.sampling_steps,
            guidance_scale=req.guidance_scale,
            shift=req.shift,
            enable_caching=req.enable_caching,
            cache_start_step=req.cache_start_step,
            cache_end_step=req.cache_end_step,
            cache_interval=req.cache_interval,
            seed=req.seed,
            job_id=f"hybrid_{int(time.time())}",
        )

        api_cfg.OUTPUT_DIR = original_output_dir

        gen_time = time.time() - start

        if result.success:
            output_path = result.video_path
            dest = str(output_dir / "output.mp4")
            if output_path != dest:
                import shutil
                shutil.copy2(output_path, dest)
                output_path = dest

            if req.schedule:
                sched = parse_schedule(req.schedule)
            else:
                steps_14B = int(req.sampling_steps * 0.3)
                steps_1_3B = req.sampling_steps - steps_14B
                sched = [("14B", steps_14B), ("1.3B", steps_1_3B)]
            summary = get_schedule_summary(sched)

            return GenerateResponse(
                status="success",
                output_path=output_path,
                num_frames=req.num_frames,
                generation_time_s=gen_time,
                schedule_summary=summary,
                cache_statistics=result.cache_statistics,
            )
        else:
            return GenerateResponse(status="error", error=result.error)

    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        return GenerateResponse(status="error", error=str(e))


@app.get("/status")
async def status():
    mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
    return {
        "model": "hybrid_wan21",
        "loaded": model_manager.is_loaded if model_manager else False,
        "loaded_models": model_manager.loaded_models if model_manager else [],
        "gpu_memory_gb": mem,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9103)
    parser.add_argument("--model-name", type=str, default="hybrid_wan21")
    args = parser.parse_args()

    logger.info(f"Starting Hybrid Wan 2.1 worker on port {args.port}")
    load_models()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
