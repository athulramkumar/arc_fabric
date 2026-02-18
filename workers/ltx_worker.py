"""LTX-Video model worker - persistent FastAPI server with pre-loaded pipeline."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticModel

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

WORKER_DIR = Path(__file__).parent
PROJECT_ROOT = WORKER_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models" / "ltx_video"
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "ltx_video"

app = FastAPI(title="LTX-Video Worker")

# Global state: loaded once, reused for every generation
_pipeline = None
_pipeline_config = None
_skip_layer_strategy = None
_model_variant = None
_precision = None


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


def load_model(variant: str):
    """Load the LTX pipeline ONCE at startup."""
    global _pipeline, _pipeline_config, _skip_layer_strategy, _model_variant, _precision

    _model_variant = variant
    sys.path.insert(0, str(MODELS_DIR))

    from ltx_video.inference import (
        create_ltx_video_pipeline,
        load_pipeline_config,
        seed_everething,
        get_device,
    )
    from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
    from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

    if variant in ("ltx_2b", "ltx_video"):
        weights_subdir = "ltxv-2b-0.9.8-distilled"
        checkpoint_file = "ltxv-2b-0.9.8-distilled.safetensors"
    else:
        weights_subdir = "ltxv-13b-0.9.8-distilled"
        checkpoint_file = "ltxv-13b-0.9.8-distilled.safetensors"

    weights_path = WEIGHTS_DIR / weights_subdir
    ckpt_path = str(weights_path / checkpoint_file)

    pipe_config = {
        "checkpoint_path": ckpt_path,
        "text_encoder_model_name_or_path": str(weights_path),
        "precision": "bfloat16",
        "sampler": "from_checkpoint",
        "prompt_enhancement_words_threshold": 0,
        "prompt_enhancer_image_caption_model_name_or_path": "",
        "prompt_enhancer_llm_model_name_or_path": "",
        "stochastic_sampling": False,
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.025,
        "stg_mode": "attention_values",
    }

    upscaler = weights_path / "ltxv-spatial-upscaler-0.9.8.safetensors"
    if upscaler.exists():
        pipe_config.update({
            "pipeline_type": "multi-scale",
            "downscale_factor": 0.6666666,
            "spatial_upscaler_model_path": str(upscaler),
            "first_pass": {
                "timesteps": [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725],
                "guidance_scale": 1, "stg_scale": 0, "rescaling_scale": 1, "skip_block_list": [42],
            },
            "second_pass": {
                "timesteps": [0.9094, 0.725, 0.4219],
                "guidance_scale": 1, "stg_scale": 0, "rescaling_scale": 1, "skip_block_list": [42],
            },
        })
    else:
        pipe_config["pipeline_type"] = "single"
        pipe_config["first_pass"] = {
            "timesteps": [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725, 0.4219],
            "guidance_scale": 1, "stg_scale": 0, "rescaling_scale": 1, "skip_block_list": [42],
        }

    _precision = pipe_config["precision"]
    device = get_device()

    logger.info(f"Loading LTX-Video ({variant}) on {torch.cuda.get_device_name(0)}")

    pipeline = create_ltx_video_pipeline(
        ckpt_path=ckpt_path,
        precision=_precision,
        text_encoder_model_name_or_path=str(weights_path),
        sampler=pipe_config.get("sampler"),
        device=device,
        enhance_prompt=False,
        prompt_enhancer_image_caption_model_name_or_path="",
        prompt_enhancer_llm_model_name_or_path="",
    )

    if pipe_config.get("pipeline_type") == "multi-scale":
        from ltx_video.inference import create_latent_upsampler
        from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline
        latent_upsampler = create_latent_upsampler(str(upscaler), pipeline.device)
        pipeline = LTXMultiScalePipeline(pipeline, latent_upsampler=latent_upsampler)

    stg_mode = pipe_config.get("stg_mode", "attention_values")
    if stg_mode.lower() in ("stg_av", "attention_values"):
        _skip_layer_strategy = SkipLayerStrategy.AttentionValues
    elif stg_mode.lower() in ("stg_as", "attention_skip"):
        _skip_layer_strategy = SkipLayerStrategy.AttentionSkip
    else:
        _skip_layer_strategy = SkipLayerStrategy.AttentionValues

    # Remove keys that are not pipeline __call__ args
    for key in ("checkpoint_path", "text_encoder_model_name_or_path", "precision",
                "sampler", "prompt_enhancement_words_threshold",
                "pipeline_type", "spatial_upscaler_model_path",
                "prompt_enhancer_image_caption_model_name_or_path",
                "prompt_enhancer_llm_model_name_or_path", "stg_mode"):
        pipe_config.pop(key, None)

    _pipeline = pipeline
    _pipeline_config = pipe_config

    logger.info(f"LTX-Video ({variant}) loaded and ready")


@app.get("/health")
async def health():
    return {
        "status": "ready" if _pipeline is not None else "loading",
        "model": f"ltx_{_model_variant}",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from ltx_video.inference import calculate_padding, get_device, seed_everething, get_unique_filename

        output_dir = Path(req.output_dir or str(PROJECT_ROOT / "outputs" / "ltx"))
        output_dir.mkdir(parents=True, exist_ok=True)

        seed_everething(req.seed)
        device = get_device()

        height_padded = ((req.height - 1) // 32 + 1) * 32
        width_padded = ((req.width - 1) // 32 + 1) * 32
        num_frames_padded = ((req.num_frames - 2) // 8 + 1) * 8 + 1

        padding = calculate_padding(req.height, req.width, height_padded, width_padded)

        sample = {
            "prompt": req.prompt,
            "prompt_attention_mask": None,
            "negative_prompt": "",
            "negative_prompt_attention_mask": None,
        }

        generator = torch.Generator(device=device).manual_seed(req.seed)

        start = time.time()

        images = _pipeline(
            **_pipeline_config,
            skip_layer_strategy=_skip_layer_strategy,
            generator=generator,
            output_type="pt",
            callback_on_step_end=None,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=24,
            **sample,
            media_items=None,
            conditioning_items=None,
            is_video=True,
            vae_per_channel_normalize=True,
            image_cond_noise_scale=0.0,
            mixed_precision=(_precision == "mixed_precision"),
            offload_to_cpu=False,
            device=device,
            enhance_prompt=False,
        ).images

        (pad_left, pad_right, pad_top, pad_bottom) = padding
        pad_bottom = -pad_bottom if pad_bottom != 0 else images.shape[3]
        pad_right = -pad_right if pad_right != 0 else images.shape[4]
        images = images[:, :, :req.num_frames, pad_top:pad_bottom, pad_left:pad_right]

        video_np = images[0].permute(1, 2, 3, 0).cpu().float().numpy()
        video_np = (video_np * 255).astype(np.uint8)

        output_filename = get_unique_filename(
            "video_output_0", ".mp4",
            prompt=req.prompt, seed=req.seed,
            resolution=(req.height, req.width, req.num_frames),
            dir=output_dir,
        )

        with imageio.get_writer(str(output_filename), fps=24) as video:
            for frame in video_np:
                video.append_data(frame)

        gen_time = time.time() - start
        logger.info(f"Generated video in {gen_time:.1f}s -> {output_filename}")

        return GenerateResponse(
            status="success",
            output_path=str(output_filename),
            num_frames=req.num_frames,
            generation_time_s=gen_time,
        )

    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        return GenerateResponse(status="error", error=str(e))


@app.get("/status")
async def status():
    mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
    return {"model": f"ltx_{_model_variant}", "loaded": _pipeline is not None, "gpu_memory_gb": mem}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9102)
    parser.add_argument("--model-name", type=str, default="ltx_2b")
    args = parser.parse_args()

    _model_variant = args.model_name
    logger.info(f"Starting LTX worker ({args.model_name}) on port {args.port}")
    load_model(args.model_name)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
