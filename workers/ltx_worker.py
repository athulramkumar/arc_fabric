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
    cache_start_step: Optional[int] = None
    cache_end_step: Optional[int] = None
    cache_interval: int = 3


class GenerateResponse(PydanticModel):
    status: str
    output_path: Optional[str] = None
    num_frames: int = 0
    generation_time_s: float = 0.0
    error: Optional[str] = None


def _resolve_config_yaml(variant: str) -> str:
    """Map model variant to its official config YAML in the repo."""
    CONFIG_MAP = {
        "ltx_2b": "ltxv-2b-0.9.8-distilled.yaml",
        "ltx_video": "ltxv-2b-0.9.8-distilled.yaml",
        "ltx_2b_dev": "ltxv-2b-0.9.6-dev.yaml",
        "ltx_13b": "ltxv-13b-0.9.8-distilled.yaml",
        "ltx_13b_dev": "ltxv-13b-0.9.8-dev.yaml",
    }
    return str(MODELS_DIR / "configs" / CONFIG_MAP.get(variant, CONFIG_MAP["ltx_2b"]))


def _resolve_weights(variant: str) -> tuple[str, Path]:
    """Map model variant to checkpoint path and weights directory."""
    VARIANT_MAP = {
        "ltx_2b": ("ltxv-2b-0.9.8-distilled", "ltxv-2b-0.9.8-distilled.safetensors"),
        "ltx_video": ("ltxv-2b-0.9.8-distilled", "ltxv-2b-0.9.8-distilled.safetensors"),
        "ltx_2b_dev": ("ltxv-2b-0.9.6-dev", "ltxv-2b-0.9.6-dev-04-25.safetensors"),
        "ltx_13b": ("ltxv-13b-0.9.8-distilled", "ltxv-13b-0.9.8-distilled.safetensors"),
        "ltx_13b_dev": ("ltxv-13b-0.9.8-dev", "ltxv-13b-0.9.8-dev.safetensors"),
    }
    subdir, ckpt_file = VARIANT_MAP.get(variant, VARIANT_MAP["ltx_2b"])
    weights_path = WEIGHTS_DIR / subdir
    return str(weights_path / ckpt_file), weights_path


def load_model(variant: str):
    """Load the LTX pipeline ONCE at startup using official config YAMLs."""
    global _pipeline, _pipeline_config, _skip_layer_strategy, _model_variant, _precision

    _model_variant = variant
    sys.path.insert(0, str(MODELS_DIR))

    from ltx_video.inference import (
        create_ltx_video_pipeline,
        load_pipeline_config,
        seed_everething,
        get_device,
    )
    from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

    ckpt_path, weights_path = _resolve_weights(variant)
    config_yaml = _resolve_config_yaml(variant)

    logger.info(f"Loading LTX-Video ({variant})")
    logger.info(f"  Config: {config_yaml}")
    logger.info(f"  Checkpoint: {ckpt_path}")

    pipe_config = load_pipeline_config(config_yaml)

    # Override paths to point to our local weights
    pipe_config["checkpoint_path"] = ckpt_path
    pipe_config["text_encoder_model_name_or_path"] = str(weights_path)
    pipe_config["prompt_enhancement_words_threshold"] = 0
    pipe_config["prompt_enhancer_image_caption_model_name_or_path"] = ""
    pipe_config["prompt_enhancer_llm_model_name_or_path"] = ""

    upscaler = weights_path / "ltxv-spatial-upscaler-0.9.8.safetensors"
    if "spatial_upscaler_model_path" in pipe_config and upscaler.exists():
        pipe_config["spatial_upscaler_model_path"] = str(upscaler)

    _precision = pipe_config.get("precision", "bfloat16")
    device = get_device()

    logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

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

    if pipe_config.get("pipeline_type") == "multi-scale" and upscaler.exists():
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

    # Remove keys that aren't pipeline __call__ args
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
            cache_start_step=req.cache_start_step,
            cache_end_step=req.cache_end_step,
            cache_interval=req.cache_interval,
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
