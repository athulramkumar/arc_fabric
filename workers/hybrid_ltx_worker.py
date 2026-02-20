"""Hybrid LTX-Video worker - loads both 13B and 2B transformers for hybrid schedule generation.

Uses the 13B model for initial high-noise steps (structure) and switches to the
2B model for refinement steps (detail).  The VAE, text encoder, tokenizer, and
scheduler are shared between both transformers.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, List

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

app = FastAPI(title="Hybrid LTX-Video Worker")

_pipeline = None
_pipeline_config = None
_skip_layer_strategy = None
_precision = None

_transformer_13b = None
_transformer_2b = None


class GenerateRequest(PydanticModel):
    prompt: str
    height: int = 480
    width: int = 704
    num_frames: int = 97
    seed: int = 42
    session_id: Optional[str] = None
    output_dir: Optional[str] = None
    schedule: Optional[List] = None  # e.g. [["13B", 7], ["2B", 3]]
    cache_start_step: Optional[int] = None
    cache_end_step: Optional[int] = None
    cache_interval: int = 3


class GenerateResponse(PydanticModel):
    status: str
    output_path: Optional[str] = None
    num_frames: int = 0
    generation_time_s: float = 0.0
    schedule_summary: Optional[str] = None
    error: Optional[str] = None


WEIGHTS_MAP = {
    "2b": ("ltxv-2b-0.9.8-distilled", "ltxv-2b-0.9.8-distilled.safetensors"),
    "13b": ("ltxv-13b-0.9.8-distilled", "ltxv-13b-0.9.8-distilled.safetensors"),
}


def _ckpt_path(variant_key: str) -> str:
    subdir, ckpt_file = WEIGHTS_MAP[variant_key]
    return str(WEIGHTS_DIR / subdir / ckpt_file)


def _weights_dir(variant_key: str) -> Path:
    subdir, _ = WEIGHTS_MAP[variant_key]
    return WEIGHTS_DIR / subdir


def load_models():
    """Load both 13B and 2B transformers, sharing everything else from the 2B pipeline."""
    global _pipeline, _pipeline_config, _skip_layer_strategy, _precision
    global _transformer_13b, _transformer_2b

    sys.path.insert(0, str(MODELS_DIR))

    from ltx_video.inference import (
        create_ltx_video_pipeline,
        create_transformer,
        load_pipeline_config,
        seed_everething,
        get_device,
    )
    from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

    device = get_device()
    weights_2b = _weights_dir("2b")

    # Load full pipeline from 2B (includes VAE, text encoder, tokenizer, scheduler)
    ckpt_2b = _ckpt_path("2b")
    config_yaml_2b = str(MODELS_DIR / "configs" / "ltxv-2b-0.9.8-distilled.yaml")

    logger.info("Loading LTX-Video 2B pipeline (VAE + text encoder + transformer)...")
    pipe_config = load_pipeline_config(config_yaml_2b)
    pipe_config["checkpoint_path"] = ckpt_2b
    pipe_config["text_encoder_model_name_or_path"] = str(weights_2b)
    pipe_config["prompt_enhancement_words_threshold"] = 0
    pipe_config["prompt_enhancer_image_caption_model_name_or_path"] = ""
    pipe_config["prompt_enhancer_llm_model_name_or_path"] = ""

    upscaler_2b = weights_2b / "ltxv-spatial-upscaler-0.9.8.safetensors"
    if "spatial_upscaler_model_path" in pipe_config and upscaler_2b.exists():
        pipe_config["spatial_upscaler_model_path"] = str(upscaler_2b)

    _precision = pipe_config.get("precision", "bfloat16")

    pipeline = create_ltx_video_pipeline(
        ckpt_path=ckpt_2b,
        precision=_precision,
        text_encoder_model_name_or_path=str(weights_2b),
        sampler=pipe_config.get("sampler"),
        device=device,
        enhance_prompt=False,
        prompt_enhancer_image_caption_model_name_or_path="",
        prompt_enhancer_llm_model_name_or_path="",
    )

    _transformer_2b = pipeline.transformer
    logger.info(f"2B transformer loaded: {sum(p.numel() for p in _transformer_2b.parameters()) / 1e9:.1f}B params")

    # Load 13B transformer separately
    ckpt_13b = _ckpt_path("13b")
    if os.path.exists(ckpt_13b):
        logger.info("Loading LTX-Video 13B transformer...")
        transformer_13b = create_transformer(ckpt_13b, _precision)
        _transformer_13b = transformer_13b.to(device)
        logger.info(f"13B transformer loaded: {sum(p.numel() for p in _transformer_13b.parameters()) / 1e9:.1f}B params")
    else:
        logger.warning(f"13B checkpoint not found at {ckpt_13b} — will use 2B only")
        _transformer_13b = None

    # Build the multi-scale pipeline if upscaler exists
    if pipe_config.get("pipeline_type") == "multi-scale" and upscaler_2b.exists():
        from ltx_video.inference import create_latent_upsampler
        from ltx_video.pipelines.pipeline_ltx_video import LTXMultiScalePipeline
        latent_upsampler = create_latent_upsampler(str(upscaler_2b), pipeline.device)
        pipeline = LTXMultiScalePipeline(pipeline, latent_upsampler=latent_upsampler)

    stg_mode = pipe_config.get("stg_mode", "attention_values")
    if stg_mode.lower() in ("stg_av", "attention_values"):
        _skip_layer_strategy = SkipLayerStrategy.AttentionValues
    elif stg_mode.lower() in ("stg_as", "attention_skip"):
        _skip_layer_strategy = SkipLayerStrategy.AttentionSkip
    else:
        _skip_layer_strategy = SkipLayerStrategy.AttentionValues

    for key in ("checkpoint_path", "text_encoder_model_name_or_path", "precision",
                "sampler", "prompt_enhancement_words_threshold",
                "pipeline_type", "spatial_upscaler_model_path",
                "prompt_enhancer_image_caption_model_name_or_path",
                "prompt_enhancer_llm_model_name_or_path", "stg_mode"):
        pipe_config.pop(key, None)

    _pipeline = pipeline
    _pipeline_config = pipe_config

    logger.info("Hybrid LTX-Video worker loaded and ready")


def _parse_schedule(schedule, total_steps: int):
    """Parse schedule into list of (variant_key, num_steps).

    If None, defaults to 70% 13B + 30% 2B (structure first, then refine).
    """
    if schedule is None:
        steps_13b = int(total_steps * 0.7)
        steps_2b = total_steps - steps_13b
        return [("13b", steps_13b), ("2b", steps_2b)]

    parsed = []
    for entry in schedule:
        variant = entry[0].lower().replace("-", "").replace("_", "")
        if "13" in variant:
            variant = "13b"
        elif "2" in variant:
            variant = "2b"
        parsed.append((variant, int(entry[1])))
    return parsed


def _get_inner_pipeline(pipeline):
    """Get the inner LTXVideoPipeline from a potential LTXMultiScalePipeline wrapper."""
    if hasattr(pipeline, "pipeline"):
        return pipeline.pipeline
    return pipeline


def _build_switch_callback(schedule, total_steps: int):
    """Build a callback that swaps the transformer at schedule boundaries.

    Returns (callback_fn, schedule_summary_str).
    """
    inner = _get_inner_pipeline(_pipeline)

    boundaries = []
    running = 0
    for variant, n_steps in schedule:
        boundaries.append((running, running + n_steps, variant))
        running += n_steps

    summary_parts = []
    for start, end, variant in boundaries:
        summary_parts.append(f"{variant.upper()} steps {start}-{end - 1}")
    summary = " → ".join(summary_parts)

    def _on_step_end(pipe, step_idx, timestep, callback_kwargs):
        next_step = step_idx + 1
        for start, end, variant in boundaries:
            if next_step == start:
                target = _transformer_13b if variant == "13b" else _transformer_2b
                if target is not None and inner.transformer is not target:
                    logger.info(f"Step {next_step}: switching to {variant.upper()} transformer")
                    inner.transformer = target
                break
        return callback_kwargs

    # Set the initial transformer for step 0
    if boundaries:
        initial_variant = boundaries[0][2]
        target = _transformer_13b if initial_variant == "13b" else _transformer_2b
        if target is not None:
            inner.transformer = target
            logger.info(f"Starting generation with {initial_variant.upper()} transformer")

    return _on_step_end, summary


@app.get("/health")
async def health():
    ready = _pipeline is not None and (_transformer_2b is not None)
    return {
        "status": "ready" if ready else "loading",
        "model": "hybrid_ltx",
        "transformers_loaded": {
            "2b": _transformer_2b is not None,
            "13b": _transformer_13b is not None,
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from ltx_video.inference import calculate_padding, get_device, seed_everething, get_unique_filename

        output_dir = Path(req.output_dir or str(PROJECT_ROOT / "outputs" / "hybrid_ltx"))
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

        # Determine total steps from pipeline config timesteps
        total_steps = len(_pipeline_config.get("timesteps", [0] * 8))
        if "first_pass" in _pipeline_config:
            total_steps = len(_pipeline_config["first_pass"].get("timesteps", [0] * 7))

        schedule = _parse_schedule(req.schedule, total_steps)
        callback, schedule_summary = _build_switch_callback(schedule, total_steps)

        logger.info(f"Hybrid schedule: {schedule_summary}")

        start = time.time()

        images = _pipeline(
            **_pipeline_config,
            skip_layer_strategy=_skip_layer_strategy,
            generator=generator,
            output_type="pt",
            callback_on_step_end=callback,
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
        logger.info(f"Generated hybrid video in {gen_time:.1f}s -> {output_filename}")

        # Restore 2B as default so the pipeline is in a clean state
        inner = _get_inner_pipeline(_pipeline)
        inner.transformer = _transformer_2b

        return GenerateResponse(
            status="success",
            output_path=str(output_filename),
            num_frames=req.num_frames,
            generation_time_s=gen_time,
            schedule_summary=schedule_summary,
        )

    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        # Restore 2B on error too
        try:
            inner = _get_inner_pipeline(_pipeline)
            inner.transformer = _transformer_2b
        except Exception:
            pass
        return GenerateResponse(status="error", error=str(e))


@app.get("/status")
async def status():
    mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
    return {
        "model": "hybrid_ltx",
        "loaded": _pipeline is not None,
        "transformers": {"2b": _transformer_2b is not None, "13b": _transformer_13b is not None},
        "gpu_memory_gb": mem,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9104)
    parser.add_argument("--model-name", type=str, default="hybrid_ltx")
    args = parser.parse_args()

    logger.info(f"Starting Hybrid LTX-Video worker on port {args.port}")
    load_models()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
