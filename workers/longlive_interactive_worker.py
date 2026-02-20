"""LongLive Interactive Worker - persistent FastAPI server wrapping VideoBuilderState
for chunk-by-chunk video generation with KV cache continuity between chunks.

The model is loaded once at startup. Each session creates a VideoBuilderState instance
that supports interactive grounding → chunk1 → chunk2 → ... → finalize workflows.
Users can change prompts between chunks while maintaining visual continuity.

The /generate endpoint also supports a "simple mode" compatible with the standard
Arc Fabric generation flow: set grounding, generate one chunk, return the video.
"""

import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Optional

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

sys.path.insert(0, str(MODELS_DIR))
os.chdir(str(MODELS_DIR))

app = FastAPI(title="LongLive Interactive Worker")

builder: Optional["VideoBuilderState"] = None
_is_model_loaded = False


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SetupRequest(PydanticModel):
    config_path: str = "configs/longlive_inference.yaml"
    output_dir: str = "videos/interactive_web"
    chunk_duration: float = 10.0
    max_chunks: int = 12
    seed: int = 42


class GroundingRequest(PydanticModel):
    grounding: str
    skip_ai: bool = False


class GenerateChunkRequest(PydanticModel):
    user_prompt: str
    processed_prompt: Optional[str] = None
    skip_ai: bool = False


class EnhanceChunkRequest(PydanticModel):
    user_prompt: str


class GenerateRequest(PydanticModel):
    """Standard Arc Fabric generate interface — simple mode.

    Performs grounding + single chunk in one call for compatibility with
    the orchestrator's generate flow.
    """
    prompt: str
    seed: int = 42
    session_id: Optional[str] = None
    output_dir: Optional[str] = None
    chunk_duration: float = 10.0
    max_chunks: int = 12


class GenerateResponse(PydanticModel):
    status: str
    output_path: Optional[str] = None
    num_frames: int = 0
    generation_time_s: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_symlinks():
    """Create symlinks that LongLive expects for weight directories."""
    wan_link = MODELS_DIR / "wan_models" / "Wan2.1-T2V-1.3B"
    wan_link.parent.mkdir(parents=True, exist_ok=True)
    if not wan_link.exists():
        wan_target = WEIGHTS_DIR / "Wan2.1-T2V-1.3B"
        logger.info(f"Creating symlink {wan_link} -> {wan_target}")
        wan_link.symlink_to(wan_target)

    ll_link = MODELS_DIR / "longlive_models"
    if not ll_link.exists():
        ll_target = WEIGHTS_DIR / "LongLive"
        logger.info(f"Creating symlink {ll_link} -> {ll_target}")
        ll_link.symlink_to(ll_target)


def _ensure_builder() -> "VideoBuilderState":
    if builder is None:
        raise HTTPException(status_code=503, detail="No active session. Call /setup first.")
    return builder


def _ensure_setup() -> "VideoBuilderState":
    b = _ensure_builder()
    if not b.is_setup:
        raise HTTPException(status_code=400, detail="Session not initialised. Call /setup first.")
    return b


def _ensure_grounding() -> "VideoBuilderState":
    b = _ensure_setup()
    if not b.grounding_set:
        raise HTTPException(status_code=400, detail="Grounding not set. Call /grounding first.")
    return b


def _get_anthropic_key() -> str:
    """Get Anthropic API key from environment or bash_aliases."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        bash_aliases = "/root/.bash_aliases"
        if os.path.exists(bash_aliases):
            with open(bash_aliases) as f:
                for line in f:
                    if "ANTHROPIC_API_KEY" in line and "export" in line:
                        api_key = line.split("=")[1].strip().strip('"\'')
                        break
    return api_key or ""


_CLAUDE_MODELS = ["claude-sonnet-4-20250514", "claude-3-haiku-20240307"]
_CLAUDE_MAX_RETRIES = 3
_CLAUDE_RETRY_BASE_DELAY = 2.0


def _patch_enhancer(enhancer) -> None:
    """Replace the enhancer's _call_claude with a version that has proper
    retry logic and model fallback so 529 overloaded errors are handled."""
    import anthropic as _anthropic

    original_system_prompt = enhancer.SYSTEM_PROMPT
    client = enhancer.client

    def _robust_call_claude(message: str) -> str:
        last_err = None
        for model in _CLAUDE_MODELS:
            for attempt in range(_CLAUDE_MAX_RETRIES):
                try:
                    resp = client.messages.create(
                        model=model,
                        max_tokens=1000,
                        system=original_system_prompt,
                        messages=[{"role": "user", "content": message}],
                    )
                    return resp.content[0].text.strip()
                except _anthropic.APIStatusError as e:
                    last_err = e
                    if e.status_code == 529:
                        delay = _CLAUDE_RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            f"Claude {model} overloaded (529), "
                            f"retry {attempt+1}/{_CLAUDE_MAX_RETRIES} in {delay:.1f}s"
                        )
                        time.sleep(delay)
                        continue
                    logger.warning(f"Claude {model} error {e.status_code}: {e}")
                    break
                except Exception as e:
                    last_err = e
                    logger.warning(f"Claude {model} unexpected error: {e}")
                    break
            logger.info(f"Model {model} exhausted, trying next fallback")

        logger.error(f"All Claude models failed: {last_err}")
        return ""

    enhancer._call_claude = _robust_call_claude
    logger.info(f"Patched enhancer with robust _call_claude (models: {_CLAUDE_MODELS})")


def _create_builder(
    config_path: str = "configs/longlive_inference.yaml",
    output_dir: str = "videos/interactive_web",
    chunk_duration: float = 10.0,
    max_chunks: int = 12,
    seed: int = 42,
) -> "VideoBuilderState":
    """Instantiate a new VideoBuilderState with the Anthropic key for
    AI prompt enhancement."""
    from web_ui.video_builder_state import VideoBuilderState

    return VideoBuilderState(
        config_path=config_path,
        output_dir=output_dir,
        anthropic_key=_get_anthropic_key(),
        chunk_duration=chunk_duration,
        max_chunks=max_chunks,
        seed=seed,
        progress_callback=lambda msg, pct: logger.info(f"[progress {pct:.0%}] {msg}"),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ready",
        "model": "longlive_interactive",
        "model_loaded": _is_model_loaded,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "gpu_memory_allocated_gb": (
            round(torch.cuda.memory_allocated(0) / 1e9, 2)
            if torch.cuda.is_available()
            else 0
        ),
        "has_session": builder is not None,
        "session_ready": builder is not None and builder.is_setup,
    }


@app.post("/setup")
async def setup(req: SetupRequest = SetupRequest()):
    """Initialise a new session and load the model (first call) or reuse the
    already-loaded model for subsequent sessions."""
    global builder, _is_model_loaded

    logger.info(
        f"Setting up session: chunk_duration={req.chunk_duration}, "
        f"max_chunks={req.max_chunks}, seed={req.seed}"
    )

    builder = _create_builder(
        config_path=req.config_path,
        output_dir=req.output_dir,
        chunk_duration=req.chunk_duration,
        max_chunks=req.max_chunks,
        seed=req.seed,
    )

    try:
        result = builder.setup()
        _is_model_loaded = True
        if builder.enhancer:
            _patch_enhancer(builder.enhancer)
        logger.info(f"Session ready: {result.get('session_id')}")
        return result
    except Exception as e:
        logger.exception("Setup failed")
        builder = None
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grounding")
async def set_grounding(req: GroundingRequest):
    """Set the initial scene description (grounding) for the video.

    When skip_ai is False, the grounding is enhanced via Claude and the
    enhanced (structured JSON) version is returned for the user to review.
    The caller should then POST /accept_grounding with the final text.

    When skip_ai is True, the grounding is accepted immediately.
    """
    b = _ensure_setup()

    try:
        grounding_result = b.set_grounding(req.grounding, skip_ai=req.skip_ai)
        enhanced = grounding_result["enhanced"]

        if req.skip_ai:
            b.accept_grounding(enhanced)

        logger.info(f"Grounding set (skip_ai={req.skip_ai}): {req.grounding[:80]}...")
        return {
            "status": "success",
            "needs_review": not req.skip_ai,
            **grounding_result,
        }
    except Exception as e:
        logger.exception("Grounding failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/accept_grounding")
async def accept_grounding(req: dict):
    """Accept a (possibly edited) enhanced grounding prompt."""
    b = _ensure_setup()
    enhanced = req.get("enhanced", "")
    if not enhanced:
        raise HTTPException(status_code=400, detail="Missing 'enhanced' field")
    try:
        b.accept_grounding(enhanced)
        logger.info(f"Grounding accepted: {enhanced[:80]}...")
        return {"status": "success"}
    except Exception as e:
        logger.exception("Accept grounding failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/regenerate_grounding")
async def regenerate_grounding(req: dict):
    """Regenerate the AI-enhanced grounding."""
    b = _ensure_setup()
    grounding = req.get("grounding", "")
    if not grounding:
        raise HTTPException(status_code=400, detail="Missing 'grounding' field")
    try:
        enhanced = b.regenerate_grounding(grounding)
        return {"status": "success", "enhanced": enhanced}
    except Exception as e:
        logger.exception("Regenerate grounding failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enhance_chunk")
async def enhance_chunk(req: EnhanceChunkRequest):
    """Enhance a chunk prompt via Claude without generating video.

    Returns the structured JSON prompt for the user to review/edit.
    """
    b = _ensure_grounding()
    try:
        enhanced = b.enhance_chunk_prompt(req.user_prompt)
        logger.info(f"Chunk prompt enhanced: {req.user_prompt[:60]}...")
        return {"status": "success", "enhanced": enhanced, "user_prompt": req.user_prompt}
    except Exception as e:
        logger.exception("Enhance chunk failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/regenerate_chunk_prompt")
async def regenerate_chunk_prompt(req: EnhanceChunkRequest):
    """Regenerate the AI-enhanced chunk prompt."""
    b = _ensure_grounding()
    try:
        enhanced = b.regenerate_chunk_prompt(req.user_prompt)
        return {"status": "success", "enhanced": enhanced}
    except Exception as e:
        logger.exception("Regenerate chunk prompt failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_chunk")
async def generate_chunk(req: GenerateChunkRequest):
    """Generate the next video chunk. Prompt can differ from previous chunks.

    If processed_prompt is structured JSON (from AI enhancement), it is
    converted to a flat text prompt for the video model while the structured
    state is stored for continuity.
    """
    b = _ensure_grounding()

    processed = req.processed_prompt or req.user_prompt

    import json as _json

    video_prompt = processed
    try:
        state = _json.loads(processed)
        if isinstance(state, dict) and "subject" in state:
            video_prompt = b.enhancer._structured_to_video_prompt(state)
            if state not in b.enhancer.structured_states:
                b.enhancer.structured_states.append(state)
            logger.info(f"Converted JSON prompt to flat: {video_prompt[:80]}...")
    except (_json.JSONDecodeError, TypeError, AttributeError):
        pass

    try:
        result = b.generate_chunk(
            user_prompt=req.user_prompt,
            processed_prompt=video_prompt,
            skip_ai=req.skip_ai,
        )
        logger.info(
            f"Chunk {result['chunk_num']} generated in {result['generation_time']}"
        )
        return result
    except Exception as e:
        logger.exception("Chunk generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Simple-mode generate compatible with the standard Arc Fabric flow.

    1. Creates / resets a session
    2. Sets grounding from the prompt
    3. Generates a single chunk
    4. Returns the chunk video path
    """
    global builder, _is_model_loaded

    start = time.time()

    try:
        if builder is None or not builder.is_setup:
            builder = _create_builder(
                output_dir=req.output_dir or str(PROJECT_ROOT / "outputs" / "longlive"),
                chunk_duration=req.chunk_duration,
                max_chunks=req.max_chunks,
                seed=req.seed,
            )
            builder.setup()
            _is_model_loaded = True
        else:
            builder.reset()

        grounding_result = builder.set_grounding(req.prompt, skip_ai=True)
        builder.accept_grounding(grounding_result["enhanced"])

        chunk_result = builder.generate_chunk(
            user_prompt=req.prompt,
            processed_prompt=req.prompt,
            skip_ai=True,
        )

        gen_time = time.time() - start
        output_path = chunk_result.get("chunk_video", "")
        fps = builder.fps
        frames = builder.latent_frames_per_chunk * builder.temporal_upsample

        logger.info(f"Simple-mode generate done in {gen_time:.1f}s -> {output_path}")

        return GenerateResponse(
            status="success",
            output_path=output_path,
            num_frames=frames,
            generation_time_s=round(gen_time, 2),
        )

    except Exception as e:
        logger.exception("Simple-mode generate failed")
        return GenerateResponse(status="error", error=str(e))


@app.post("/go_back")
async def go_back():
    """Go back one chunk so it can be re-generated with a different prompt."""
    b = _ensure_grounding()

    try:
        result = b.go_back()
        logger.info(f"Went back: {result}")
        return result
    except Exception as e:
        logger.exception("Go-back failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/finalize")
async def finalize():
    """Export the final assembled video from all generated chunks."""
    b = _ensure_grounding()

    try:
        result = b.finalize()
        logger.info(f"Finalised: {result.get('final_video')}")
        return result
    except Exception as e:
        logger.exception("Finalize failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset():
    """Reset the session for a new video while keeping the model loaded."""
    b = _ensure_setup()

    try:
        result = b.reset()
        logger.info(f"Session reset: {result.get('new_session_id')}")
        return result
    except Exception as e:
        logger.exception("Reset failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status():
    """Return current session status and GPU memory usage."""
    mem = (
        round(torch.cuda.memory_allocated(0) / 1e9, 2)
        if torch.cuda.is_available()
        else 0
    )

    if builder is not None and builder.is_setup:
        session_status = builder.get_status()
    else:
        session_status = {
            "is_setup": False,
            "grounding_set": False,
            "grounding": None,
            "current_chunk": 0,
            "max_chunks": 0,
            "session_id": None,
            "session_dir": None,
            "chunks_generated": 0,
        }

    return {
        "model": "longlive_interactive",
        "model_loaded": _is_model_loaded,
        "gpu_memory_gb": mem,
        **session_status,
    }


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongLive Interactive Worker")
    parser.add_argument("--port", type=int, default=9102)
    parser.add_argument("--model-name", type=str, default="longlive_interactive")
    parser.add_argument("--preload", action="store_true",
                        help="Load model at startup instead of waiting for first /setup call")
    args = parser.parse_args()

    logger.info(f"Starting LongLive Interactive worker on port {args.port}")

    setup_symlinks()

    if args.preload:
        logger.info("Pre-loading model via /setup defaults...")
        builder = _create_builder()
        result = builder.setup()
        _is_model_loaded = True
        logger.info(f"Pre-load complete: {result}")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
