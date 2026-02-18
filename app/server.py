"""
Arc Fabric UI Server
A fal.ai-style web interface for video generation with multiple models.
Serves both the API and the static frontend.
"""

import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/workspace/arc_fabric")
OUTPUTS = ROOT / "outputs" / "ui"
OUTPUTS.mkdir(parents=True, exist_ok=True)

GPU_COUNT = int(os.environ.get("ARC_GPU_COUNT", "2"))


class ModelStatus(str, Enum):
    COLD = "cold"
    WARMING = "warming"
    WARM = "warm"
    ERROR = "error"


@dataclass
class ModelInfo:
    id: str
    display_name: str
    description: str
    conda_env: str
    gpu_memory_gb: float
    default_height: int
    default_width: int
    default_frames: int
    default_steps: int
    fps: int
    status: ModelStatus = ModelStatus.COLD
    gpu_id: Optional[int] = None
    last_used: float = 0.0
    error_msg: Optional[str] = None


MODELS: dict[str, ModelInfo] = {
    "wan21_1_3b": ModelInfo(
        id="wan21_1_3b",
        display_name="Wan 2.1 — 1.3B",
        description="Fast text-to-video diffusion model. Good quality at 480p with 50-step sampling.",
        conda_env=str(ROOT / "envs" / "af-wan21"),
        gpu_memory_gb=8.0,
        default_height=480, default_width=832,
        default_frames=33, default_steps=30, fps=16,
    ),
    "longlive": ModelInfo(
        id="longlive",
        display_name="LongLive — 1.3B",
        description="Autoregressive long-video generation with KV cache. LoRA-tuned for fast multi-step.",
        conda_env=str(ROOT / "envs" / "af-longlive"),
        gpu_memory_gb=25.0,
        default_height=480, default_width=832,
        default_frames=30, default_steps=4, fps=16,
    ),
    "ltx_2b": ModelInfo(
        id="ltx_2b",
        display_name="LTX-Video — 2B Distilled",
        description="Ultra-fast DiT with multi-scale pipeline. Sub-second per step at 704x480.",
        conda_env=str(ROOT / "envs" / "af-ltx"),
        gpu_memory_gb=26.0,
        default_height=480, default_width=704,
        default_frames=97, default_steps=8, fps=24,
    ),
}

gpu_assignments: dict[int, Optional[str]] = {i: None for i in range(GPU_COUNT)}


@dataclass
class Job:
    job_id: str
    model_id: str
    prompt: str
    height: int
    width: int
    num_frames: int
    seed: int
    status: str = "queued"
    progress: float = 0.0
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def elapsed(self) -> Optional[float]:
        if self.started_at:
            end = self.completed_at or time.time()
            return round(end - self.started_at, 1)
        return None


jobs: dict[str, Job] = {}

app = FastAPI(title="Arc Fabric", version="0.1.0")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")


class GenerateRequest(BaseModel):
    model_id: str
    prompt: str
    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: Optional[int] = None
    seed: int = 42
    gpu_id: Optional[int] = None


def _assign_gpu(model_id: str, preferred: Optional[int] = None) -> int:
    for gid, mid in gpu_assignments.items():
        if mid == model_id:
            return gid
    if preferred is not None and gpu_assignments.get(preferred) is None:
        gpu_assignments[preferred] = model_id
        return preferred
    for gid in range(GPU_COUNT):
        if gpu_assignments[gid] is None:
            gpu_assignments[gid] = model_id
            return gid
    lru_gid = min(
        range(GPU_COUNT),
        key=lambda g: MODELS[gpu_assignments[g]].last_used if gpu_assignments[g] else 0,
    )
    old_model = gpu_assignments[lru_gid]
    if old_model:
        logger.info(f"Evicting {old_model} from GPU {lru_gid}")
        MODELS[old_model].status = ModelStatus.COLD
        MODELS[old_model].gpu_id = None
    gpu_assignments[lru_gid] = model_id
    return lru_gid


def _run_generation(job: Job):
    model = MODELS[job.model_id]
    job.status = "running"
    job.started_at = time.time()
    model.status = ModelStatus.WARMING
    model.last_used = time.time()

    gpu_id = _assign_gpu(job.model_id)
    model.gpu_id = gpu_id

    job_dir = OUTPUTS / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        if job.model_id == "wan21_1_3b":
            _run_wan21(job, model, job_dir, env)
        elif job.model_id == "longlive":
            _run_longlive(job, model, job_dir, env)
        elif job.model_id == "ltx_2b":
            _run_ltx(job, model, job_dir, env)
        else:
            raise ValueError(f"Unknown model: {job.model_id}")

        model.status = ModelStatus.WARM
        job.status = "completed"
        job.completed_at = time.time()
        job.progress = 1.0

    except Exception as e:
        logger.exception(f"Job {job.job_id} failed")
        job.status = "failed"
        job.error = str(e)
        job.completed_at = time.time()
        model.status = ModelStatus.ERROR
        model.error_msg = str(e)


def _run_wan21(job: Job, model: ModelInfo, job_dir: Path, env: dict):
    """Use the proven generate.py CLI that already works."""
    python = str(Path(model.conda_env) / "bin" / "python")
    script = str(ROOT / "models" / "wan21" / "generate.py")
    output_file = str(job_dir / "output.mp4")

    cmd = [
        python, script,
        "--task", "t2v-1.3B",
        "--size", f"{job.height}*{job.width}",
        "--ckpt_dir", str(ROOT / "weights" / "wan21" / "Wan2.1-T2V-1.3B"),
        "--frame_num", str(job.num_frames),
        "--sample_steps", str(model.default_steps),
        "--base_seed", str(job.seed),
        "--prompt", job.prompt,
        "--save_file", output_file,
    ]
    _exec(cmd, env, job_dir, str(ROOT / "models" / "wan21"))
    job.output_path = f"/outputs/{job.job_id}/output.mp4"


def _run_longlive(job: Job, model: ModelInfo, job_dir: Path, env: dict):
    """Run LongLive via inline script - matching the tested worker approach."""
    python = str(Path(model.conda_env) / "bin" / "python")
    working_dir = str(ROOT / "models" / "longlive")

    prompt_escaped = job.prompt.replace("\\", "\\\\").replace("'", "\\'")
    output_mp4 = str(job_dir / "output.mp4")

    script = f"""
import sys, os, torch
sys.path.insert(0, '{working_dir}')
os.chdir('{working_dir}')
torch.set_grad_enabled(False)

from omegaconf import OmegaConf
from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from torchvision.io import write_video
from einops import rearrange

set_seed({job.seed})
device = torch.device('cuda')

config = OmegaConf.load('configs/longlive_inference.yaml')
config.distributed = False
config.output_folder = '{job_dir}'
config.num_output_frames = {job.num_frames}
config.num_samples = 1

pipeline = CausalInferencePipeline(config, device=device)

state_dict = torch.load(config.generator_ckpt, map_location='cpu')
key = 'generator_ema' if config.use_ema else 'generator'
raw = state_dict.get(key, state_dict.get('model'))
pipeline.generator.load_state_dict(raw)

from utils.lora_utils import configure_lora_for_model
import peft
pipeline.generator.model = configure_lora_for_model(
    pipeline.generator.model,
    model_name='generator',
    lora_config=config.adapter,
    is_main_process=True,
)
lora_ckpt = torch.load(config.lora_ckpt, map_location='cpu')
if isinstance(lora_ckpt, dict) and 'generator_lora' in lora_ckpt:
    peft.set_peft_model_state_dict(pipeline.generator.model, lora_ckpt['generator_lora'])
else:
    peft.set_peft_model_state_dict(pipeline.generator.model, lora_ckpt)
pipeline.is_lora_enabled = True

pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

prompts = ['{prompt_escaped}']
sampled_noise = torch.randn(
    [1, {job.num_frames}, 16, 60, 104], device=device, dtype=torch.bfloat16
)

video, latents = pipeline.inference(
    noise=sampled_noise,
    text_prompts=prompts,
    return_latents=True,
    low_memory=False,
    profile=False,
)
video_out = rearrange(video, 'b t c h w -> b t h w c').cpu()
video_out = (255.0 * video_out[0]).clamp(0, 255).to(torch.uint8)
write_video('{output_mp4}', video_out, fps={model.fps})
pipeline.vae.model.clear_cache()
print('ARCFABRIC_DONE')
"""
    script_path = job_dir / "run.py"
    script_path.write_text(script)
    cmd = [python, str(script_path)]
    _exec(cmd, env, job_dir, working_dir)

    if (job_dir / "output.mp4").exists():
        job.output_path = f"/outputs/{job.job_id}/output.mp4"
    else:
        mp4s = sorted(job_dir.glob("**/*.mp4"), key=os.path.getmtime)
        if mp4s:
            job.output_path = f"/outputs/{job.job_id}/{mp4s[-1].name}"


def _run_ltx(job: Job, model: ModelInfo, job_dir: Path, env: dict):
    """Run LTX-Video with dynamic config."""
    python = str(Path(model.conda_env) / "bin" / "python")
    working_dir = str(ROOT / "models" / "ltx_video")
    weights = ROOT / "weights" / "ltx_video" / "ltxv-2b-0.9.8-distilled"

    pipe_cfg = {
        "checkpoint_path": str(weights / "ltxv-2b-0.9.8-distilled.safetensors"),
        "text_encoder_model_name_or_path": str(weights),
        "precision": "bfloat16",
        "sampler": "from_checkpoint",
        "prompt_enhancement_words_threshold": 0,
        "stochastic_sampling": False,
        "decode_timestep": 0.05,
        "decode_noise_scale": 0.025,
        "stg_mode": "attention_values",
        "prompt_enhancer_image_caption_model_name_or_path": "",
        "prompt_enhancer_llm_model_name_or_path": "",
    }

    upscaler = weights / "ltxv-spatial-upscaler-0.9.8.safetensors"
    if upscaler.exists():
        pipe_cfg.update({
            "pipeline_type": "multi-scale",
            "downscale_factor": 0.6666666,
            "spatial_upscaler_model_path": str(upscaler),
            "first_pass": {"timesteps": [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725],
                           "guidance_scale": 1, "stg_scale": 0, "rescaling_scale": 1, "skip_block_list": [42]},
            "second_pass": {"timesteps": [0.9094, 0.725, 0.4219],
                            "guidance_scale": 1, "stg_scale": 0, "rescaling_scale": 1, "skip_block_list": [42]},
        })
    else:
        pipe_cfg["pipeline_type"] = "single"
        pipe_cfg["first_pass"] = {
            "timesteps": [1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725, 0.4219],
            "guidance_scale": 1, "stg_scale": 0, "rescaling_scale": 1, "skip_block_list": [42],
        }

    config_path = job_dir / "pipeline_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(pipe_cfg, f)

    prompt_escaped = job.prompt.replace("\\", "\\\\").replace("'", "\\'")
    output_mp4 = str(job_dir / "output.mp4")

    script = f"""
import sys, os
sys.path.insert(0, '{working_dir}')
os.chdir('{working_dir}')
from ltx_video.inference import infer, InferenceConfig
config = InferenceConfig(
    pipeline_config='{config_path}',
    prompt='{prompt_escaped}',
    height={job.height}, width={job.width},
    num_frames={job.num_frames}, seed={job.seed},
    output_path='{output_mp4}',
)
infer(config=config)
print('ARCFABRIC_DONE')
"""
    script_path = job_dir / "run.py"
    script_path.write_text(script)
    cmd = [python, str(script_path)]
    _exec(cmd, env, job_dir, working_dir)

    # LTX may write output with its own naming convention
    if (job_dir / "output.mp4").exists():
        job.output_path = f"/outputs/{job.job_id}/output.mp4"
    else:
        mp4s = sorted(job_dir.glob("**/*.mp4"), key=os.path.getmtime)
        if mp4s:
            rel = mp4s[-1].relative_to(job_dir)
            job.output_path = f"/outputs/{job.job_id}/{rel}"


def _exec(cmd: list[str], env: dict, job_dir: Path, cwd: str):
    log_path = job_dir / "log.txt"
    logger.info(f"Executing: {' '.join(cmd[:4])}...")
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
            cwd=cwd,
        )
        proc.wait()
    if proc.returncode != 0:
        log_text = log_path.read_text()[-3000:]
        raise RuntimeError(f"Process exited {proc.returncode}:\n{log_text}")


# ──────────────────────── API routes ────────────────────────

@app.get("/api/models")
async def api_models():
    return [
        {
            "id": m.id,
            "display_name": m.display_name,
            "description": m.description,
            "status": m.status.value,
            "gpu_id": m.gpu_id,
            "gpu_memory_gb": m.gpu_memory_gb,
            "defaults": {
                "height": m.default_height,
                "width": m.default_width,
                "num_frames": m.default_frames,
                "steps": m.default_steps,
                "fps": m.fps,
            },
        }
        for m in MODELS.values()
    ]


@app.get("/api/gpus")
async def api_gpus():
    return [
        {
            "gpu_id": gid,
            "model_id": gpu_assignments[gid],
            "model_name": MODELS[gpu_assignments[gid]].display_name if gpu_assignments[gid] else None,
            "status": "occupied" if gpu_assignments[gid] else "free",
        }
        for gid in range(GPU_COUNT)
    ]


@app.post("/api/generate")
async def api_generate(req: GenerateRequest, background: BackgroundTasks):
    if req.model_id not in MODELS:
        raise HTTPException(404, f"Unknown model: {req.model_id}")
    model = MODELS[req.model_id]
    job = Job(
        job_id=str(uuid.uuid4())[:8],
        model_id=req.model_id,
        prompt=req.prompt,
        height=req.height or model.default_height,
        width=req.width or model.default_width,
        num_frames=req.num_frames or model.default_frames,
        seed=req.seed,
    )
    jobs[job.job_id] = job
    background.add_task(_run_generation, job)
    return {"job_id": job.job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job.job_id,
        "model_id": job.model_id,
        "prompt": job.prompt,
        "status": job.status,
        "progress": job.progress,
        "output_path": job.output_path,
        "error": job.error,
        "elapsed": job.elapsed,
        "created_at": job.created_at,
    }


@app.get("/api/jobs")
async def api_jobs():
    return [
        {
            "job_id": j.job_id,
            "model_id": j.model_id,
            "prompt": j.prompt[:80],
            "status": j.status,
            "output_path": j.output_path,
            "elapsed": j.elapsed,
            "created_at": j.created_at,
        }
        for j in sorted(jobs.values(), key=lambda x: x.created_at, reverse=True)
    ]


@app.get("/api/logs/{job_id}")
async def api_logs(job_id: str):
    log_path = OUTPUTS / job_id / "log.txt"
    if not log_path.exists():
        raise HTTPException(404, "Log not found")
    text = log_path.read_text()
    return {"log": text[-5000:]}


@app.get("/")
async def index():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
