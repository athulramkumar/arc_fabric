"""
Arc Fabric UI Server
Persistent model workers stay loaded in GPU between generations.
Each model runs as a long-lived FastAPI subprocess in its own conda env.
"""

import logging
import os
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import requests as http_requests
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
WORKER_BASE_PORT = 9100


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
    worker_script: str
    status: ModelStatus = ModelStatus.COLD
    gpu_id: Optional[int] = None
    worker_port: Optional[int] = None
    worker_proc: Optional[subprocess.Popen] = None
    last_used: float = 0.0
    error_msg: Optional[str] = None


MODELS: dict[str, ModelInfo] = {
    # ── Wan 2.1 family ──
    "wan21_1_3b": ModelInfo(
        id="wan21_1_3b",
        display_name="Wan 2.1 — 1.3B",
        description="Fast text-to-video diffusion model. Good quality at 480p with 30-step sampling.",
        conda_env=str(ROOT / "envs" / "af-wan21"),
        gpu_memory_gb=8.0,
        default_height=480, default_width=832,
        default_frames=33, default_steps=30, fps=16,
        worker_script=str(ROOT / "workers" / "wan21_worker.py"),
    ),
    "wan21_14b": ModelInfo(
        id="wan21_14b",
        display_name="Wan 2.1 — 14B",
        description="High-quality 14B text-to-video model. Slower but significantly better detail and coherence.",
        conda_env=str(ROOT / "envs" / "af-wan21"),
        gpu_memory_gb=40.0,
        default_height=480, default_width=832,
        default_frames=81, default_steps=50, fps=16,
        worker_script=str(ROOT / "workers" / "wan21_worker.py"),
    ),
    "hybrid_wan21": ModelInfo(
        id="hybrid_wan21",
        display_name="Wan 2.1 — Hybrid (14B+1.3B)",
        description="Hybrid schedule: 14B for structure (30%), 1.3B for refinement (70%). Best quality-speed tradeoff.",
        conda_env=str(ROOT / "envs" / "af-wan21"),
        gpu_memory_gb=50.0,
        default_height=480, default_width=832,
        default_frames=81, default_steps=50, fps=16,
        worker_script=str(ROOT / "workers" / "hybrid_wan_worker.py"),
    ),
    # ── LongLive family ──
    "longlive": ModelInfo(
        id="longlive",
        display_name="LongLive — 1.3B",
        description="Autoregressive long-video generation with KV cache. LoRA-tuned for fast multi-step.",
        conda_env=str(ROOT / "envs" / "af-longlive"),
        gpu_memory_gb=25.0,
        default_height=480, default_width=832,
        default_frames=30, default_steps=4, fps=16,
        worker_script=str(ROOT / "workers" / "longlive_worker.py"),
    ),
    "longlive_interactive": ModelInfo(
        id="longlive_interactive",
        display_name="LongLive — Interactive",
        description="Interactive chunk-by-chunk generation. Change prompts mid-video with KV cache continuity.",
        conda_env=str(ROOT / "envs" / "af-longlive"),
        gpu_memory_gb=25.0,
        default_height=480, default_width=832,
        default_frames=160, default_steps=4, fps=16,
        worker_script=str(ROOT / "workers" / "longlive_interactive_worker.py"),
    ),
    # ── LTX-Video family ──
    "ltx_2b": ModelInfo(
        id="ltx_2b",
        display_name="LTX-Video — 2B Distilled",
        description="Ultra-fast DiT with multi-scale pipeline. Sub-second per step at 704x480.",
        conda_env=str(ROOT / "envs" / "af-ltx"),
        gpu_memory_gb=26.0,
        default_height=480, default_width=704,
        default_frames=97, default_steps=8, fps=24,
        worker_script=str(ROOT / "workers" / "ltx_worker.py"),
    ),
    "ltx_2b_dev": ModelInfo(
        id="ltx_2b_dev",
        display_name="LTX-Video — 2B (Full)",
        description="Full 2B model with CFG guidance. Higher quality, 40 steps. Single-pass pipeline.",
        conda_env=str(ROOT / "envs" / "af-ltx"),
        gpu_memory_gb=26.0,
        default_height=480, default_width=704,
        default_frames=97, default_steps=40, fps=24,
        worker_script=str(ROOT / "workers" / "ltx_worker.py"),
    ),
    "ltx_13b": ModelInfo(
        id="ltx_13b",
        display_name="LTX-Video — 13B Distilled",
        description="Large 13B DiT model, distilled for fewer steps. Best quality from LTX family.",
        conda_env=str(ROOT / "envs" / "af-ltx"),
        gpu_memory_gb=50.0,
        default_height=480, default_width=704,
        default_frames=97, default_steps=8, fps=24,
        worker_script=str(ROOT / "workers" / "ltx_worker.py"),
    ),
    "ltx_13b_dev": ModelInfo(
        id="ltx_13b_dev",
        display_name="LTX-Video — 13B (Full)",
        description="Full 13B model with complex guidance schedule. Highest quality, 30+ steps per pass.",
        conda_env=str(ROOT / "envs" / "af-ltx"),
        gpu_memory_gb=55.0,
        default_height=480, default_width=704,
        default_frames=97, default_steps=30, fps=24,
        worker_script=str(ROOT / "workers" / "ltx_worker.py"),
    ),
    "hybrid_ltx": ModelInfo(
        id="hybrid_ltx",
        display_name="LTX-Video — Hybrid (13B+2B)",
        description="Hybrid schedule: 13B for structure (70%), 2B for refinement (30%). Best quality-speed tradeoff.",
        conda_env=str(ROOT / "envs" / "af-ltx"),
        gpu_memory_gb=60.0,
        default_height=480, default_width=704,
        default_frames=97, default_steps=10, fps=24,
        worker_script=str(ROOT / "workers" / "hybrid_ltx_worker.py"),
    ),
}

gpu_assignments: dict[int, Optional[str]] = {i: None for i in range(GPU_COUNT)}
_next_port = WORKER_BASE_PORT


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------
def _get_next_port() -> int:
    global _next_port
    port = _next_port
    _next_port += 1
    return port


def _start_worker(model: ModelInfo, gpu_id: int) -> bool:
    """Start a persistent worker subprocess for a model on a specific GPU."""
    python = str(Path(model.conda_env) / "bin" / "python")
    port = _get_next_port()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_dir = OUTPUTS / "_workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model.id}.log"

    cmd = [python, model.worker_script, "--port", str(port), "--model-name", model.id]
    logger.info(f"Starting worker for {model.id} on GPU {gpu_id}, port {port}")

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )

    model.worker_proc = proc
    model.worker_port = port
    model.gpu_id = gpu_id
    model.status = ModelStatus.WARMING

    # Wait for worker to become healthy
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + 180  # 3 min timeout for model loading
    while time.time() < deadline:
        if proc.poll() is not None:
            model.status = ModelStatus.ERROR
            model.error_msg = f"Worker crashed (exit {proc.returncode})"
            try:
                model.error_msg += "\n" + log_path.read_text()[-1000:]
            except Exception:
                pass
            logger.error(f"Worker {model.id} crashed: {model.error_msg}")
            return False
        try:
            r = http_requests.get(url, timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ready":
                model.status = ModelStatus.WARM
                model.last_used = time.time()
                logger.info(f"Worker {model.id} is ready on port {port}")
                return True
        except Exception:
            pass
        time.sleep(2)

    model.status = ModelStatus.ERROR
    model.error_msg = "Worker startup timed out"
    _kill_worker(model)
    return False


def _kill_worker(model: ModelInfo):
    """Kill a worker subprocess and free resources."""
    if model.worker_proc:
        try:
            os.kill(model.worker_proc.pid, signal.SIGTERM)
            model.worker_proc.wait(timeout=10)
        except Exception:
            try:
                os.kill(model.worker_proc.pid, signal.SIGKILL)
            except Exception:
                pass
    model.worker_proc = None
    model.worker_port = None
    model.status = ModelStatus.COLD
    model.error_msg = None
    logger.info(f"Worker {model.id} stopped")


def _assign_gpu(model_id: str, preferred: Optional[int] = None) -> int:
    """Assign a GPU. If model already warm, return its GPU. Otherwise allocate or evict LRU."""
    model = MODELS[model_id]

    # Already running on a GPU
    if model.gpu_id is not None and model.status == ModelStatus.WARM:
        return model.gpu_id

    # Preferred GPU free?
    if preferred is not None and gpu_assignments.get(preferred) is None:
        gpu_assignments[preferred] = model_id
        return preferred

    # Any free GPU?
    for gid in range(GPU_COUNT):
        if gpu_assignments[gid] is None:
            gpu_assignments[gid] = model_id
            return gid

    # Evict LRU
    lru_gid = min(
        range(GPU_COUNT),
        key=lambda g: MODELS[gpu_assignments[g]].last_used if gpu_assignments[g] else 0,
    )
    old_model_id = gpu_assignments[lru_gid]
    if old_model_id:
        old_model = MODELS[old_model_id]
        logger.info(f"Evicting {old_model_id} from GPU {lru_gid}")
        _kill_worker(old_model)
        old_model.gpu_id = None
    gpu_assignments[lru_gid] = model_id
    return lru_gid


def _ensure_worker(model_id: str) -> ModelInfo:
    """Ensure the model's worker is running. Start it if needed."""
    model = MODELS[model_id]

    if model.status == ModelStatus.WARM and model.worker_proc and model.worker_proc.poll() is None:
        model.last_used = time.time()
        return model

    # Need to start the worker
    gpu_id = _assign_gpu(model_id)
    ok = _start_worker(model, gpu_id)
    if not ok:
        raise RuntimeError(f"Failed to start worker for {model_id}: {model.error_msg}")
    return model


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
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


def _run_generation(job: Job):
    model = MODELS[job.model_id]
    job.status = "running"
    job.started_at = time.time()

    job_dir = OUTPUTS / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        worker = _ensure_worker(job.model_id)
        url = f"http://127.0.0.1:{worker.worker_port}/generate"

        payload = {
            "prompt": job.prompt,
            "num_frames": job.num_frames,
            "seed": job.seed,
            "output_dir": str(job_dir),
        }
        if job.model_id in ("ltx_2b", "ltx_13b"):
            payload["height"] = job.height
            payload["width"] = job.width

        logger.info(f"Sending generate request to {url} for job {job.job_id}")
        r = http_requests.post(url, json=payload, timeout=600)
        result = r.json()

        if result.get("status") == "success":
            worker_output = result.get("output_path", "")

            # Move/copy the output to our job dir if it's elsewhere
            if worker_output and Path(worker_output).exists():
                dest = job_dir / "output.mp4"
                if Path(worker_output) != dest:
                    shutil.copy2(worker_output, dest)
                job.output_path = f"/outputs/{job.job_id}/output.mp4"
            else:
                # Search job_dir for any mp4
                mp4s = sorted(job_dir.glob("**/*.mp4"), key=os.path.getmtime)
                if mp4s:
                    job.output_path = f"/outputs/{job.job_id}/{mp4s[-1].name}"

            job.status = "completed"
            job.completed_at = time.time()
            job.progress = 1.0
            worker.last_used = time.time()
        else:
            raise RuntimeError(result.get("error", "Unknown worker error"))

    except Exception as e:
        logger.exception(f"Job {job.job_id} failed")
        job.status = "failed"
        job.error = str(e)[:2000]
        job.completed_at = time.time()
        model.status = ModelStatus.ERROR
        model.error_msg = str(e)[:500]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Arc Fabric", version="0.2.0")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")


class GenerateRequest(BaseModel):
    model_id: str
    prompt: str
    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: Optional[int] = None
    seed: int = 42
    gpu_id: Optional[int] = None


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
    if log_path.exists():
        return {"log": log_path.read_text()[-5000:]}
    # Fall back to worker log
    job = jobs.get(job_id)
    if job:
        worker_log = OUTPUTS / "_workers" / f"{job.model_id}.log"
        if worker_log.exists():
            return {"log": worker_log.read_text()[-5000:]}
    raise HTTPException(404, "Log not found")


@app.on_event("shutdown")
async def shutdown():
    for model in MODELS.values():
        if model.worker_proc:
            _kill_worker(model)


@app.get("/")
async def index():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
