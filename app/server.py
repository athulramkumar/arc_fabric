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
import threading
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
        description="Interactive chunk-by-chunk generation with AI prompt enhancement and context carryover.",
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
    # ── DreamDojo (Video2World) ──
    "dreamdojo_2b": ModelInfo(
        id="dreamdojo_2b",
        display_name="DreamDojo — 2B GR-1",
        description="Action-conditioned Video2World: predicts future video from initial frame + robot actions. Fast 2B variant.",
        conda_env=str(ROOT / "envs" / "af-dreamdojo"),
        gpu_memory_gb=58.0,
        default_height=480, default_width=640,
        default_frames=49, default_steps=35, fps=10,
        worker_script=str(ROOT / "workers" / "dreamdojo_worker.py"),
    ),
    "dreamdojo_14b": ModelInfo(
        id="dreamdojo_14b",
        display_name="DreamDojo — 14B GR-1",
        description="Action-conditioned Video2World: 14B variant with higher quality predictions. Slower but more accurate.",
        conda_env=str(ROOT / "envs" / "af-dreamdojo"),
        gpu_memory_gb=70.0,
        default_height=480, default_width=640,
        default_frames=49, default_steps=35, fps=10,
        worker_script=str(ROOT / "workers" / "dreamdojo_worker.py"),
    ),
}

# ---------------------------------------------------------------------------
# Model sharing: hybrid workers can serve standalone model requests
# ---------------------------------------------------------------------------
HYBRID_PROVIDERS: dict[str, dict[str, list]] = {
    "hybrid_ltx": {
        "ltx_2b": [["2B", 9999]],
        "ltx_13b": [["13B", 9999]],
    },
    "hybrid_wan21": {
        "wan21_1_3b": [["1.3B", 9999]],
        "wan21_14b": [["14B", 9999]],
    },
}

SHARED_BY: dict[str, str] = {}
for _hybrid_id, _subs in HYBRID_PROVIDERS.items():
    for _sub_id in _subs:
        SHARED_BY[_sub_id] = _hybrid_id

_LTX_MODELS = {"ltx_2b", "ltx_2b_dev", "ltx_13b", "ltx_13b_dev", "hybrid_ltx"}
_DREAMDOJO_MODELS = {"dreamdojo_2b", "dreamdojo_14b"}

gpu_assignments: dict[int, Optional[str]] = {i: None for i in range(GPU_COUNT)}
_next_port = WORKER_BASE_PORT
_gpu_lock = threading.RLock()
_model_start_locks: dict[str, threading.Lock] = {mid: threading.Lock() for mid in MODELS}


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------
def _cleanup_stale_workers():
    """Kill any leftover worker processes from a previous server run."""
    for port in range(WORKER_BASE_PORT, WORKER_BASE_PORT + 100):
        try:
            out = subprocess.check_output(
                ["fuser", f"{port}/tcp"], stderr=subprocess.DEVNULL, text=True
            ).strip()
            if out:
                for pid_str in out.split():
                    try:
                        pid = int(pid_str)
                        logger.info(f"Killing stale process PID {pid} on port {port}")
                        os.kill(pid, signal.SIGKILL)
                    except (ValueError, ProcessLookupError):
                        pass
        except subprocess.CalledProcessError:
            pass
    time.sleep(1)


def _is_port_free(port: int) -> bool:
    """Check if a port is free before attempting to use it."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _get_next_port() -> int:
    global _next_port
    while not _is_port_free(_next_port):
        logger.warning(f"Port {_next_port} is in use, skipping")
        _next_port += 1
    port = _next_port
    _next_port += 1
    return port


def _release_gpu(model: ModelInfo):
    """Release a model's GPU slot from the assignment table."""
    with _gpu_lock:
        if model.gpu_id is not None:
            if gpu_assignments.get(model.gpu_id) == model.id:
                gpu_assignments[model.gpu_id] = None
                logger.info(f"Released GPU {model.gpu_id} from {model.id}")
            model.gpu_id = None


def _kill_worker(model: ModelInfo):
    """Kill a worker subprocess and release all resources including GPU slot."""
    if model.worker_proc:
        try:
            os.kill(model.worker_proc.pid, signal.SIGTERM)
            model.worker_proc.wait(timeout=10)
        except Exception:
            try:
                os.kill(model.worker_proc.pid, signal.SIGKILL)
            except Exception:
                pass
    _release_gpu(model)
    model.worker_proc = None
    model.worker_port = None
    model.status = ModelStatus.COLD
    model.error_msg = None
    logger.info(f"Worker {model.id} stopped, resources released")


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

    # Give the subprocess a moment to crash if port is already taken
    time.sleep(2)

    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            model.status = ModelStatus.ERROR
            model.error_msg = f"Worker crashed (exit {proc.returncode})"
            try:
                model.error_msg += "\n" + log_path.read_text()[-1000:]
            except Exception:
                pass
            logger.error(f"Worker {model.id} crashed: {model.error_msg}")
            _release_gpu(model)
            return False
        try:
            r = http_requests.get(url, timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ready":
                # Verify our subprocess is still alive — guards against a stale
                # process on the same port answering the health check.
                time.sleep(0.5)
                if proc.poll() is not None:
                    logger.warning(
                        f"Worker {model.id} health OK but subprocess died "
                        f"(stale process on port {port}?)"
                    )
                    continue
                model.status = ModelStatus.WARM
                model.last_used = time.time()
                logger.info(f"Worker {model.id} is ready on GPU {gpu_id}, port {port}")
                return True
        except Exception:
            pass
        time.sleep(2)

    model.status = ModelStatus.ERROR
    model.error_msg = "Worker startup timed out"
    _kill_worker(model)
    return False


def _is_worker_alive(model: ModelInfo) -> bool:
    """Check if the worker process is running. Auto-recovers status if needed."""
    if model.worker_proc is None:
        return False
    if model.worker_proc.poll() is not None:
        logger.warning(f"Worker {model.id} process died (was on GPU {model.gpu_id})")
        _release_gpu(model)
        model.status = ModelStatus.COLD
        model.worker_proc = None
        model.worker_port = None
        return False
    if model.status == ModelStatus.ERROR:
        logger.info(f"Worker {model.id} process alive — recovering to WARM")
        model.status = ModelStatus.WARM
        model.error_msg = None
    return model.status == ModelStatus.WARM


def _assign_gpu(model_id: str, preferred: Optional[int] = None) -> int:
    """Thread-safe GPU assignment with LRU eviction."""
    with _gpu_lock:
        model = MODELS[model_id]

        # Already assigned and alive (WARM or still WARMING)
        if model.gpu_id is not None and model.status in (ModelStatus.WARM, ModelStatus.WARMING):
            return model.gpu_id

        # Clean up stale assignment if model is COLD/ERROR but still holds a GPU slot
        if model.gpu_id is not None:
            if gpu_assignments.get(model.gpu_id) == model_id:
                gpu_assignments[model.gpu_id] = None
            model.gpu_id = None

        if preferred is not None and gpu_assignments.get(preferred) is None:
            gpu_assignments[preferred] = model_id
            return preferred

        for gid in range(GPU_COUNT):
            if gpu_assignments[gid] is None:
                gpu_assignments[gid] = model_id
                return gid

        # Evict LRU — pick the GPU with the oldest-used model
        lru_gid = min(
            range(GPU_COUNT),
            key=lambda g: MODELS[gpu_assignments[g]].last_used if gpu_assignments[g] else 0,
        )
        old_model_id = gpu_assignments[lru_gid]
        if old_model_id:
            old_model = MODELS[old_model_id]
            logger.info(f"Evicting {old_model_id} from GPU {lru_gid}")
            _kill_worker(old_model)
        gpu_assignments[lru_gid] = model_id
        return lru_gid


def _find_shared_worker(model_id: str) -> Optional[ModelInfo]:
    """If a warm hybrid worker can serve this standalone model, return it."""
    hybrid_id = SHARED_BY.get(model_id)
    if not hybrid_id:
        return None
    hybrid = MODELS[hybrid_id]
    if _is_worker_alive(hybrid):
        logger.info(
            f"{model_id} → reusing hybrid {hybrid_id} "
            f"(GPU {hybrid.gpu_id}, port {hybrid.worker_port})"
        )
        return hybrid
    return None


def _find_any_serving_worker(model_id: str) -> Optional[tuple[ModelInfo, Optional[list]]]:
    """Find any live worker that can serve this model.

    Returns (worker, schedule_override) or None.
    """
    model = MODELS[model_id]

    if _is_worker_alive(model):
        model.last_used = time.time()
        return model, None

    shared = _find_shared_worker(model_id)
    if shared:
        hybrid_id = shared.id
        schedule_override = HYBRID_PROVIDERS[hybrid_id][model_id]
        return shared, schedule_override

    return None


def _ensure_worker(model_id: str) -> ModelInfo:
    """Ensure a worker is running. Uses per-model lock to prevent duplicate starts."""
    model = MODELS[model_id]

    if _is_worker_alive(model):
        model.last_used = time.time()
        return model

    # Per-model lock: if another thread is already starting this model, wait for it
    with _model_start_locks[model_id]:
        # Re-check after acquiring lock — another thread may have finished starting it
        if _is_worker_alive(model):
            model.last_used = time.time()
            return model

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
    enable_caching: bool = False
    cache_start_step: Optional[int] = None
    cache_end_step: Optional[int] = None
    cache_interval: int = 3
    gpu_id: Optional[int] = None
    status: str = "queued"
    progress: float = 0.0
    output_path: Optional[str] = None
    error: Optional[str] = None
    served_by: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    sample_id: Optional[str] = None
    gt_path: Optional[str] = None
    merged_path: Optional[str] = None
    actions_path: Optional[str] = None
    metrics: Optional[dict] = None

    @property
    def elapsed(self) -> Optional[float]:
        if self.started_at:
            end = self.completed_at or time.time()
            return round(end - self.started_at, 1)
        return None


jobs: dict[str, Job] = {}


def _build_payload(job: Job, worker: ModelInfo, schedule_override: Optional[list] = None) -> dict:
    """Build the request payload for a worker, handling model sharing and caching."""
    is_dreamdojo = job.model_id in _DREAMDOJO_MODELS

    if is_dreamdojo:
        return {
            "sample_id": job.sample_id,
            "output_dir": str(OUTPUTS / job.job_id),
            "num_frames": job.num_frames,
            "seed": job.seed,
            "prompt": job.prompt or "",
        }

    payload: dict = {
        "prompt": job.prompt,
        "num_frames": job.num_frames,
        "seed": job.seed,
        "output_dir": str(OUTPUTS / job.job_id),
    }

    is_ltx = job.model_id in _LTX_MODELS or worker.id in _LTX_MODELS
    is_hybrid = worker.id.startswith("hybrid_")

    if is_ltx:
        payload["height"] = job.height
        payload["width"] = job.width

    if schedule_override:
        payload["schedule"] = schedule_override

    if is_hybrid:
        if job.enable_caching:
            payload["cache_start_step"] = job.cache_start_step
            payload["cache_end_step"] = job.cache_end_step
            payload["cache_interval"] = job.cache_interval
            if worker.id.startswith("hybrid_wan"):
                payload["enable_caching"] = True
        else:
            if worker.id.startswith("hybrid_ltx"):
                payload["cache_start_step"] = None
                payload["cache_end_step"] = None
    elif job.enable_caching:
        payload["cache_start_step"] = job.cache_start_step
        payload["cache_end_step"] = job.cache_end_step
        payload["cache_interval"] = job.cache_interval

    return payload


def _run_generation(job: Job):
    model = MODELS[job.model_id]
    job.status = "running"
    job.started_at = time.time()

    job_dir = OUTPUTS / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = _find_any_serving_worker(job.model_id)
        if result:
            worker, schedule_override = result
            job.served_by = worker.id
            if schedule_override:
                logger.info(
                    f"Job {job.job_id}: {job.model_id} → reusing {worker.id} "
                    f"(no reload, schedule: {schedule_override})"
                )
            else:
                logger.info(f"Job {job.job_id}: {job.model_id} already warm on GPU {worker.gpu_id}")
            worker.last_used = time.time()
        else:
            logger.info(f"Job {job.job_id}: starting fresh worker for {job.model_id}")
            worker = _ensure_worker(job.model_id)
            schedule_override = None
            job.served_by = worker.id

        url = f"http://127.0.0.1:{worker.worker_port}/generate"
        payload = _build_payload(job, worker, schedule_override)

        logger.info(f"Job {job.job_id}: POST {url}")
        r = http_requests.post(url, json=payload, timeout=1800)
        result = r.json()

        if result.get("status") == "success":
            worker_output = result.get("output_path", "")

            if job.model_id in _DREAMDOJO_MODELS:
                for key in ("gt_path", "merged_path", "actions_path"):
                    val = result.get(key, "")
                    if val and Path(val).exists():
                        setattr(job, key, f"/outputs/{job.job_id}/{Path(val).name}")
                job.metrics = result.get("metrics")
                pred_p = result.get("pred_path", "")
                if pred_p and Path(pred_p).exists():
                    job.output_path = f"/outputs/{job.job_id}/{Path(pred_p).name}"
            elif worker_output and Path(worker_output).exists():
                dest = job_dir / "output.mp4"
                if Path(worker_output) != dest:
                    shutil.copy2(worker_output, dest)
                job.output_path = f"/outputs/{job.job_id}/output.mp4"
            else:
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

        serving_model = MODELS.get(job.served_by) if job.served_by else model
        if serving_model and serving_model.worker_proc:
            if serving_model.worker_proc.poll() is not None:
                logger.error(f"Worker {serving_model.id} process died")
                _release_gpu(serving_model)
                serving_model.status = ModelStatus.ERROR
                serving_model.error_msg = str(e)[:500]
            else:
                logger.info(f"Worker {serving_model.id} still alive after job failure")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Arc Fabric", version="0.4.0")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")


class GenerateRequest(BaseModel):
    model_id: str
    prompt: str = ""
    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: Optional[int] = None
    seed: int = 42
    gpu_id: Optional[int] = None
    enable_caching: bool = False
    cache_start_step: Optional[int] = None
    cache_end_step: Optional[int] = None
    cache_interval: int = 3
    sample_id: Optional[str] = None


def _model_effective_status(m: ModelInfo) -> tuple[str, Optional[str]]:
    """Return (effective_status, shared_via) for a model."""
    if _is_worker_alive(m):
        return "warm", None

    hybrid_id = SHARED_BY.get(m.id)
    if hybrid_id:
        hybrid = MODELS[hybrid_id]
        if _is_worker_alive(hybrid):
            return "warm", hybrid_id

    return m.status.value, None


@app.get("/api/models")
async def api_models():
    result = []
    for m in MODELS.values():
        effective_status, shared_via = _model_effective_status(m)
        effective_gpu = m.gpu_id
        if shared_via:
            effective_gpu = MODELS[shared_via].gpu_id
        result.append({
            "id": m.id,
            "display_name": m.display_name,
            "description": m.description,
            "status": effective_status,
            "shared_via": shared_via,
            "gpu_id": effective_gpu,
            "gpu_memory_gb": m.gpu_memory_gb,
            "is_hybrid": m.id.startswith("hybrid_"),
            "defaults": {
                "height": m.default_height,
                "width": m.default_width,
                "num_frames": m.default_frames,
                "steps": m.default_steps,
                "fps": m.fps,
            },
        })
    return result


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
        prompt=req.prompt or "",
        height=req.height or model.default_height,
        width=req.width or model.default_width,
        num_frames=req.num_frames or model.default_frames,
        seed=req.seed,
        enable_caching=req.enable_caching,
        cache_start_step=req.cache_start_step,
        cache_end_step=req.cache_end_step,
        cache_interval=req.cache_interval,
        gpu_id=req.gpu_id,
        sample_id=req.sample_id,
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
        "served_by": job.served_by,
        "enable_caching": job.enable_caching,
        "gt_path": job.gt_path,
        "merged_path": job.merged_path,
        "actions_path": job.actions_path,
        "metrics": job.metrics,
        "sample_id": job.sample_id,
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
            "served_by": j.served_by,
        }
        for j in sorted(jobs.values(), key=lambda x: x.created_at, reverse=True)
    ]


@app.get("/api/logs/{job_id}")
async def api_logs(job_id: str):
    log_path = OUTPUTS / job_id / "log.txt"
    if log_path.exists():
        return {"log": log_path.read_text()[-5000:]}
    job = jobs.get(job_id)
    if job:
        for worker_id in [job.served_by, job.model_id]:
            if worker_id:
                worker_log = OUTPUTS / "_workers" / f"{worker_id}.log"
                if worker_log.exists():
                    return {"log": worker_log.read_text()[-5000:]}
    raise HTTPException(404, "Log not found")


# ---------------------------------------------------------------------------
# DreamDojo — fetch available dataset samples from the running worker
# ---------------------------------------------------------------------------

@app.get("/api/dreamdojo/samples")
async def dreamdojo_samples():
    """Get available dataset samples from a running DreamDojo worker."""
    for mid in _DREAMDOJO_MODELS:
        m = MODELS[mid]
        if _is_worker_alive(m):
            try:
                r = http_requests.get(f"http://127.0.0.1:{m.worker_port}/samples", timeout=10)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
    return {"samples": [], "message": "No DreamDojo worker running. Start a generation to load samples."}


# ---------------------------------------------------------------------------
# Interactive chunk-by-chunk video generation (proxied to longlive_interactive)
# ---------------------------------------------------------------------------

class InteractiveSetupRequest(BaseModel):
    seed: int = 42
    chunk_duration: float = 10.0
    max_chunks: int = 12


class InteractiveGroundingRequest(BaseModel):
    grounding: str
    skip_ai: bool = False


class InteractiveAcceptGroundingRequest(BaseModel):
    enhanced: str


class InteractiveRegenerateGroundingRequest(BaseModel):
    grounding: str


class InteractiveChunkRequest(BaseModel):
    user_prompt: str
    processed_prompt: Optional[str] = None
    skip_ai: bool = False


class InteractiveEnhanceChunkRequest(BaseModel):
    user_prompt: str


_interactive_session: dict = {"session_id": None, "session_dir": None}


def _interactive_worker_url() -> str:
    model = MODELS["longlive_interactive"]
    if not _is_worker_alive(model):
        raise HTTPException(503, "Interactive worker not available. Call setup first.")
    return f"http://127.0.0.1:{model.worker_port}"


def _path_to_url(abs_path: str) -> Optional[str]:
    """Convert an absolute file path under OUTPUTS to a /outputs/... URL."""
    if not abs_path:
        return None
    try:
        return "/outputs/" + str(Path(abs_path).relative_to(OUTPUTS))
    except ValueError:
        return None


@app.post("/api/interactive/setup")
async def api_interactive_setup(req: InteractiveSetupRequest = InteractiveSetupRequest()):
    worker = _ensure_worker("longlive_interactive")
    output_dir = str(OUTPUTS / "interactive")
    try:
        r = http_requests.post(
            f"http://127.0.0.1:{worker.worker_port}/setup",
            json={
                "output_dir": output_dir,
                "seed": req.seed,
                "chunk_duration": req.chunk_duration,
                "max_chunks": req.max_chunks,
            },
            timeout=300,
        )
        r.raise_for_status()
    except http_requests.exceptions.HTTPError:
        raise HTTPException(502, f"Worker setup failed: {r.text[:500]}")
    except Exception as e:
        raise HTTPException(502, f"Worker setup failed: {e}")
    result = r.json()
    _interactive_session["session_id"] = result.get("session_id")
    _interactive_session["session_dir"] = result.get("session_dir")
    worker.last_used = time.time()
    return result


@app.post("/api/interactive/grounding")
async def api_interactive_grounding(req: InteractiveGroundingRequest):
    url = _interactive_worker_url()
    try:
        r = http_requests.post(f"{url}/grounding", json=req.dict(), timeout=60)
        r.raise_for_status()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Grounding failed: {e}")
    MODELS["longlive_interactive"].last_used = time.time()
    return r.json()


@app.post("/api/interactive/accept_grounding")
async def api_interactive_accept_grounding(req: InteractiveAcceptGroundingRequest):
    url = _interactive_worker_url()
    try:
        r = http_requests.post(f"{url}/accept_grounding", json=req.dict(), timeout=30)
        r.raise_for_status()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Accept grounding failed: {e}")
    MODELS["longlive_interactive"].last_used = time.time()
    return r.json()


@app.post("/api/interactive/regenerate_grounding")
async def api_interactive_regenerate_grounding(req: InteractiveRegenerateGroundingRequest):
    url = _interactive_worker_url()
    try:
        r = http_requests.post(f"{url}/regenerate_grounding", json=req.dict(), timeout=60)
        r.raise_for_status()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Regenerate grounding failed: {e}")
    MODELS["longlive_interactive"].last_used = time.time()
    return r.json()


@app.post("/api/interactive/enhance_chunk")
async def api_interactive_enhance_chunk(req: InteractiveEnhanceChunkRequest):
    url = _interactive_worker_url()
    try:
        r = http_requests.post(f"{url}/enhance_chunk", json=req.dict(), timeout=60)
        r.raise_for_status()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Enhance chunk failed: {e}")
    MODELS["longlive_interactive"].last_used = time.time()
    return r.json()


@app.post("/api/interactive/regenerate_chunk_prompt")
async def api_interactive_regenerate_chunk_prompt(req: InteractiveEnhanceChunkRequest):
    url = _interactive_worker_url()
    try:
        r = http_requests.post(f"{url}/regenerate_chunk_prompt", json=req.dict(), timeout=60)
        r.raise_for_status()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Regenerate chunk prompt failed: {e}")
    MODELS["longlive_interactive"].last_used = time.time()
    return r.json()


@app.post("/api/interactive/generate_chunk")
async def api_interactive_generate_chunk(req: InteractiveChunkRequest):
    url = _interactive_worker_url()
    try:
        r = http_requests.post(f"{url}/generate_chunk", json=req.dict(), timeout=600)
        r.raise_for_status()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Chunk generation failed: {e}")
    result = r.json()
    for key in ("chunk_video", "running_video"):
        if key in result:
            result[f"{key}_url"] = _path_to_url(result[key])
    MODELS["longlive_interactive"].last_used = time.time()
    return result


@app.post("/api/interactive/go_back")
async def api_interactive_go_back():
    url = _interactive_worker_url()
    try:
        r = http_requests.post(f"{url}/go_back", timeout=30)
        r.raise_for_status()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Go-back failed: {e}")
    MODELS["longlive_interactive"].last_used = time.time()
    return r.json()


@app.post("/api/interactive/finalize")
async def api_interactive_finalize():
    url = _interactive_worker_url()
    try:
        r = http_requests.post(f"{url}/finalize", timeout=120)
        r.raise_for_status()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Finalize failed: {e}")
    result = r.json()
    if "final_video" in result:
        result["final_video_url"] = _path_to_url(result["final_video"])
    MODELS["longlive_interactive"].last_used = time.time()
    return result


@app.post("/api/interactive/reset")
async def api_interactive_reset():
    url = _interactive_worker_url()
    try:
        r = http_requests.post(f"{url}/reset", timeout=30)
        r.raise_for_status()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Reset failed: {e}")
    result = r.json()
    _interactive_session["session_id"] = result.get("new_session_id")
    _interactive_session["session_dir"] = result.get("session_dir")
    MODELS["longlive_interactive"].last_used = time.time()
    return result


@app.get("/api/interactive/status")
async def api_interactive_status():
    try:
        url = _interactive_worker_url()
    except HTTPException:
        return {"is_setup": False, "model_loaded": False}
    try:
        r = http_requests.get(f"{url}/status", timeout=10)
        if r.status_code == 200:
            result = r.json()
            sd = result.get("session_dir")
            cc = result.get("current_chunk", 0)
            if sd and cc > 0:
                result["last_chunk_url"] = _path_to_url(
                    str(Path(sd) / f"chunk_{cc}.mp4")
                )
                result["running_video_url"] = _path_to_url(
                    str(Path(sd) / f"running_{cc}.mp4")
                )
            return result
    except Exception:
        pass
    return {"is_setup": False, "model_loaded": False}


@app.get("/api/interactive/logs")
async def api_interactive_logs():
    log_path = OUTPUTS / "_workers" / "longlive_interactive.log"
    if log_path.exists():
        return {"log": log_path.read_text()[-5000:]}
    return {"log": ""}


@app.on_event("shutdown")
async def shutdown():
    for model in MODELS.values():
        if model.worker_proc:
            _kill_worker(model)


@app.get("/")
async def index():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


@app.on_event("startup")
async def startup():
    _cleanup_stale_workers()
    logger.info(f"Arc Fabric server starting with {GPU_COUNT} GPUs, "
                f"worker ports from {WORKER_BASE_PORT}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
