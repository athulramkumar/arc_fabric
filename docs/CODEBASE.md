# Arc Fabric — Codebase Reference

## Directory Structure

```
/workspace/arc_fabric/
├── app/                           # UI Server (user-facing)
│   ├── server.py                  #   FastAPI backend + generation logic
│   └── static/index.html          #   fal.ai-style frontend
├── arc_fabric/                    # Core platform library
│   ├── __init__.py                #   Package init (__version__ = "0.1.0")
│   ├── config.py                  #   Model registry, GPU config, paths
│   ├── gpu_manager.py             #   GPU allocation with LRU eviction
│   ├── session_manager.py         #   Session lifecycle management
│   ├── worker_manager.py          #   Worker subprocess lifecycle
│   └── api/server.py              #   Orchestrator API (routes to workers)
├── workers/                       # Standalone model workers
│   ├── longlive_worker.py         #   LongLive FastAPI worker
│   └── ltx_worker.py              #   LTX-Video FastAPI worker
├── tests/                         # Test suite
│   ├── test_gpu_manager.py        #   GPU manager unit tests
│   ├── test_video_quality.py      #   CLIP-based quality validation
│   └── test_longlive_smoke.py     #   LongLive smoke tests
├── models/                        # Git submodules (model source code)
│   ├── wan21/                     #   Wan 2.1 (T2V, hybrid schedule)
│   ├── longlive/                  #   LongLive (AR long video)
│   ├── ltx_video/                 #   LTX-Video (fast DiT)
│   └── inferix/                   #   Inferix (reference patterns)
├── weights/                       # Model weights (~140GB)
│   ├── wan21/                     #   Wan2.1-T2V-1.3B, Wan2.1-T2V-14B
│   ├── longlive/                  #   Wan base + LongLive LoRA
│   └── ltx_video/                 #   ltxv-2b, ltxv-13b distilled
├── envs/                          # Portable conda environments
│   ├── af-wan21/
│   ├── af-longlive/
│   └── af-ltx/
├── outputs/                       # Generated videos
│   └── ui/                        #   Videos generated via the UI (per job-id)
├── requirements.txt               # Python deps for orchestrator
└── docs/
    ├── ARCHITECTURE.md            # Architecture + design decisions
    └── CODEBASE.md                # This file
```

---

## `app/server.py` — UI Server

The primary user-facing server. Serves the web frontend and handles video generation requests by spawning model subprocesses.

### Classes

| Class | Purpose |
|-------|---------|
| `ModelStatus(Enum)` | Model state: `cold`, `warming`, `warm`, `error` |
| `ModelInfo(dataclass)` | Model metadata: id, name, description, env path, GPU memory, defaults, current status |
| `Job(dataclass)` | Generation job: id, model, prompt, dimensions, status, output path, timing |
| `GenerateRequest(BaseModel)` | Pydantic request: `model_id`, `prompt`, `height`, `width`, `num_frames`, `seed`, `gpu_id` |

### Global State

```python
MODELS: dict[str, ModelInfo]           # 3 registered models (wan21_1_3b, longlive, ltx_2b)
gpu_assignments: dict[int, str|None]   # GPU -> model mapping (LRU eviction)
jobs: dict[str, Job]                   # Active/completed jobs
```

### Key Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `_assign_gpu` | `(model_id: str, preferred: int?) -> int` | Allocate a GPU. If all occupied, evict LRU model. |
| `_run_generation` | `(job: Job) -> None` | Background task: assign GPU, dispatch to model-specific runner, update job status. |
| `_run_wan21` | `(job, model, job_dir, env)` | Execute `models/wan21/generate.py` CLI via subprocess. |
| `_run_longlive` | `(job, model, job_dir, env)` | Generate inline Python script, execute via af-longlive env. Uses `pipeline.inference()`. |
| `_run_ltx` | `(job, model, job_dir, env)` | Write dynamic YAML config, generate script, execute via af-ltx env. Uses `ltx_video.inference.infer()`. |
| `_exec` | `(cmd, env, job_dir, cwd)` | Run subprocess, capture stdout to `log.txt`, raise on non-zero exit. |

### API Endpoints

| Method | Path | Request | Response | Description |
|--------|------|---------|----------|-------------|
| `GET` | `/` | — | HTML | Serve static frontend |
| `GET` | `/api/models` | — | `[{id, display_name, description, status, gpu_id, gpu_memory_gb, defaults}]` | List models with warm/cold status |
| `GET` | `/api/gpus` | — | `[{gpu_id, model_id, model_name, status}]` | GPU allocation status |
| `POST` | `/api/generate` | `{model_id, prompt, height?, width?, num_frames?, seed, gpu_id?}` | `{job_id, status}` | Queue a generation job |
| `GET` | `/api/jobs/{job_id}` | — | `{job_id, model_id, prompt, status, progress, output_path, error, elapsed}` | Job details |
| `GET` | `/api/jobs` | — | `[{job_id, model_id, prompt, status, output_path, elapsed}]` | List all jobs (newest first) |
| `GET` | `/api/logs/{job_id}` | — | `{log: string}` | Last 5000 chars of subprocess log |

### Model Execution Flow

```
POST /api/generate
  → Job created (status: queued)
  → BackgroundTasks.add_task(_run_generation)
    → _assign_gpu() — allocate or evict LRU
    → model.status = WARMING
    → _run_wan21() / _run_longlive() / _run_ltx()
      → Write run.py script (or use CLI)
      → subprocess.Popen(env_python, run.py, CUDA_VISIBLE_DEVICES=gpu_id)
      → Wait for completion
      → Find output .mp4
    → job.status = completed / failed
    → model.status = WARM / ERROR
```

---

## `arc_fabric/config.py` — Model Registry

Defines the canonical model specifications used by the orchestrator and worker manager.

### Key Types

```python
@dataclass
class ModelSpec:
    name: str                    # e.g. "longlive"
    conda_env: str               # Path to conda env, e.g. "/workspace/arc_fabric/envs/af-longlive"
    worker_module: str           # e.g. "workers.longlive_worker"
    gpu_memory_gb: float         # Memory footprint
    weight_paths: dict[str, str] # Named weight directory paths
    working_dir: str             # CWD for the worker
    multi_gpu: bool = False      # Whether to use multi-GPU

@dataclass
class GPUInfo:
    index: int
    total_memory_gb: float = 80.0

@dataclass
class PlatformConfig:
    gpus: list[GPUInfo]
    worker_base_port: int = 9100
    orchestrator_port: int = 8000
    outputs_dir: str
```

### Constants

```python
ROOT_DIR = Path("/workspace/arc_fabric")
WEIGHTS_DIR = ROOT_DIR / "weights"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
ENVS_DIR = ROOT_DIR / "envs"

MODEL_REGISTRY = {
    "longlive":   ModelSpec(conda_env=ENVS_DIR/"af-longlive", gpu_memory_gb=25.0, ...),
    "ltx_2b":     ModelSpec(conda_env=ENVS_DIR/"af-ltx",      gpu_memory_gb=26.0, ...),
    "ltx_13b":    ModelSpec(conda_env=ENVS_DIR/"af-ltx",      gpu_memory_gb=50.0, ...),
    "wan21_1_3b": ModelSpec(conda_env=ENVS_DIR/"af-wan21",    gpu_memory_gb=8.0,  ...),
    "wan21_14b":  ModelSpec(conda_env=ENVS_DIR/"af-wan21",    gpu_memory_gb=35.0, ...),
}
```

---

## `arc_fabric/gpu_manager.py` — GPU Allocation

Manages GPU resources with automatic LRU eviction when all GPUs are occupied.

### Class: `GPUManager`

```python
class GPUManager:
    def __init__(self, gpus: list[GPUInfo])

    def allocate(self, model_spec: ModelSpec, worker_port: int) -> GPUAllocation
    def release(self, model_name: str) -> Optional[GPUAllocation]
    def can_allocate(self, model_spec: ModelSpec) -> bool
    def get_lru_model(self) -> Optional[str]
    def evict_lru(self) -> Optional[GPUAllocation]
    def touch(self, model_name: str)              # Update last_used timestamp
    def status(self) -> dict                       # Full status report
```

### `GPUAllocation` dataclass

```python
@dataclass
class GPUAllocation:
    model_name: str
    gpu_indices: list[int]
    worker_port: int
    worker_pid: Optional[int]
    allocated_at: float
    last_used: float
```

---

## `arc_fabric/session_manager.py` — Session Lifecycle

Manages active video generation sessions.

### Class: `SessionManager`

```python
class SessionManager:
    def create_session(self, model_name: str, config: Optional[dict] = None) -> Session
    def get_session(self, session_id: str) -> Optional[Session]
    def end_session(self, session_id: str) -> Optional[Session]
    def list_sessions(self) -> list[dict]
    def get_sessions_for_model(self, model_name: str) -> list[Session]
```

### `Session` dataclass

```python
@dataclass
class Session:
    session_id: str
    model_name: str
    created_at: float
    last_active: float
    status: str                  # "active" / "ended"
    generation_config: dict
    output_files: list[str]
```

---

## `arc_fabric/worker_manager.py` — Worker Subprocess Management

Spawns and manages model worker processes (each in its own conda env).

### Class: `WorkerManager`

```python
class WorkerManager:
    def __init__(self, gpu_manager: GPUManager, config: PlatformConfig)

    def start_worker(self, model_name: str) -> WorkerProcess
    def wait_for_ready(self, model_name: str, timeout: float = 120) -> bool
    def stop_worker(self, model_name: str)
    def stop_all(self)
    def get_worker_url(self, model_name: str) -> Optional[str]
    def status(self) -> dict
```

Key implementation detail: `_resolve_conda_run` always uses `conda run --prefix <env_path>` for portability.

---

## `arc_fabric/api/server.py` — Orchestrator API

A higher-level FastAPI app that routes requests to model workers via HTTP.

### Factory

```python
def create_app(config: Optional[PlatformConfig] = None) -> FastAPI
```

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET /` | Root info (name, version, models) |
| `GET /models` | Model registry with memory requirements |
| `GET /gpus` | GPU allocation status |
| `GET /sessions` | List active sessions |
| `POST /sessions` | Create session (starts worker if needed) |
| `POST /generate` | Generate video (proxied to worker) |
| `DELETE /sessions/{session_id}` | End session |
| `GET /workers` | Worker subprocess status |

---

## `workers/longlive_worker.py` — LongLive Worker

Standalone FastAPI server for LongLive model inference. Loaded in the `af-longlive` conda env.

### Key Functions

| Function | Purpose |
|----------|---------|
| `setup_symlinks()` | Creates `wan_models/Wan2.1-T2V-1.3B` and `longlive_models` symlinks expected by LongLive |
| `load_model()` | Loads CausalInferencePipeline, generator checkpoint, LoRA adapter (rank=256) |

### Model Loading Sequence

```python
pipeline = CausalInferencePipeline(config, device=device)
pipeline.generator.load_state_dict(checkpoint)
pipeline.generator.model = configure_lora_for_model(
    pipeline.generator.model, model_name="generator", lora_config=config.adapter
)
peft.set_peft_model_state_dict(pipeline.generator.model, lora_weights)
```

### Endpoints

| Method | Path | Request | Response |
|--------|------|---------|----------|
| `GET` | `/health` | — | `{status, model, gpu, gpu_memory_allocated_gb}` |
| `POST` | `/generate` | `{prompt, num_frames, seed, session_id?, output_dir?}` | `{status, output_path, num_frames, generation_time_s, error?}` |
| `GET` | `/status` | — | `{model, loaded, gpu_memory_gb}` |

---

## `workers/ltx_worker.py` — LTX-Video Worker

Standalone FastAPI server for LTX-Video inference. Loaded in the `af-ltx` conda env.

### Key Functions

| Function | Purpose |
|----------|---------|
| `create_local_config(variant)` | Generates YAML config with absolute paths to local weights. Detects multi-scale vs single pipeline. |
| `load_model(variant)` | Imports `ltx_video.inference` and prepares config path. |

### Endpoints

Same shape as LongLive worker but with additional `height`/`width` fields in the generate request.

---

## `tests/test_gpu_manager.py` — GPU Manager Tests

```python
def test_basic_allocation()    # Allocate one model to one GPU
def test_lru_eviction()        # Fill GPUs, verify LRU model gets evicted
def test_multi_gpu()           # Allocate model needing >1 GPU
def test_status()              # Verify status report format
```

## `tests/test_video_quality.py` — CLIP Quality Tests

```python
def extract_frames(video_path: str, num_frames: int = 8) -> list[PIL.Image]
def compute_clip_score(video_path: str, prompt: str, num_frames: int = 8) -> dict
    # Returns: {mean_score, max_score, min_score, per_frame_scores}
def test_video(video_path: str, prompt: str, threshold: float = 0.20) -> bool
```

Uses `open_clip` ViT-B-32 model to compute cosine similarity between video frames and text prompt.

---

## `app/static/index.html` — Web Frontend

Single-page app with embedded CSS and JavaScript. No build step required.

### Layout

```
┌─────────────────────────────────────────────────┐
│  Arc Fabric                    [GPU 0] [GPU 1]  │  ← Header
├─────────────┬───────────────────────────────────┤
│  MODEL      │                                   │
│  ┌────────┐ │      ┌───────────────────┐        │
│  │Wan 2.1 │ │      │                   │        │
│  │ WARM   │ │      │   Video Player    │        │
│  └────────┘ │      │                   │        │
│  ┌────────┐ │      └───────────────────┘        │
│  │LongLive│ │  Model: Wan 2.1  Time: 77s       │
│  │ COLD   │ │                                   │
│  └────────┘ │  ┌─ GENERATION LOG ─────────────┐ │
│  ┌────────┐ │  │ [2026-02-18] INFO: Loading...│ │
│  │LTX 2B  │ │  └─────────────────────────────┘ │
│  │ COLD   │ │                                   │
│  └────────┘ │  RECENT GENERATIONS               │
│             │  ┌──────┐ ┌──────┐ ┌──────┐       │
│  PARAMETERS │  │ vid1 │ │ vid2 │ │ vid3 │       │
│  Prompt: ...│  └──────┘ └──────┘ └──────┘       │
│  W: 832     │                                   │
│  H: 480     │                                   │
│  Frames: 33 │                                   │
│  Seed: 42   │                                   │
│             │                                   │
│ [Generate]  │                                   │
└─────────────┴───────────────────────────────────┘
```

### JavaScript Functions

| Function | Purpose |
|----------|---------|
| `loadModels()` | Fetch `GET /api/models`, render model cards with status badges |
| `selectModel(id)` | Set selected model, populate parameter defaults |
| `loadGPUs()` | Fetch `GET /api/gpus`, render GPU chips in header |
| `generate()` | `POST /api/generate`, start loading overlay, begin polling |
| `pollJob()` | Poll `GET /api/jobs/{id}` every 2s, update timer, fetch logs, detect completion |
| `showOverlay(text)` | Show spinner + timer + status text overlay on video box |
| `showVideo(job)` | Set video `src`, show metadata (model, time, seed) |
| `showError(error)` | Display error message in overlay |
| `loadHistory()` | Fetch `GET /api/jobs`, render completed jobs as thumbnail grid |
| `replayVideo(...)` | Click history thumbnail to replay in main player |

### Auto-refresh

- Models + GPUs polled every 8 seconds
- Active job polled every 2 seconds
- Timer updated every 100ms during generation
