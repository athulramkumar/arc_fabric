# Arc Fabric Architecture

## Overview

Arc Fabric is a stateful execution and control layer that runs generative video
as long-lived sessions, not one-shot inference calls. It treats latents, timesteps,
and noise schedules as first-class, persistent state -- enabling editable video
without full regeneration.

## Hardware

- 2x NVIDIA H100 80GB HBM3 (expandable to 4x or 8x)
- 600GB NVMe storage
- Models are isolated in per-model conda environments under `/workspace/arc_fabric/envs/`

## Current Status (2026-02-18)

### Models Onboarded

| Model | Status | Env | Weights | Test |
|-------|--------|-----|---------|------|
| Wan 2.1 1.3B | WORKING | af-wan21 (envs/af-wan21) | 17GB | 832x480, 33 frames, ~77s |
| LongLive 1.3B | WORKING | af-longlive (envs/af-longlive) | 17GB base + 8.2GB LongLive | 832x480, 30 frames, ~74s |
| LTX-Video 2B | WORKING | af-ltx (envs/af-ltx) | 26GB (2B) + 27GB (13B) | 704x480, 97 frames, ~18s |
| DreamDojo 2B GR-1 | WORKING | af-dreamdojo (envs/af-dreamdojo) | ~34GB VRAM | 640x480, 49 frames, ~42s |
| DreamDojo 14B GR-1 | WORKING | af-dreamdojo (envs/af-dreamdojo) | ~70GB VRAM | 640x480, 49 frames, ~104s |
| Inferix | REFERENCE | -- | -- | -- |

### Platform Components Built

- `app/server.py` - Full-stack UI server (FastAPI + static frontend)
- `app/static/index.html` - fal.ai-style web UI with model selection, GPU status, video generation
- `arc_fabric/config.py` - Model registry, GPU config, platform config
- `arc_fabric/gpu_manager.py` - GPU allocation with LRU eviction (tested)
- `arc_fabric/session_manager.py` - Session lifecycle management
- `arc_fabric/worker_manager.py` - Subprocess lifecycle for model workers
- `arc_fabric/api/server.py` - FastAPI orchestrator (routes requests to workers)
- `workers/longlive_worker.py` - LongLive FastAPI worker
- `workers/ltx_worker.py` - LTX-Video FastAPI worker
- `workers/dreamdojo_worker.py` - DreamDojo FastAPI worker (action-conditioned Video2World)
- `tests/test_gpu_manager.py` - GPU manager unit tests (passing)
- `tests/test_video_quality.py` - CLIP-based quality validation (passing)

## Architecture

```
                     ┌────────────────────────────────────────────────┐
  Browser ────────── │     Arc Fabric UI Server (:8000)               │
  (HTTP)             │                                                │
                     │  ┌──────────────────────────────────────────┐  │
                     │  │  Static Frontend (index.html)            │  │
                     │  │  - Model selection with warm/cold status │  │
                     │  │  - Video parameter controls              │  │
                     │  │  - Live generation timer & log streaming │  │
                     │  │  - Video playback + generation history   │  │
                     │  │  - GPU allocation dashboard              │  │
                     │  └──────────────────────────────────────────┘  │
                     │                                                │
                     │  ┌──────────┐  ┌────────────┐                 │
                     │  │ GPU Mgr  │  │ Job Queue   │                │
                     │  │ (LRU     │  │ (async bg   │                │
                     │  │  evict)  │  │  tasks)     │                │
                     │  └────┬─────┘  └──────┬──────┘                │
                     └───────┼───────────────┼───────────────────────┘
                             │               │
              ┌──────────────┼───────────────┼──────────────┐
              ▼              ▼               ▼              ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │ Wan 2.1 1.3B │ │  LongLive    │ │  LTX-Video   │ │  DreamDojo   │
     │ (af-wan21)   │ │ (af-longlive)│ │  (af-ltx)    │ │(af-dreamdojo)│
     │ GPU: 0 or 1  │ │ GPU: 0 or 1  │ │  GPU: 0 or 1 │ │ GPU: 0 or 1  │
     │              │ │              │ │              │ │              │
     │ generate.py  │ │ pipeline.    │ │  ltx_video.  │ │ vid2world    │
     │ CLI          │ │ inference()  │ │  inference() │ │ inference()  │
     └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

## Key Design Decisions

### Web UI (fal.ai-style)
- Single-page app served by the same FastAPI process
- Model cards show warm/cold status and GPU assignment
- Parameters auto-fill with model-specific defaults
- Live generation timer with log streaming from subprocess stdout
- Video playback with generation history grid

### Model Execution
Each model runs via subprocess using the model's dedicated conda env python binary.
The UI server spawns jobs as background tasks, writing output to a shared filesystem.
No tensor transfer between processes -- latents stay on GPU inside the subprocess.

### GPU Management
- LRU eviction: oldest-accessed model gets evicted when GPUs are full
- Multi-GPU: use model repo's native FSDP/xDiT support
- GPU pool is expandable (just set `ARC_GPU_COUNT` env var)
- Status tracked in-memory and displayed in real-time on the UI

### Environment Portability
All conda environments live under `/workspace/arc_fabric/envs/` for portability.
When moving to a new machine, the envs directory carries all dependencies.

### Automated Quality Testing
- CLIP scoring: compute cosine similarity between video frames and text prompt
- Threshold: 0.20 (both models score 0.31-0.35)
- Catches broken/nonsensical generations automatically

## Model-Specific Notes

### Wan 2.1 (1.3B)
- Full diffusion T2V using generate.py CLI
- Supports hybrid schedule with 14B variant (pending implementation)
- ~8GB GPU memory, ~77s for 33 frames at 832x480

### LongLive
- Frame-level autoregressive with KV cache + frame sink
- LoRA adapter (rank=256) on Wan 1.3B base
- Expects `wan_models/` and `longlive_models/` symlinks in working dir
- ~25GB GPU memory, ~74s for 30 frames at 832x480

### LTX-Video
- DiT-based, distilled for fast inference
- Multi-scale pipeline: first pass (7 steps) + spatial upscaler second pass (3 steps)
- Needs local config with absolute paths (stock config uses relative/HF paths)
- ~26GB GPU memory, ~18s generation for 97 frames at 704x480

### DreamDojo (2B & 14B)
- Action-conditioned Video2World model (NOT text-to-video)
- Takes initial frame + 7D robot actions → predicts future video
- Built on Cosmos Predict2 (Wan 2.1 architecture) with action cross-attention
- 12-frame chunk autoregressive generation, 480×640 resolution
- Worker: `workers/dreamdojo_worker.py`, serves `/health`, `/samples`, `/generate`
- UI: custom sidebar with dataset sample dropdown, synced GT vs predicted video playback, action norms chart, quality metrics (PSNR, SSIM, LPIPS)
- Dataset: nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1 (10 eval tasks, 100 samples)
- 14B variant uses LAM (Latent Action Model) for additional conditioning
- Checkpoints stored in `models/dreamdojo/checkpoints/`
- Detailed setup: `models/dreamdojo/docs/ONBOARDING.md`

### Inferix (reference)
- Block-diffusion inference engine for semi-AR models
- Patterns to adopt: KV cache management, block-level state, streaming decode
- Not run as a separate model; code patterns integrated into our workers

## Storage Layout

```
/workspace/arc_fabric/
├── app/                        # UI Server
│   ├── server.py               # FastAPI backend with generation logic
│   └── static/index.html       # Web frontend
├── models/                     # Git submodules (source code)
│   ├── wan21/                  # Wan 2.1 (with hybrid schedule notebooks)
│   ├── longlive/               # LongLive (interactive video)
│   ├── ltx_video/              # LTX-Video (fast DiT)
│   ├── dreamdojo/              # DreamDojo (action-conditioned Video2World)
│   └── inferix/                # Inferix (reference patterns)
├── weights/                    # ~140GB total
│   ├── wan21/                  # 1.3B (17GB) + 14B (65GB)
│   ├── longlive/               # Wan base (17GB) + LongLive (8.2GB)
│   └── ltx_video/              # 2B (26GB) + 13B (27GB) + shared components
├── outputs/                    # Generated videos
│   └── ui/                     # UI-generated videos (per job-id)
├── envs/                       # Portable conda environments
│   ├── af-wan21/
│   ├── af-longlive/
│   ├── af-ltx/
│   └── af-dreamdojo/
├── workers/                    # Model worker FastAPI servers (standalone)
├── arc_fabric/                 # Core platform code
│   ├── config.py
│   ├── gpu_manager.py
│   ├── session_manager.py
│   ├── worker_manager.py
│   └── api/server.py
├── tests/                      # GPU manager tests + CLIP quality tests
└── docs/ARCHITECTURE.md        # This file
```

## Running

```bash
# Start the UI server
cd /workspace/arc_fabric
python3 app/server.py

# Open http://localhost:8000 in browser
```

## Next Steps

1. Implement Wan hybrid schedule (1.3B high-noise + 14B refinement)
2. Implement LTX hybrid schedule (2B + 13B)
3. Integrate Inferix KV cache patterns into LongLive worker
4. Add session state persistence (resume/fork sessions)
5. Streaming output support (progressive frames via WebSocket)
6. DreamDojo distillation pipeline (real-time 10 FPS generation)
7. DreamDojo multi-embodiment support (G1, AgileBot, EgoDex)
