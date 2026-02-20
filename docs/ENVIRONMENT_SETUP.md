# Arc Fabric — Environment Setup Guide

This guide walks through setting up the three model environments, downloading
weights, and verifying the installation. All environments live under
`envs/` so they can be carried across machines with the workspace.

## Prerequisites

| Requirement | Minimum |
|-------------|---------|
| GPU | 2× NVIDIA H100 80 GB (works on A100 as well) |
| CUDA | 12.8+ (system CUDA 13.0 is also compatible) |
| conda | 25.x (Miniforge / Mambaforge recommended) |
| Storage | ~200 GB for weights, ~5 GB for environments |
| OS | Linux (tested on Ubuntu 22.04) |

```bash
# Confirm GPU access
nvidia-smi
# Confirm conda
conda --version
```

---

## 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/athulramkumar/arc_fabric.git
cd arc_fabric
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

---

## 2. Download model weights

All weights go under `weights/`. Download them with `huggingface-cli`
(install it with `pip install huggingface-hub[cli]` if not available).

### Wan 2.1

```bash
# 1.3B (required)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir weights/wan21/Wan2.1-T2V-1.3B

# 14B (needed for hybrid schedule)
huggingface-cli download Wan-AI/Wan2.1-T2V-14B \
    --local-dir weights/wan21/Wan2.1-T2V-14B
```

**Size**: ~81 GB total (2.5 GB for 1.3B, 28 GB for 14B, shared VAE and T5).

### LongLive

```bash
# Base model weights (shared Wan 1.3B architecture)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir weights/longlive/Wan2.1-T2V-1.3B

# LongLive LoRA and checkpoints
huggingface-cli download Efficient-Large-Model/LongLive \
    --local-dir weights/longlive/LongLive
```

**Size**: ~25 GB total.

Then create the symlinks LongLive expects (the worker does this automatically,
but for manual runs):

```bash
cd models/longlive
mkdir -p wan_models
ln -sfn ../../weights/longlive/Wan2.1-T2V-1.3B wan_models/Wan2.1-T2V-1.3B
ln -sfn ../../weights/longlive/LongLive longlive_models
cd ../..
```

### LTX-Video

```bash
# 2B distilled (primary)
huggingface-cli download Lightricks/LTX-Video \
    ltxv-2b-0.9.8-distilled.safetensors \
    ltxv-spatial-upscaler-0.9.8.safetensors \
    --local-dir weights/ltx_video/ltxv-2b-0.9.8-distilled

# Text encoder (T5) — stored in the same directory
huggingface-cli download Lightricks/LTX-Video \
    --include "text_encoder/*" "scheduler/*" "model_index.json" \
    --local-dir weights/ltx_video/ltxv-2b-0.9.8-distilled
```

#### 13B (needed for hybrid schedule)

```bash
huggingface-cli download Lightricks/LTX-Video \
    ltxv-13b-0.9.8-distilled.safetensors \
    --local-dir weights/ltx_video/ltxv-13b-0.9.8-distilled
```

**Size**: ~80 GB total (2B ≈ 5 GB, 13B ≈ 27 GB, plus shared text encoder/upscaler).

---

## 3. Create conda environments

Each model gets its own isolated conda environment under `envs/`.
All environments use Python 3.10.

### Wan 2.1 (`envs/af-wan21`)

```bash
conda create --prefix envs/af-wan21 python=3.10 -y

# Install PyTorch first (CUDA 12.8)
envs/af-wan21/bin/pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Install Wan 2.1 dependencies
envs/af-wan21/bin/pip install \
    diffusers>=0.31.0 \
    transformers>=4.49.0 \
    accelerate>=1.1.1 \
    tokenizers>=0.20.3 \
    opencv-python>=4.9.0 \
    imageio imageio-ffmpeg \
    easydict ftfy tqdm dashscope \
    "numpy>=1.23.5,<2" \
    fastapi uvicorn \
    open_clip_torch

# Install flash attention (see "Installing flash-attn" section below)
envs/af-wan21/bin/pip install flash-attn --no-build-isolation
```

**Alternatively**, reproduce the exact tested environment:

```bash
conda create --prefix envs/af-wan21 python=3.10 -y
envs/af-wan21/bin/pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
envs/af-wan21/bin/pip install -r envs/wan21_freeze.txt
```

### LongLive (`envs/af-longlive`)

```bash
conda create --prefix envs/af-longlive python=3.10 -y

# Install PyTorch first (CUDA 12.8)
envs/af-longlive/bin/pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Install LongLive dependencies
envs/af-longlive/bin/pip install \
    omegaconf antlr4-python3-runtime==4.9.3 \
    einops peft accelerate diffusers==0.31.0 \
    transformers tokenizers \
    opencv-python imageio imageio-ffmpeg \
    easydict ftfy tqdm datasets \
    wandb lmdb matplotlib \
    flask flask-socketio \
    fastapi uvicorn \
    open_clip_torch av

# Install flash attention (see "Installing flash-attn" section below)
envs/af-longlive/bin/pip install flash-attn --no-build-isolation

# Install CLIP from source
envs/af-longlive/bin/pip install git+https://github.com/openai/CLIP.git
```

**Alternatively**, reproduce the exact tested environment:

```bash
conda create --prefix envs/af-longlive python=3.10 -y
envs/af-longlive/bin/pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
envs/af-longlive/bin/pip install -r envs/longlive_freeze.txt
```

### LTX-Video (`envs/af-ltx`)

```bash
conda create --prefix envs/af-ltx python=3.10 -y

# Install PyTorch first (CUDA 12.8)
envs/af-ltx/bin/pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Install LTX-Video from the submodule
envs/af-ltx/bin/pip install -e models/ltx_video

# Additional dependencies
envs/af-ltx/bin/pip install \
    imageio imageio-ffmpeg \
    fastapi uvicorn \
    open_clip_torch

# Install flash attention (see "Installing flash-attn" section below)
envs/af-ltx/bin/pip install flash-attn --no-build-isolation
```

**Alternatively**, reproduce the exact tested environment:

```bash
conda create --prefix envs/af-ltx python=3.10 -y
envs/af-ltx/bin/pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
envs/af-ltx/bin/pip install -r envs/ltx_freeze.txt
envs/af-ltx/bin/pip install -e models/ltx_video
```

---

## 4. Install orchestrator dependencies

The main UI server runs under its own conda environment (`af-main`):

```bash
conda create --prefix envs/af-main python=3.10 -y
envs/af-main/bin/pip install fastapi uvicorn pyyaml requests
```

---

## 5. Verify the setup

### Check environments

```bash
for env in af-wan21 af-longlive af-ltx; do
    echo "=== $env ==="
    envs/$env/bin/python -c "import torch; print(f'Python: {__import__(\"sys\").version.split()[0]}  torch: {torch.__version__}  CUDA: {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}')"
    echo
done
```

Expected output (versions may vary):

```
=== af-wan21 ===
Python: 3.10.x  torch: 2.10.0+cu128  CUDA: 12.8  GPU: NVIDIA H100 80GB HBM3

=== af-longlive ===
Python: 3.10.x  torch: 2.10.0+cu128  CUDA: 12.8  GPU: NVIDIA H100 80GB HBM3

=== af-ltx ===
Python: 3.10.x  torch: 2.10.0+cu128  CUDA: 12.8  GPU: NVIDIA H100 80GB HBM3
```

### Check flash attention

```bash
for env in af-wan21 af-longlive af-ltx; do
    echo -n "$env: "
    envs/$env/bin/python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>&1
done
```

### Check weights

```bash
echo "Wan 2.1 1.3B:  $(ls weights/wan21/Wan2.1-T2V-1.3B/*.pth 2>/dev/null | wc -l) files"
echo "Wan 2.1 14B:   $(ls weights/wan21/Wan2.1-T2V-14B/*.pth 2>/dev/null | wc -l) files"
echo "LongLive:      $(ls weights/longlive/LongLive/models/ 2>/dev/null | wc -l) files"
echo "LTX-Video 2B:  $(ls weights/ltx_video/ltxv-2b-0.9.8-distilled/*.safetensors 2>/dev/null | wc -l) files"
echo "LTX-Video 13B: $(ls weights/ltx_video/ltxv-13b-0.9.8-distilled/*.safetensors 2>/dev/null | wc -l) files"
```

### Quick smoke test (Wan 2.1)

```bash
CUDA_VISIBLE_DEVICES=0 envs/af-wan21/bin/python models/wan21/generate.py \
    --task t2v-1.3B \
    --size "480*832" \
    --ckpt_dir weights/wan21/Wan2.1-T2V-1.3B \
    --frame_num 33 \
    --sample_steps 30 \
    --base_seed 42 \
    --prompt "a cat sitting on a windowsill" \
    --save_file outputs/test_wan21.mp4
```

### Quick smoke test (LTX-Video)

```bash
CUDA_VISIBLE_DEVICES=0 envs/af-ltx/bin/python -c "
import sys; sys.path.insert(0, 'models/ltx_video')
from ltx_video.inference import infer, InferenceConfig
# You need a pipeline config — see workers/ltx_worker.py for how to generate one
"
```

---

## 6. Start the platform

```bash
envs/af-main/bin/python app/server.py
```

Open http://localhost:8000 in your browser. Models start **cold** and are
loaded into GPU on first use. Once warm, subsequent generations skip the
loading step entirely. Hybrid models (Wan 2.1 Hybrid, LTX Hybrid) load
both transformer variants and can serve standalone model requests without
reloading.

---

## Directory layout

```
arc_fabric/
├── app/
│   ├── server.py            # Main UI server (orchestrator)
│   └── static/index.html    # Web frontend
├── envs/                    # Conda environments (portable)
│   ├── af-wan21/
│   ├── af-longlive/
│   ├── af-ltx/
│   ├── wan21_freeze.txt     # pip freeze snapshots
│   ├── longlive_freeze.txt
│   └── ltx_freeze.txt
├── models/                  # Git submodules (source code)
│   ├── wan21/
│   ├── longlive/
│   ├── ltx_video/
│   └── inferix/
├── weights/                 # Downloaded model weights
│   ├── wan21/
│   ├── longlive/
│   └── ltx_video/
├── workers/                 # Persistent model worker servers
│   ├── wan21_worker.py
│   ├── hybrid_wan21_worker.py
│   ├── longlive_worker.py
│   ├── ltx_worker.py
│   └── hybrid_ltx_worker.py
├── outputs/                 # Generated videos
└── docs/
    ├── ARCHITECTURE.md
    ├── CODEBASE.md
    └── ENVIRONMENT_SETUP.md  # ← This file
```

---

## Installing flash-attn

Flash Attention 2 (`flash-attn`) provides significant speedups for all
Wan 2.1 and LongLive models. It must be built from source against your
CUDA toolkit.

### Standard install

```bash
envs/af-wan21/bin/pip install flash-attn --no-build-isolation
envs/af-longlive/bin/pip install flash-attn --no-build-isolation
envs/af-ltx/bin/pip install flash-attn --no-build-isolation
```

### CUDA version mismatch workaround

If your system CUDA toolkit (e.g. 13.0) differs from the version PyTorch
was compiled against (e.g. 12.8), the build will fail with a
`RuntimeError: The detected CUDA version mismatches...` error.

Since CUDA is backwards compatible, you can work around this by creating
an `nvcc` wrapper that reports the version PyTorch expects:

```bash
mkdir -p /tmp/cuda_compat/bin

cat > /tmp/cuda_compat/bin/nvcc << 'SCRIPT'
#!/bin/bash
if [[ "$*" == *"--version"* ]] || [[ "$*" == *"-V"* ]]; then
    /usr/local/cuda/bin/nvcc "$@" 2>&1 | sed 's/13\.0/12.8/g'
else
    exec /usr/local/cuda/bin/nvcc "$@"
fi
SCRIPT
chmod +x /tmp/cuda_compat/bin/nvcc

# Symlink real CUDA headers and libraries
for d in lib64 include targets; do
    ln -sfn /usr/local/cuda/$d /tmp/cuda_compat/$d
done

# Build with the compat wrapper
for env in af-wan21 af-longlive af-ltx; do
    CUDA_HOME=/tmp/cuda_compat PATH=/tmp/cuda_compat/bin:$PATH \
        MAX_JOBS=4 envs/$env/bin/pip install flash-attn --no-build-isolation
done
```

The build compiles CUDA kernels and takes **~30-50 minutes** (longer if
building all three environments in parallel).

### Skipping flash-attn

If you cannot install flash-attn, models will fall back to PyTorch's
`scaled_dot_product_attention` (SDPA). This is slower but fully
functional — no code changes required.

---

## Troubleshooting

### `conda: command not found`

Ensure conda is on your PATH. With Miniforge:

```bash
export PATH="$HOME/miniforge3/bin:$PATH"
conda init bash && source ~/.bashrc
```

### Out of GPU memory

Each model's approximate VRAM usage:

| Model | VRAM (approx) |
|-------|---------|
| Wan 2.1 1.3B | ~8 GB |
| Wan 2.1 14B | ~40 GB |
| Wan 2.1 Hybrid (14B+1.3B) | ~63 GB |
| LongLive 1.3B | ~25 GB |
| LTX-Video 2B | ~26 GB |
| LTX-Video 13B | ~50 GB |
| LTX-Video Hybrid (13B+2B) | ~52 GB |

With 2× H100 (80 GB each), you can run any two models simultaneously
on separate GPUs. The orchestrator handles LRU eviction automatically.
A warm hybrid model can serve standalone model requests (e.g. a warm
Wan Hybrid serves both Wan 1.3B and 14B requests) without reloading.

### Worker fails to start

Check the worker log:

```bash
cat outputs/ui/_workers/<model_id>.log
```

Common issues:
- Missing weights → re-run the download commands above
- Missing symlinks for LongLive → re-run the symlink commands
- Port already in use → kill stale processes: `lsof -ti:9100 | xargs kill -9`

### Migrating to a new machine

1. Copy the entire `arc_fabric/` directory (or re-clone + copy `envs/` and `weights/`)
2. Ensure the new machine has the same CUDA version
3. If CUDA versions differ, recreate envs using the freeze files
4. Verify with the smoke tests above
