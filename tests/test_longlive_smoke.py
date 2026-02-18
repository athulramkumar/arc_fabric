#!/usr/bin/env python3
"""
LongLive smoke tests — layered validation of the onboarded submodule.

Tier 1: Import verification (no GPU, no weights)
Tier 2: Weight file existence checks
Tier 3: Pipeline construction + minimal inference (requires GPU + weights)
"""
import sys
import os
import time

SUBMODULE_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "longlive")
SUBMODULE_DIR = os.path.abspath(SUBMODULE_DIR)
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights", "longlive")
WEIGHTS_DIR = os.path.abspath(WEIGHTS_DIR)

passed = 0
failed = 0
errors = []


def report(name, ok, detail=""):
    global passed, failed, errors
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    if ok:
        passed += 1
    else:
        failed += 1
        errors.append((name, detail))


# ──────────────────────────────────────────────
# Tier 1: Import verification
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("TIER 1: Import verification")
print("=" * 70)

sys.path.insert(0, SUBMODULE_DIR)

try:
    import torch
    report("import torch", True, f"v{torch.__version__}")
except Exception as e:
    report("import torch", False, str(e))

try:
    from omegaconf import OmegaConf
    report("import omegaconf", True)
except Exception as e:
    report("import omegaconf", False, str(e))

try:
    from einops import rearrange
    report("import einops", True)
except Exception as e:
    report("import einops", False, str(e))

try:
    import peft
    report("import peft", True, f"v{peft.__version__}")
except Exception as e:
    report("import peft", False, str(e))

try:
    from utils.scheduler import FlowMatchScheduler
    report("import FlowMatchScheduler", True)
except Exception as e:
    report("import FlowMatchScheduler", False, str(e))

try:
    from utils.lora_utils import configure_lora_for_model
    report("import lora_utils", True)
except Exception as e:
    report("import lora_utils", False, str(e))

try:
    from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
    report("import memory utils", True)
except Exception as e:
    report("import memory utils", False, str(e))

try:
    from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
    report("import WanDiffusionWrapper (class def only)", True)
except Exception as e:
    report("import WanDiffusionWrapper", False, str(e))

try:
    from pipeline import CausalInferencePipeline
    report("import CausalInferencePipeline (class def only)", True)
except Exception as e:
    report("import CausalInferencePipeline", False, str(e))

try:
    from pipeline import InteractiveCausalInferencePipeline
    report("import InteractiveCausalInferencePipeline (class def only)", True)
except Exception as e:
    report("import InteractiveCausalInferencePipeline", False, str(e))

try:
    from wan.modules.causal_model import CausalWanModel
    report("import CausalWanModel", True)
except Exception as e:
    report("import CausalWanModel", False, str(e))

try:
    from wan.modules.vae import _video_vae
    report("import _video_vae", True)
except Exception as e:
    report("import _video_vae", False, str(e))

try:
    from wan.modules.t5 import umt5_xxl
    report("import umt5_xxl", True)
except Exception as e:
    report("import umt5_xxl", False, str(e))

try:
    from utils.dataset import TextDataset
    report("import TextDataset", True)
except Exception as e:
    report("import TextDataset", False, str(e))

# ──────────────────────────────────────────────
# Tier 2: Weight file existence
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("TIER 2: Weight file existence checks")
print("=" * 70)

wan_base = os.path.join(WEIGHTS_DIR, "Wan2.1-T2V-1.3B")
ll_base = os.path.join(WEIGHTS_DIR, "LongLive")

expected_files = {
    "T5 encoder weights": os.path.join(wan_base, "models_t5_umt5-xxl-enc-bf16.pth"),
    "VAE weights": os.path.join(wan_base, "Wan2.1_VAE.pth"),
    "Diffusion model (safetensors)": os.path.join(wan_base, "diffusion_pytorch_model.safetensors"),
    "Wan config.json": os.path.join(wan_base, "config.json"),
    "T5 tokenizer (spiece.model)": os.path.join(wan_base, "google", "umt5-xxl", "spiece.model"),
    "T5 tokenizer_config.json": os.path.join(wan_base, "google", "umt5-xxl", "tokenizer_config.json"),
    "LongLive base checkpoint": os.path.join(ll_base, "models", "longlive_base.pt"),
    "LongLive LoRA checkpoint": os.path.join(ll_base, "models", "lora.pt"),
}

for name, path in expected_files.items():
    exists = os.path.isfile(path)
    size_str = ""
    if exists:
        size_gb = os.path.getsize(path) / (1024 ** 3)
        size_str = f"{size_gb:.2f} GB" if size_gb > 0.01 else f"{os.path.getsize(path)} bytes"
    report(name, exists, size_str if exists else f"NOT FOUND: {path}")

symlink_wan = os.path.join(SUBMODULE_DIR, "wan_models")
symlink_ll = os.path.join(SUBMODULE_DIR, "longlive_models")
report("wan_models symlink", os.path.islink(symlink_wan), f"-> {os.readlink(symlink_wan)}" if os.path.islink(symlink_wan) else "MISSING")
report("longlive_models symlink", os.path.islink(symlink_ll), f"-> {os.readlink(symlink_ll)}" if os.path.islink(symlink_ll) else "MISSING")

# ──────────────────────────────────────────────
# Tier 3: Pipeline construction + minimal inference
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("TIER 3: Pipeline construction + minimal inference (GPU required)")
print("=" * 70)

if not torch.cuda.is_available():
    print("  [SKIP] No CUDA device available — skipping Tier 3")
else:
    print(f"  CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        print(f"    GPU {i}: {name} ({mem:.1f} GB)")

    os.chdir(SUBMODULE_DIR)

    # 3a: Construct pipeline from config
    print("\n  --- 3a: Constructing CausalInferencePipeline from config ---")
    t0 = time.time()
    try:
        config = OmegaConf.load(os.path.join(SUBMODULE_DIR, "configs", "longlive_inference.yaml"))
        config.distributed = False

        device = torch.device("cuda")
        pipeline = CausalInferencePipeline(config, device=device)
        report("Pipeline __init__ (loads Wan base model + T5 + VAE)", True, f"{time.time() - t0:.1f}s")
    except Exception as e:
        report("Pipeline __init__", False, str(e))
        pipeline = None

    # 3b: Load generator checkpoint
    if pipeline is not None:
        print("\n  --- 3b: Loading generator (longlive_base.pt) ---")
        t0 = time.time()
        try:
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict or "generator_ema" in state_dict:
                raw = state_dict["generator_ema" if config.use_ema else "generator"]
            elif "model" in state_dict:
                raw = state_dict["model"]
            else:
                raise ValueError("Generator key not found in checkpoint")
            pipeline.generator.load_state_dict(raw)
            report("Load longlive_base.pt into generator", True, f"{time.time() - t0:.1f}s")
        except Exception as e:
            report("Load longlive_base.pt", False, str(e))

    # 3c: Apply LoRA
    if pipeline is not None:
        print("\n  --- 3c: Applying LoRA adapter ---")
        t0 = time.time()
        try:
            pipeline.generator.model = configure_lora_for_model(
                pipeline.generator.model,
                model_name="generator",
                lora_config=config.adapter,
                is_main_process=True,
            )
            lora_ckpt = torch.load(config.lora_ckpt, map_location="cpu")
            if isinstance(lora_ckpt, dict) and "generator_lora" in lora_ckpt:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_ckpt["generator_lora"])
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_ckpt)
            pipeline.is_lora_enabled = True
            report("LoRA applied + lora.pt loaded", True, f"{time.time() - t0:.1f}s")
        except Exception as e:
            report("LoRA application", False, str(e))

    # 3d: Move to device and run minimal inference
    if pipeline is not None:
        print("\n  --- 3d: Moving to device + minimal 1-block inference ---")
        t0 = time.time()
        try:
            pipeline = pipeline.to(dtype=torch.bfloat16)
            DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
            pipeline.generator.to(device=device)
            pipeline.vae.to(device=device)
            report("Pipeline moved to GPU (bf16)", True, f"{time.time() - t0:.1f}s")
        except Exception as e:
            report("Move to device", False, str(e))

        print("\n  --- 3e: Running single-block inference (3 frames) ---")
        t0 = time.time()
        try:
            num_frames = 3
            sampled_noise = torch.randn(
                [1, num_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
            )
            prompts = ["A cat sitting on a windowsill watching birds outside"]

            with torch.no_grad():
                video, latents = pipeline.inference(
                    noise=sampled_noise,
                    text_prompts=prompts,
                    return_latents=True,
                    low_memory=True,
                    profile=False,
                )

            report(
                "Single-block inference (3 frames)",
                True,
                f"video shape={list(video.shape)}, latents shape={list(latents.shape)}, {time.time() - t0:.1f}s",
            )

            v_min, v_max = video.min().item(), video.max().item()
            report(
                "Output range check",
                0.0 <= v_min and v_max <= 1.0,
                f"min={v_min:.4f}, max={v_max:.4f}",
            )
        except Exception as e:
            import traceback
            report("Single-block inference", False, str(e))
            traceback.print_exc()

    # Cleanup
    if pipeline is not None:
        del pipeline
        torch.cuda.empty_cache()

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"SUMMARY: {passed} passed, {failed} failed")
print("=" * 70)
if errors:
    print("\nFailed tests:")
    for name, detail in errors:
        print(f"  - {name}: {detail}")

sys.exit(0 if failed == 0 else 1)
