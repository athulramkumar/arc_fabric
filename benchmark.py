#!/usr/bin/env python3
"""
Benchmark all available Arc Fabric models with a fixed prompt.
Tests: switching, concurrent GPUs, all models including caching variants.

Usage:
    python benchmark.py                    # full benchmark
    python benchmark.py --phase switching  # only switching test
    python benchmark.py --phase concurrent # only concurrent test
    python benchmark.py --phase generate   # only sequential generation
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API = "http://127.0.0.1:8000"
PROMPT = (
    "Generate anime style video - Two anthropomorphic cats playfully sparring "
    "in dojo, dynamic martial arts poses, wooden floor, traditional Japanese "
    "interior, dramatic lighting, high quality animation"
)
SEED = 42
TARGET_DURATION_S = 5

_LTX_MODELS = {"ltx_2b", "ltx_2b_dev", "ltx_13b", "ltx_13b_dev", "hybrid_ltx"}

MODEL_FPS = {
    "wan21_1_3b": 16, "wan21_14b": 16, "hybrid_wan21": 16,
    "longlive": 16, "longlive_interactive": 16,
    "ltx_2b": 24, "ltx_2b_dev": 24, "ltx_13b": 24, "ltx_13b_dev": 24, "hybrid_ltx": 24,
}

# LongLive uses latent-space frames: Wan2.1 VAE has 4:1 temporal compression.
# noise_frames = (target_video_frames - 1) / 4, and must be divisible by num_frame_per_block=3.
LONGLIVE_LATENT_FRAMES = 21  # → ~85 output frames at 16fps ≈ 5.3s


def _calc_frames(model_id: str) -> int:
    if model_id in ("longlive", "longlive_interactive"):
        return LONGLIVE_LATENT_FRAMES
    fps = MODEL_FPS.get(model_id, 16)
    raw = fps * TARGET_DURATION_S
    if model_id.startswith("ltx") or model_id == "hybrid_ltx":
        return ((raw - 2) // 8) * 8 + 1
    return ((raw - 1) // 4) * 4 + 1


def _gpu_memory():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        result = {}
        for line in out.strip().split("\n"):
            parts = line.split(",")
            if len(parts) == 2:
                result[int(parts[0].strip())] = int(parts[1].strip())
        return result
    except Exception:
        return {}


def _gpu_status():
    return requests.get(f"{API}/api/gpus").json()


def _model_status():
    models = requests.get(f"{API}/api/models").json()
    return {m["id"]: m for m in models}


def _submit_job(model_id: str, enable_caching: bool = False,
                cache_start_step: int = None, cache_end_step: int = None,
                cache_interval: int = 3) -> str:
    is_ltx = model_id in _LTX_MODELS
    num_frames = _calc_frames(model_id)
    payload = {
        "model_id": model_id,
        "prompt": PROMPT,
        "height": 480,
        "width": 704 if is_ltx else 832,
        "num_frames": num_frames,
        "seed": SEED,
        "enable_caching": enable_caching,
    }
    if enable_caching:
        if cache_start_step is not None:
            payload["cache_start_step"] = cache_start_step
        if cache_end_step is not None:
            payload["cache_end_step"] = cache_end_step
        payload["cache_interval"] = cache_interval
    r = requests.post(f"{API}/api/generate", json=payload)
    return r.json()["job_id"]


def _wait_job(job_id: str, timeout: int = 1800) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{API}/api/jobs/{job_id}")
        job = r.json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(3)
    return {"status": "timeout", "error": f"Timed out after {timeout}s"}


def _run_single(model_id: str, label: str = "", enable_caching: bool = False,
                cache_start_step: int = None, cache_end_step: int = None,
                cache_interval: int = 3) -> dict:
    fps = MODEL_FPS.get(model_id, 16)
    num_frames = _calc_frames(model_id)
    prefix = f"  [{label}] " if label else "  "
    cache_str = " [CACHING ON]" if enable_caching else ""

    print(f"{prefix}{model_id}: {num_frames}f @ {fps}fps{cache_str}")

    mem_before = _gpu_memory()
    t0 = time.time()
    job_id = _submit_job(model_id, enable_caching, cache_start_step,
                         cache_end_step, cache_interval)
    print(f"{prefix}  Job {job_id} submitted...")

    job = _wait_job(job_id)
    wall_time = time.time() - t0
    mem_after = _gpu_memory()

    entry = {
        "model_id": model_id,
        "label": label,
        "job_id": job_id,
        "status": job["status"],
        "num_frames": num_frames,
        "fps": fps,
        "video_duration_s": round(num_frames / fps, 2),
        "generation_time_s": job.get("elapsed"),
        "wall_time_s": round(wall_time, 1),
        "served_by": job.get("served_by"),
        "output_path": job.get("output_path"),
        "error": job.get("error"),
        "gpu_memory_mib": mem_after,
        "enable_caching": enable_caching,
        "reloaded": job.get("served_by") == model_id if job.get("served_by") else None,
    }

    if job["status"] == "completed":
        served = job.get("served_by", "")
        reuse_msg = ""
        if served and served != model_id:
            reuse_msg = f" [REUSED {served}]"
        print(f"{prefix}  OK: {job['elapsed']:.1f}s gen, {wall_time:.1f}s wall{reuse_msg}")
    else:
        err = (job.get("error") or "")[:200]
        print(f"{prefix}  FAILED: {err}")

    mem_str = ", ".join(f"GPU{k}:{v}MiB" for k, v in sorted(mem_after.items()))
    print(f"{prefix}  Memory: {mem_str}")
    return entry


# ---------------------------------------------------------------------------
# Phase 1: Switching test — hybrid ↔ standalone ↔ hybrid
# ---------------------------------------------------------------------------
def test_switching():
    print("\n" + "=" * 80)
    print("PHASE 1: HYBRID ↔ STANDALONE SWITCHING TEST")
    print("=" * 80)
    results = []

    print("\n--- Wan 2.1 family ---")
    r = _run_single("hybrid_wan21", "1a-hybrid")
    results.append(r)

    r = _run_single("wan21_1_3b", "1b-standalone-1.3B")
    results.append(r)
    if r.get("served_by") != "hybrid_wan21":
        print("  *** BUG: Expected reuse of hybrid_wan21! ***")

    r = _run_single("wan21_14b", "1c-standalone-14B")
    results.append(r)
    if r.get("served_by") != "hybrid_wan21":
        print("  *** BUG: Expected reuse of hybrid_wan21! ***")

    r = _run_single("hybrid_wan21", "1d-hybrid-again")
    results.append(r)
    if r.get("served_by") != "hybrid_wan21":
        print("  *** BUG: hybrid_wan21 was reloaded! ***")

    gpus = _gpu_status()
    print(f"\n  GPU state: {json.dumps(gpus, indent=2)}")

    print("\n--- LTX family ---")
    r = _run_single("hybrid_ltx", "2a-hybrid")
    results.append(r)

    r = _run_single("ltx_2b", "2b-standalone-2B")
    results.append(r)
    if r.get("served_by") != "hybrid_ltx":
        print("  *** BUG: Expected reuse of hybrid_ltx! ***")

    r = _run_single("ltx_13b", "2c-standalone-13B")
    results.append(r)
    if r.get("served_by") != "hybrid_ltx":
        print("  *** BUG: Expected reuse of hybrid_ltx! ***")

    r = _run_single("hybrid_ltx", "2d-hybrid-again")
    results.append(r)
    if r.get("served_by") != "hybrid_ltx":
        print("  *** BUG: hybrid_ltx was reloaded! ***")

    gpus = _gpu_status()
    print(f"\n  GPU state: {json.dumps(gpus, indent=2)}")

    return results


# ---------------------------------------------------------------------------
# Phase 2: Concurrent generation on different GPUs
# ---------------------------------------------------------------------------
def test_concurrent():
    print("\n" + "=" * 80)
    print("PHASE 2: CONCURRENT GENERATION ON SEPARATE GPUs")
    print("=" * 80)

    gpus = _gpu_status()
    print(f"  Current GPUs: {json.dumps(gpus, indent=2)}")

    model_a = "hybrid_wan21"
    model_b = "ltx_2b"

    print(f"\n  Submitting {model_a} and {model_b} simultaneously...")

    t0 = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(_submit_job, model_a): model_a,
            pool.submit(_submit_job, model_b): model_b,
        }
        job_ids = {}
        for fut in as_completed(futures):
            mid = futures[fut]
            jid = fut.result()
            job_ids[mid] = jid
            print(f"  {mid}: job {jid}")

    print("  Both submitted. Waiting for completion...")

    with ThreadPoolExecutor(max_workers=2) as pool:
        wait_futures = {
            pool.submit(_wait_job, jid, 900): mid
            for mid, jid in job_ids.items()
        }
        for fut in as_completed(wait_futures):
            mid = wait_futures[fut]
            job = fut.result()
            wall = time.time() - t0
            gen = job.get("elapsed", "?")
            print(f"  {mid}: {job['status']} — gen {gen}s, wall {wall:.1f}s, served_by={job.get('served_by')}")
            results.append({
                "model_id": mid,
                "label": f"concurrent-{mid}",
                "job_id": job_ids[mid],
                "status": job["status"],
                "generation_time_s": job.get("elapsed"),
                "wall_time_s": round(wall, 1),
                "served_by": job.get("served_by"),
                "error": job.get("error"),
            })

    gpus = _gpu_status()
    mem = _gpu_memory()
    print(f"\n  GPU state after concurrent run:")
    for g in gpus:
        m = mem.get(g["gpu_id"], 0)
        print(f"    GPU {g['gpu_id']}: {g['model_id'] or 'free'} ({m} MiB)")

    occupied = [g for g in gpus if g["model_id"]]
    if len(occupied) >= 2:
        print("  PASS: Both GPUs used concurrently")
    else:
        print("  *** WARNING: Expected 2 GPUs in use ***")

    return results


# ---------------------------------------------------------------------------
# Phase 3: All models + hybrid caching variants
# ---------------------------------------------------------------------------
def test_sequential():
    print("\n" + "=" * 80)
    print("PHASE 3: ALL MODELS + CACHING VARIANTS")
    print("=" * 80)

    results = []
    step = [0]

    def run(model_id, label, **kwargs):
        step[0] += 1
        total = 11  # total expected runs
        print(f"\n[{step[0]}/{total}]")
        r = _run_single(model_id, label, **kwargs)
        results.append(r)
        return r

    # --- Wan 2.1 family ---
    run("wan21_1_3b",   "wan21-1.3B")
    run("wan21_14b",    "wan21-14B")
    run("hybrid_wan21", "wan21-hybrid")
    run("hybrid_wan21", "wan21-hybrid-cached",
        enable_caching=True, cache_start_step=5, cache_end_step=45, cache_interval=3)

    # --- LTX family ---
    run("ltx_2b",     "ltx-2B")
    run("ltx_13b",    "ltx-13B")
    run("hybrid_ltx", "ltx-hybrid")
    run("hybrid_ltx", "ltx-hybrid-cached",
        enable_caching=True, cache_start_step=2, cache_end_step=8, cache_interval=2)

    # --- LongLive ---
    run("longlive", "longlive")

    # --- Hybrid caching comparison (caching off vs on, back-to-back) ---
    run("hybrid_wan21", "wan21-hybrid-nocache")
    run("hybrid_wan21", "wan21-hybrid-cache-compare",
        enable_caching=True, cache_start_step=5, cache_end_step=45, cache_interval=3)

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(all_results: list[dict]):
    print(f"\n{'='*100}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*100}")
    header = (f"{'Label':<30} {'Model':<20} {'Status':<9} {'Gen(s)':<9} "
              f"{'Wall(s)':<9} {'Cache':<6} {'Served By':<18}")
    print(header)
    print("-" * 100)
    for r in all_results:
        label = (r.get("label") or "")[:29]
        gen = f"{r['generation_time_s']:.1f}" if r.get("generation_time_s") else "—"
        wall = f"{r['wall_time_s']:.1f}" if r.get("wall_time_s") else "—"
        served = r.get("served_by") or "—"
        status = r.get("status", "?")
        cache = "yes" if r.get("enable_caching") else "no"
        print(f"{label:<30} {r['model_id']:<20} {status:<9} {gen:<9} "
              f"{wall:<9} {cache:<6} {served:<18}")

    out_path = Path("/workspace/arc_fabric/outputs/benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["switching", "concurrent", "generate", "all"],
                        default="all")
    args = parser.parse_args()

    try:
        requests.get(f"{API}/api/models", timeout=5)
    except Exception as e:
        print(f"Server not reachable at {API}: {e}")
        sys.exit(1)

    all_results = []

    if args.phase in ("switching", "all"):
        all_results.extend(test_switching())

    if args.phase in ("concurrent", "all"):
        all_results.extend(test_concurrent())

    if args.phase in ("generate", "all"):
        all_results.extend(test_sequential())

    print_summary(all_results)


if __name__ == "__main__":
    main()
