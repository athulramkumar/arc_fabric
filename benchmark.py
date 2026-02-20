#!/usr/bin/env python3
"""
Benchmark all available Arc Fabric models with a fixed prompt.
Tests: hybrid↔standalone switching, concurrent GPU requests, quality metrics.

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


def _calc_frames(model_id: str) -> int:
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


def _submit_job(model_id: str) -> str:
    is_ltx = model_id in _LTX_MODELS
    num_frames = _calc_frames(model_id)
    payload = {
        "model_id": model_id,
        "prompt": PROMPT,
        "height": 480,
        "width": 704 if is_ltx else 832,
        "num_frames": num_frames,
        "seed": SEED,
    }
    r = requests.post(f"{API}/api/generate", json=payload)
    return r.json()["job_id"]


def _wait_job(job_id: str, timeout: int = 600) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{API}/api/jobs/{job_id}")
        job = r.json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(3)
    return {"status": "timeout", "error": f"Timed out after {timeout}s"}


def _run_single(model_id: str, label: str = "") -> dict:
    fps = MODEL_FPS.get(model_id, 16)
    num_frames = _calc_frames(model_id)
    prefix = f"  [{label}] " if label else "  "

    print(f"{prefix}{model_id}: {num_frames}f @ {fps}fps = {num_frames/fps:.1f}s")

    mem_before = _gpu_memory()
    t0 = time.time()
    job_id = _submit_job(model_id)
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
        "reloaded": job.get("served_by") == model_id if job.get("served_by") else None,
    }

    if job["status"] == "completed":
        served = job.get("served_by", "")
        reuse_msg = ""
        if served and served != model_id:
            reuse_msg = f" [REUSED {served}]"
        print(f"{prefix}  OK: {job['elapsed']:.1f}s gen, {wall_time:.1f}s wall{reuse_msg}")
    else:
        err = (job.get("error") or "")[:100]
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
    # 1a: Load hybrid → both transformers loaded
    r = _run_single("hybrid_wan21", "1a-hybrid")
    results.append(r)

    # 1b: Run standalone 1.3B → should reuse hybrid worker, NO reload
    r = _run_single("wan21_1_3b", "1b-standalone-1.3B")
    results.append(r)
    if r.get("served_by") != "hybrid_wan21":
        print("  *** BUG: Expected reuse of hybrid_wan21 but got fresh worker! ***")

    # 1c: Run standalone 14B → should reuse hybrid worker, NO reload
    r = _run_single("wan21_14b", "1c-standalone-14B")
    results.append(r)
    if r.get("served_by") != "hybrid_wan21":
        print("  *** BUG: Expected reuse of hybrid_wan21 but got fresh worker! ***")

    # 1d: Run hybrid again → should still be warm, NO reload
    r = _run_single("hybrid_wan21", "1d-hybrid-again")
    results.append(r)
    if r.get("served_by") != "hybrid_wan21":
        print("  *** BUG: hybrid_wan21 was reloaded unnecessarily! ***")

    # Verify GPU state
    gpus = _gpu_status()
    print(f"\n  GPU state: {json.dumps(gpus, indent=2)}")

    print("\n--- LTX family ---")
    # 2a: Load hybrid LTX
    r = _run_single("hybrid_ltx", "2a-hybrid")
    results.append(r)

    # 2b: Run standalone 2B → should reuse hybrid
    r = _run_single("ltx_2b", "2b-standalone-2B")
    results.append(r)
    if r.get("served_by") != "hybrid_ltx":
        print("  *** BUG: Expected reuse of hybrid_ltx but got fresh worker! ***")

    # 2c: Back to hybrid → still warm
    r = _run_single("hybrid_ltx", "2c-hybrid-again")
    results.append(r)
    if r.get("served_by") != "hybrid_ltx":
        print("  *** BUG: hybrid_ltx was reloaded unnecessarily! ***")

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

    models_status = _model_status()
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
            pool.submit(_wait_job, jid, 600): mid
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
# Phase 3: Sequential generation across all models
# ---------------------------------------------------------------------------
def test_sequential():
    print("\n" + "=" * 80)
    print("PHASE 3: SEQUENTIAL GENERATION — ALL MODELS")
    print("=" * 80)

    order = [
        "wan21_1_3b",
        "wan21_14b",
        "hybrid_wan21",
        "ltx_2b",
        "hybrid_ltx",
        "longlive",
    ]

    results = []
    for i, model_id in enumerate(order):
        print(f"\n[{i+1}/{len(order)}]")
        r = _run_single(model_id, f"seq-{i+1}")
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(all_results: list[dict]):
    print(f"\n{'='*90}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*90}")
    print(f"{'Label':<28} {'Model':<20} {'Status':<9} {'Gen(s)':<9} {'Wall(s)':<9} {'Served By':<18}")
    print("-" * 90)
    for r in all_results:
        label = (r.get("label") or "")[:27]
        gen = f"{r['generation_time_s']:.1f}" if r.get("generation_time_s") else "—"
        wall = f"{r['wall_time_s']:.1f}" if r.get("wall_time_s") else "—"
        served = r.get("served_by") or "—"
        status = r.get("status", "?")
        print(f"{label:<28} {r['model_id']:<20} {status:<9} {gen:<9} {wall:<9} {served:<18}")

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
