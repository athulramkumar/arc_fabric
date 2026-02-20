#!/usr/bin/env python3
"""
Video quality metrics for Arc Fabric benchmark outputs.
Computes per-video: resolution, frames, fps, duration, file size,
frame sharpness (Laplacian variance), temporal consistency (SSIM between consecutive frames),
and color diversity.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


def compute_ssim_gray(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simplified SSIM between two grayscale images."""
    C1, C2 = 6.5025, 58.5225
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(num / den)


def analyze_video(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": f"Cannot open {path}"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / fps if fps > 0 else 0
    file_size_mb = Path(path).stat().st_size / (1024 * 1024)

    sharpness_scores = []
    ssim_scores = []
    color_variances = []
    prev_gray = None

    sample_interval = max(1, frame_count // 50)  # sample ~50 frames max

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

            lap = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness_scores.append(float(lap.var()))

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_variances.append(float(hsv[:, :, 0].std()))

            if prev_gray is not None:
                ssim_scores.append(compute_ssim_gray(prev_gray, gray))
            prev_gray = gray

        idx += 1

    cap.release()

    return {
        "file": str(path),
        "resolution": f"{width}x{height}",
        "frame_count": frame_count,
        "fps": round(fps, 1),
        "duration_s": round(duration_s, 2),
        "file_size_mb": round(file_size_mb, 2),
        "sharpness_mean": round(np.mean(sharpness_scores), 1) if sharpness_scores else None,
        "sharpness_std": round(np.std(sharpness_scores), 1) if sharpness_scores else None,
        "temporal_consistency_ssim": round(np.mean(ssim_scores), 4) if ssim_scores else None,
        "temporal_consistency_min": round(min(ssim_scores), 4) if ssim_scores else None,
        "color_diversity": round(np.mean(color_variances), 1) if color_variances else None,
    }


def main():
    benchmark_path = Path("/workspace/arc_fabric/outputs/benchmark_results.json")
    if not benchmark_path.exists():
        print("No benchmark results found. Run benchmark.py first.")
        sys.exit(1)

    with open(benchmark_path) as f:
        results = json.load(f)

    print(f"{'='*100}")
    print("VIDEO QUALITY METRICS")
    print(f"{'='*100}")

    enriched = []
    for r in results:
        if r.get("status") != "completed" or not r.get("output_path"):
            enriched.append(r)
            continue

        video_path = Path("/workspace/arc_fabric/outputs/ui") / r["job_id"] / "output.mp4"
        if not video_path.exists():
            alt = list(Path(f"/workspace/arc_fabric/outputs/ui/{r['job_id']}").glob("*.mp4"))
            if alt:
                video_path = alt[0]
            else:
                r["quality"] = {"error": "video not found"}
                enriched.append(r)
                continue

        metrics = analyze_video(str(video_path))
        r["quality"] = metrics
        enriched.append(r)

        label = r.get("label") or r["model_id"]
        print(f"\n{label} ({r['model_id']})")
        print(f"  Job: {r['job_id']} | Served by: {r.get('served_by', '—')}")
        print(f"  Video: {metrics['resolution']}, {metrics['frame_count']}f @ {metrics['fps']}fps, "
              f"{metrics['duration_s']}s, {metrics['file_size_mb']:.1f}MB")
        gen_time = r.get("generation_time_s", "?")
        print(f"  Inference: {gen_time}s generation | {r.get('wall_time_s', '?')}s wall")
        mem = r.get("gpu_memory_mib", {})
        if mem:
            mem_str = ", ".join(f"GPU{k}:{v}MiB" for k, v in sorted(mem.items()))
            print(f"  GPU Memory: {mem_str}")
        print(f"  Sharpness: {metrics.get('sharpness_mean', '?')} ± {metrics.get('sharpness_std', '?')}")
        print(f"  Temporal SSIM: {metrics.get('temporal_consistency_ssim', '?')} "
              f"(min: {metrics.get('temporal_consistency_min', '?')})")
        print(f"  Color diversity: {metrics.get('color_diversity', '?')}")

    # Summary comparison table
    print(f"\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")

    # Group by unique (model_id, label) — show one row per unique test
    seen = set()
    rows = []
    for r in enriched:
        if r.get("status") != "completed" or "quality" not in r:
            continue
        key = r.get("label") or r["model_id"]
        if key in seen:
            continue
        seen.add(key)
        q = r["quality"]
        rows.append({
            "label": key[:25],
            "model": r["model_id"][:18],
            "gen_s": r.get("generation_time_s", 0),
            "sharp": q.get("sharpness_mean", 0),
            "ssim": q.get("temporal_consistency_ssim", 0),
            "color": q.get("color_diversity", 0),
            "size_mb": q.get("file_size_mb", 0),
            "res": q.get("resolution", "?"),
        })

    header = (f"{'Label':<26} {'Model':<19} {'Gen(s)':<8} {'Res':<10} "
              f"{'Sharp':<8} {'SSIM':<7} {'Color':<7} {'Size(MB)':<9}")
    print(header)
    print("-" * 100)
    for row in rows:
        gen = f"{row['gen_s']:.1f}" if row['gen_s'] else "—"
        sharp = f"{row['sharp']:.0f}" if row['sharp'] else "—"
        ssim = f"{row['ssim']:.3f}" if row['ssim'] else "—"
        color = f"{row['color']:.0f}" if row['color'] else "—"
        size = f"{row['size_mb']:.1f}"
        print(f"{row['label']:<26} {row['model']:<19} {gen:<8} {row['res']:<10} "
              f"{sharp:<8} {ssim:<7} {color:<7} {size:<9}")

    out_path = Path("/workspace/arc_fabric/outputs/benchmark_with_quality.json")
    with open(out_path, "w") as f:
        json.dump(enriched, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
