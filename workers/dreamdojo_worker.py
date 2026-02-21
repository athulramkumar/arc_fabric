"""
DreamDojo worker — action-conditioned Video2World inference.
Loads model once, serves predictions from dataset samples.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/workspace/arc_fabric")
DREAMDOJO_DIR = ROOT / "models" / "dreamdojo"
DATASET_BASE = DREAMDOJO_DIR / "datasets" / "PhysicalAI-Robotics-GR00T-Teleop-GR1" / "In-lab_Eval"

os.chdir(str(DREAMDOJO_DIR))
sys.path.insert(0, str(DREAMDOJO_DIR))

app = FastAPI()

_state = {
    "video2world_cli": None,
    "datasets": {},
    "sample_list": [],
    "ready": False,
    "model_name": "dreamdojo_2b",
}

_EXPERIMENTS = {
    "dreamdojo_2b": "dreamdojo_2b_480_640_gr1",
    "dreamdojo_14b": "dreamdojo_14b_480_640_gr1",
}

_CHECKPOINTS = {
    "dreamdojo_2b": str(DREAMDOJO_DIR / "checkpoints" / "2B_GR1_post-train" / "iter_000050000" / "model_ema_bf16.pt"),
    "dreamdojo_14b": str(DREAMDOJO_DIR / "checkpoints" / "14B_GR1_post-train" / "iter_000050000" / "model_ema_bf16.pt"),
}


def _load_dataset():
    """Load evaluation dataset samples from all available tasks."""
    from groot_dreams.dataloader import MultiVideoActionDataset

    all_samples = []
    datasets = {}

    if not DATASET_BASE.exists():
        logger.warning(f"Dataset directory not found: {DATASET_BASE}")
        return datasets, all_samples

    task_dirs = sorted([d.name for d in DATASET_BASE.iterdir() if d.is_dir()])
    for task_name in task_dirs:
        task_path = str(DATASET_BASE / task_name)
        try:
            ds = MultiVideoActionDataset(
                num_frames=49,
                dataset_path=task_path,
                data_split="full",
                deterministic_uniform_sampling=True,
            )
            datasets[task_name] = ds
            n_samples = min(len(ds), 10)
            for i in range(n_samples):
                display = task_name.replace("gr1_unified.", "").replace("_robot", "").replace("_", " ").title()
                all_samples.append({
                    "id": f"{task_name}_{i}",
                    "task": task_name,
                    "task_display": display,
                    "index": i,
                    "label": f"{display} #{i}",
                })
            logger.info(f"Loaded task {task_name}: {n_samples} samples")
        except Exception as e:
            logger.warning(f"Could not load dataset {task_name}: {e}")

    return datasets, all_samples


def _load_model(model_name: str):
    """Load the DreamDojo model and dataset."""
    logger.info(f"Loading DreamDojo model: {model_name}")
    t0 = time.time()

    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    experiment = _EXPERIMENTS[model_name]
    ckpt_path = _CHECKPOINTS[model_name]

    logger.info(f"Loading Video2WorldInference: experiment={experiment}, ckpt={ckpt_path}")
    video2world_cli = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=ckpt_path,
        s3_credential_path="",
        context_parallel_size=1,
        config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )

    load_time = time.time() - t0
    mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    logger.info(f"Model loaded in {load_time:.1f}s. GPU memory: {mem_gb:.2f} GB")

    datasets, sample_list = _load_dataset()

    _state["video2world_cli"] = video2world_cli
    _state["datasets"] = datasets
    _state["sample_list"] = sample_list
    _state["model_name"] = model_name
    _state["ready"] = True

    logger.info(f"Ready. {len(sample_list)} samples available across {len(datasets)} tasks.")


@app.on_event("startup")
async def startup():
    model_name = os.environ.get("DREAMDOJO_MODEL", "dreamdojo_2b")
    _load_model(model_name)


@app.get("/health")
async def health():
    if _state["ready"]:
        return {"status": "ready", "model": _state["model_name"], "samples": len(_state["sample_list"])}
    return {"status": "loading"}


@app.get("/samples")
async def list_samples():
    return {"samples": _state["sample_list"]}


class GenerateRequest(BaseModel):
    sample_id: str
    output_dir: str
    guidance: float = 0.0
    prompt: str = ""
    num_frames: int = 49
    seed: int = 42


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not _state["ready"]:
        raise HTTPException(503, "Model not ready")

    sample = next((s for s in _state["sample_list"] if s["id"] == req.sample_id), None)
    if not sample:
        raise HTTPException(404, f"Sample not found: {req.sample_id}")

    t0 = time.time()
    output_dir = Path(req.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import mediapy
        import torchvision
        import piq

        ds = _state["datasets"][sample["task"]]
        data = ds[sample["index"]]

        gt_video = data["video"].permute(1, 2, 3, 0)  # (T, H, W, C)
        img_array = data["video"].transpose(0, 1)[:1]  # (1, C, H, W)
        actions = data["action"][:req.num_frames - 1].numpy()
        lam_video = data["lam_video"]

        video2world_cli = _state["video2world_cli"]
        chunk_size = 12
        chunk_video = []
        first_round = True

        for i in range(0, len(actions), chunk_size):
            actions_chunk = actions[i:i + chunk_size]
            if actions_chunk.shape[0] != chunk_size:
                break

            current_lam_video = lam_video[i * 2:(i + chunk_size) * 2]

            if not first_round:
                img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0) * 255.0
            else:
                img_tensor = img_array
            first_round = False

            num_video_frames = actions_chunk.shape[0] + 1
            vid_input = torch.cat([img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], dim=0)
            vid_input = vid_input.to(torch.uint8).unsqueeze(0).permute(0, 2, 1, 3, 4)

            video = video2world_cli.generate_vid2world(
                prompt="",
                input_path=vid_input,
                action=torch.from_numpy(actions_chunk).float(),
                guidance=req.guidance,
                num_video_frames=num_video_frames,
                num_latent_conditional_frames=1,
                resolution="480,640",
                seed=req.seed + i,
                negative_prompt="",
                lam_video=current_lam_video,
            )

            video_normalized = (video - (-1)) / (1 - (-1))
            video_clamped = (torch.clamp(video_normalized[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
            img_array = video_clamped[-1]
            chunk_video.append(video_clamped)
            logger.info(f"Chunk {len(chunk_video)} done (frames {i}-{i+chunk_size})")

        chunk_list = [chunk_video[0]] + [chunk_video[k][:chunk_size] for k in range(1, len(chunk_video))]
        pred_frames = np.concatenate(chunk_list, axis=0)

        gt_path = str(output_dir / "gt.mp4")
        pred_path = str(output_dir / "pred.mp4")
        merged_path = str(output_dir / "merged.mp4")
        actions_path = str(output_dir / "actions.json")

        mediapy.write_video(gt_path, gt_video.numpy(), fps=10)
        mediapy.write_video(pred_path, pred_frames, fps=10)

        min_len = min(len(gt_video), len(pred_frames))
        merged = np.concatenate([gt_video[:min_len].numpy(), pred_frames[:min_len]], axis=2)
        mediapy.write_video(merged_path, merged, fps=10)

        action_summary = _summarize_actions(actions, sample["task"])
        with open(actions_path, "w") as f:
            json.dump(action_summary, f)

        x_batch = torch.clamp(torch.from_numpy(pred_frames[:min_len]).float() / 255.0, 0, 1).permute(0, 3, 1, 2)
        y_batch = torch.clamp(gt_video[:min_len].float() / 255.0, 0, 1).permute(0, 3, 1, 2)
        psnr_val = piq.psnr(x_batch, y_batch).mean().item()
        ssim_val = piq.ssim(x_batch, y_batch).mean().item()
        lpips_val = piq.LPIPS()(x_batch, y_batch).mean().item()

        elapsed = time.time() - t0
        logger.info(f"Generation done in {elapsed:.1f}s. PSNR={psnr_val:.2f} SSIM={ssim_val:.3f} LPIPS={lpips_val:.3f}")

        return {
            "status": "success",
            "gt_path": gt_path,
            "pred_path": pred_path,
            "merged_path": merged_path,
            "actions_path": actions_path,
            "output_path": pred_path,
            "metrics": {"psnr": round(psnr_val, 3), "ssim": round(ssim_val, 3), "lpips": round(lpips_val, 3)},
            "elapsed": round(elapsed, 1),
            "sample": sample,
        }

    except Exception as e:
        logger.exception(f"Generation failed for {req.sample_id}")
        return {"status": "error", "error": str(e)}


def _summarize_actions(actions: np.ndarray, task_name: str) -> dict:
    """Convert raw action tensor to JSON-serializable summary for chart visualization."""
    T = actions.shape[0]
    dim = actions.shape[1] if len(actions.shape) > 1 else 1

    norms = np.linalg.norm(actions, axis=-1).tolist() if len(actions.shape) > 1 else actions.tolist()

    labels_7d = ["Dx", "Dy", "Dz", "Droll", "Dpitch", "Dyaw", "Gripper"]
    if dim == 7:
        groups = [actions[:, i].tolist() for i in range(7)]
        group_labels = labels_7d
    else:
        n_groups = min(8, dim)
        group_size = dim // n_groups
        groups = []
        group_labels = []
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size
            groups.append(np.linalg.norm(actions[:, start:end], axis=-1).tolist())
            group_labels.append(f"Group {g+1} (dim {start}-{end})")

    return {
        "timesteps": T,
        "action_dim": dim,
        "norms": norms,
        "groups": groups,
        "group_labels": group_labels,
        "task": task_name,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9110)
    parser.add_argument("--model-name", type=str, default="dreamdojo_2b")
    args = parser.parse_args()
    os.environ["DREAMDOJO_MODEL"] = args.model_name
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
