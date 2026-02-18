"""CLIP-based automated video quality testing.

Validates that generated videos are semantically meaningful by computing
CLIP similarity between video frames and the text prompt.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CLIP_SCORE_THRESHOLD = 0.20


def extract_frames(video_path: str, num_frames: int = 8) -> list:
    """Extract evenly-spaced frames from a video file."""
    import imageio.v3 as iio

    frames = []
    try:
        all_frames = list(iio.imread(video_path, plugin="pyav"))
    except Exception:
        all_frames = list(iio.imiter(video_path))

    if not all_frames:
        raise ValueError(f"No frames found in {video_path}")

    total = len(all_frames)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)

    for idx in indices:
        frame = all_frames[idx]
        if isinstance(frame, np.ndarray):
            frames.append(frame)

    return frames


def compute_clip_score(video_path: str, prompt: str, num_frames: int = 8) -> dict:
    """Compute CLIP similarity between video frames and text prompt."""
    import open_clip
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    frames = extract_frames(video_path, num_frames)
    logger.info(f"Extracted {len(frames)} frames from {video_path}")

    images = []
    for frame in frames:
        img = Image.fromarray(frame)
        images.append(preprocess(img))

    image_input = torch.stack(images).to(device)
    text_input = tokenizer([prompt]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarities = (image_features @ text_features.T).squeeze(-1)

    scores = similarities.cpu().numpy()
    mean_score = float(np.mean(scores))
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))

    del model, image_input, text_input
    torch.cuda.empty_cache()

    return {
        "mean_clip_score": mean_score,
        "min_clip_score": min_score,
        "max_clip_score": max_score,
        "per_frame_scores": scores.tolist(),
        "num_frames_sampled": len(frames),
        "prompt": prompt,
        "video_path": video_path,
        "passed": mean_score >= CLIP_SCORE_THRESHOLD,
    }


def test_video(video_path: str, prompt: str, threshold: float = CLIP_SCORE_THRESHOLD) -> bool:
    """Run CLIP quality test on a video. Returns True if passed."""
    result = compute_clip_score(video_path, prompt)

    logger.info(f"CLIP Score Results for: {Path(video_path).name}")
    logger.info(f"  Prompt: {prompt[:80]}...")
    logger.info(f"  Mean CLIP score: {result['mean_clip_score']:.4f}")
    logger.info(f"  Min/Max: {result['min_clip_score']:.4f} / {result['max_clip_score']:.4f}")
    logger.info(f"  Threshold: {threshold}")
    logger.info(f"  PASSED: {result['passed']}")

    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP-based video quality test")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--prompt", required=True, help="Text prompt used for generation")
    parser.add_argument("--threshold", type=float, default=CLIP_SCORE_THRESHOLD)
    args = parser.parse_args()

    passed = test_video(args.video, args.prompt, args.threshold)
    sys.exit(0 if passed else 1)
