# DreamDojo -- Architecture & Technical Reference

## Overview
Action-conditioned Video2World model by NVIDIA GEAR Team, built on Cosmos Predict2 (based on Wan 2.1). Predicts future video from initial frame + robot actions. NOT text-to-video -- it is a world model for robotics.

Location: models/dreamdojo/, Worker: workers/dreamdojo_worker.py, Env: envs/af-dreamdojo
Detailed docs: models/dreamdojo/docs/ (ONBOARDING.md, ARCHITECTURE.md, claude.md)

## Model Variants
- 2B GR-1: channels=1536, heads=12, blocks=30. Checkpoint: models/dreamdojo/checkpoints/2B_GR1_post-train/iter_000050000/model_ema_bf16.pt
- 14B GR-1: channels=5120, heads=40, blocks=40. Checkpoint: models/dreamdojo/checkpoints/14B_GR1_post-train/iter_000050000/model_ema_bf16.pt. Requires LAM: checkpoints/DreamDojo/LAM_400k.ckpt

## Architecture
VAE: Same Wan 2.1 VAE. 16 latent channels, 8x spatial, 4x temporal. Requires gated repo workaround for Wan2.1_VAE.pth.
DiT: MinimalV1LVGDiT. 2B: 1536/12/30. 14B: 5120/40/40. Same as Wan 2.1 but with action conditioning.
Action Embedder: 7-dim actions (dx,dy,dz,droll,dpitch,dyaw,gripper) scaled by 20.0, injected via cross-attention/AdaLN.
Text Encoder: T5-based (minimal use -- primarily action-conditioned).
LAM (14B only): LAM_400k.ckpt, processes lam_video at 2x temporal resolution.

Resolution: 480x640. Chunk size: 12 frames. Action dim: 7.

## Inputs & Outputs
Inputs: initial frame [1,C,H,W] uint8, actions [T-1,7] float, lam_video [T*2,...], prompt (usually ""), guidance 0.0
Outputs: predicted video [B,C,T,H,W] range [-1,1]
Internal: frame -> VAE -> [1,16,1,60,80], noise [1,16,T_lat,60,80] -> denoise with action cond -> VAE decode

## Noise Schedule: Flow Matching (Rectified Flow)
Same as Wan 2.1. Steps: 35. Guidance: 0.0. Same v-prediction update.

## Conditioning
Primary: robot actions (7D per timestep) -- relative position, rotation, gripper. Scaled by 20.0.
Secondary: text (usually empty string).

## Inference Pipeline
1. Load dataset sample (video, actions, lam_video)
2. Prepare initial frame
3. Chunk-wise autoregressive (12 actions per chunk):
   - Prepare video input (initial frame + zeros)
   - generate_vid2world() with actions + lam_video
   - Extract last frame for next chunk
4. Concatenate chunks
5. Compute metrics (PSNR, SSIM, LPIPS)

## Eval Results
2B GR-1: PSNR 25.284, SSIM 0.839, LPIPS 0.125
14B GR-1: PSNR 24.746, SSIM 0.855, LPIPS 0.106

## Dataset
nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1 on HuggingFace. 10 eval tasks under In-lab_Eval/. LeRobot-style format.

## Integration Notes
NOT T2V -- requires action sequences and initial frames from robot dataset. arc_fabric provides: dataset sample dropdown, synced GT vs pred playback, action chart, quality metrics.

## Notable Features
1. Physics-grounded world model
2. 12-frame chunk autoregressive
3. LAM conditioning for 14B
4. Same latent space as Wan 2.1 (compatible)
5. Multi-embodiment (GR-1, G1, AgibBot, EgoDex)
6. Gated repo workarounds needed
