# Wan 2.1 -- Architecture & Technical Reference

## Overview
Wan 2.1 is a family of Diffusion Transformer (DiT) models for video generation built on Flow Matching. Developed by Alibaba, it supports text-to-video (T2V), image-to-video (I2V), first-last-frame-to-video (FLF2V), and video editing (VACE).

Location: models/wan21/, Worker: workers/wan21_worker.py, Env: envs/af-wan21

## Model Variants
- 1.3B: dim=1536, ffn=8960, heads=12, layers=30, channels=16. Weights: weights/wan21/Wan2.1-T2V-1.3B
- 14B: dim=5120, ffn=13824, heads=40, layers=40, channels=16. Weights: weights/wan21/Wan2.1-T2V-14B

## Architecture
### VAE: Wan-VAE (3D Causal)
Class WanVAE in wan/modules/vae.py. 16 latent channels, 8x spatial compression (stride 4,8,8), 4x temporal. Supports unlimited-length 1080P video. Example: 480x832 @ 81 frames -> latent [B, ~20, 16, 60, 104]

### Text Encoder: UMT5-XXL
google/umt5-xxl (~4.7B params, encoder-only). Max 512 tokens, output dim 4096. Location: wan/modules/t5.py

### Transformer: WanModel (DiT)
Class WanModel in wan/modules/model.py. Patch size (1,2,2). 3D RoPE positional embedding. RMSNorm for Q/K. Sinusoidal time embedding -> MLP -> 6 modulation params. Cross-attention: WanT2VCrossAttention (T2V), WanI2VCrossAttention (I2V with CLIP). SiLU activation.

### Image Encoder (I2V): CLIP Vision Transformer in wan/modules/clip.py

## Inputs & Outputs
Inputs: prompt (str, max 512 tokens), height (480), width (832), num_frames (81/33), seed (42), guidance_scale (5.0/6.0), steps (50/40)
Outputs: Video [C,N,H,W] RGB, FPS 16
Internal: text_emb [B,512,4096], latent [B,T,16,H/8,W/8], transformer [B,L,dim]

## Noise Schedule: Flow Matching (Rectified Flow)
Scheduler: FlowUniPCMultistepScheduler or FlowDPMSolverMultistepScheduler. 1000 training timesteps. flow_prediction type. Shift: 5.0 (720P) / 3.0 (480P). Formula: sigma = (shift*sigma)/(1+(shift-1)*sigma). Steps: 50 (T2V), 40 (I2V).
Update rule: v = model(x_t, t, text_emb); x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * v

## Conditioning
Text: UMT5-XXL -> cross-attention in each block, CFG: output = uncond + scale*(cond-uncond)
Image (I2V): CLIP ViT features with separate K/V projections

## Inference Pipeline
1. Text encoding -> UMT5-XXL -> [B,512,4096]
2. Noise sampling z~N(0,1) shape [B,T,16,H/8,W/8]
3. Patch embedding Conv3D(1,2,2) -> [B,L,dim]
4. Denoising loop (50 steps): 3D RoPE, self-attn + cross-attn, predict velocity, update via solver
5. VAE decode -> pixels

## Notable Features
1. 3D causal VAE unlimited-length 1080P
2. Rectified Flow (simpler ODE, fewer steps)
3. 3D RoPE separate per dimension
4. Optional Qwen prompt enhancement
5. FSDP + xDiT USP multi-GPU
6. Visual text generation
7. Hybrid 14B+1.3B in arc_fabric
