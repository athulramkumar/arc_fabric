# LTX-Video -- Architecture & Technical Reference

## Overview
Fast DiT family by Lightricks. Multi-scale pipelines, distilled 8-step models (no CFG), FP8 quantization. Supports T2V, I2V, V2V, video extension, multi-keyframe.

Location: models/ltx_video/, Worker: workers/ltx_worker.py, Env: envs/af-ltx

## Model Variants
- 2B Distilled: 8 steps, no CFG. weights/ltx_video/ltxv-2b-0.9.8-distilled/
- 2B Dev: 40 steps, CFG
- 13B Distilled: 8 steps, no CFG. weights/ltx_video/ltxv-13b-0.9.8-distilled/
- 13B Dev: 30+ steps, CFG. weights/ltx_video/ltxv-13b-0.9.8-dev/
- Spatial Upscaler: ltxv-spatial-upscaler-0.9.8.safetensors

## Architecture
VAE: AutoencoderKLWrapper in ltx_video/models/autoencoders/vae.py. Configurable latent channels (4 or 16), 8x spatial.
Text Encoder: PixArt T5 (PixArt-alpha/PixArt-XL-2-1024-MS)
Transformer: Transformer3DModel in ltx_video/models/transformers/transformer3d.py with BasicTransformerBlock. RoPE, LayerNorm/RMSNorm, AdaLN, GEGLU activation.

Resolution: 1216x704 or 704x480 at 30 FPS. Frames must be 8n+1 (97, 121, 257). Resolution must be divisible by 32.

## Inputs & Outputs
Inputs: prompt, height (480), width (704), num_frames (97, must be 8n+1), seed, conditioning_media (for I2V/V2V), conditioning_start_frames
Outputs: Video [B,T,C,H,W] RGB, FPS 24
Internal: latents [B,C,T,H_lat,W_lat], transformer [B,L,inner_dim], text [B,seq,text_dim]

## Noise Schedule: Rectified Flow
RFScheduler in ltx_video/schedulers/rf.py. 1000 training timesteps. Distilled: 8 steps no CFG. Full: 30-40 steps.
Multi-scale timesteps: first pass [1.0,0.9937,0.9875,...,0.7250], second pass [0.9094,0.7250,0.4219]
Same v-prediction update rule as Wan 2.1.

## Conditioning
Text: Cross-attention via PixArt T5. CFG for full models only.
Image/Video: Multi-keyframe support at different frame positions, per-item conditioning strength.

## Inference Pipeline
Standard: text encode -> (optional) encode conditioning -> pad -> noise -> denoise -> VAE decode -> crop
Multi-scale: first pass low-res -> spatial upscaler -> second pass full-res -> decode -> crop

## Notable Features
1. Multi-scale pipeline (low-res preview -> high-res)
2. Distilled 8-step no-CFG models
3. Skip Layer Strategy for speed
4. Spatial upscaler model
5. FP8 quantization
6. Up to 60s video (13B)
7. IC-LoRA control (depth, pose, canny)
8. Hybrid 13B+2B in arc_fabric
