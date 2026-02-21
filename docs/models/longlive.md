# LongLive -- Architecture & Technical Reference

## Overview
Causal autoregressive extension of Wan 2.1 for long videos (up to 240s) at real-time (20.7 FPS on H100). Modifies bidirectional attention to causal with KV cache and Frame Sink.

Location: models/longlive/, Workers: workers/longlive_worker.py, workers/longlive_interactive_worker.py, Env: envs/af-longlive

## Architecture
Base: Wan 2.1-T2V-1.3B (frozen, dim=1536, heads=12, layers=30, 16-ch VAE)
CausalWanModel in wan/modules/causal_model.py: CausalWanSelfAttention replaces bidirectional. KV caching per block. Local attention window (12 frames). Frame Sink (3 frames) for long-range consistency.
LoRA adapter rank=256 on top of frozen base. Checkpoint: longlive_models/models/lora.pt. Training: 32 GPU-days.
Same Wan-VAE and UMT5-XXL.

Dims: 16 latent channels, 1536 hidden, 12 heads, 30 layers, 8x spatial, 4x temporal, 1560 tokens/frame

## Inputs & Outputs
Inputs: prompt(s) (can switch mid-video), num_frames (30 latent=~30s), seed (42), denoising_steps [1000,750,500,250], local_attn_size 12, sink_size 3
Outputs: Video [B,T,3,H,W] e.g. [1,120,3,480,832], FPS 16, max 240s
Internal: noise [B,T,16,60,104], KV cache [B,cache,12,128] per block, text [B,512,4096]

## Noise Schedule: Flow Matching (same as Wan 2.1)
Block-wise: 3 frames per block, 4 denoising steps [1000,750,500,250]
Same velocity prediction and sigma schedule

## Conditioning
Same text cross-attention as Wan 2.1. KV-Recache for smooth prompt switching mid-video.

## Inference Pipeline
1. Text encoding (same as Wan 2.1)
2. KV cache init for 30 blocks
3. Autoregressive: 3-frame blocks, 4 denoise steps each, local window + global sink, update KV cache, repeat
4. VAE decode

## Notable Features
1. Frame Sink for global context
2. KV-Recache for prompt switching
3. Streaming Long Tuning training
4. O(n) attention via local window + sink
5. 20.7 FPS real-time on H100
6. LoRA-only fine-tuning
7. Interactive chunk-by-chunk UI in arc_fabric
