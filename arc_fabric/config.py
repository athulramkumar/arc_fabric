"""Platform configuration for Arc Fabric."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


ROOT_DIR = Path("/workspace/arc_fabric")
WEIGHTS_DIR = ROOT_DIR / "weights"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"


@dataclass
class ModelSpec:
    name: str
    conda_env: str
    worker_module: str
    gpu_memory_gb: float
    weight_paths: dict = field(default_factory=dict)
    working_dir: str = ""
    multi_gpu: bool = False


ENVS_DIR = ROOT_DIR / "envs"

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "longlive": ModelSpec(
        name="longlive",
        conda_env=str(ENVS_DIR / "af-longlive"),
        worker_module="workers.longlive_worker",
        gpu_memory_gb=25.0,
        weight_paths={
            "wan_base": str(WEIGHTS_DIR / "longlive" / "Wan2.1-T2V-1.3B"),
            "longlive": str(WEIGHTS_DIR / "longlive" / "LongLive"),
        },
        working_dir=str(MODELS_DIR / "longlive"),
    ),
    "ltx_2b": ModelSpec(
        name="ltx_2b",
        conda_env=str(ENVS_DIR / "af-ltx"),
        worker_module="workers.ltx_worker",
        gpu_memory_gb=26.0,
        weight_paths={
            "checkpoint": str(WEIGHTS_DIR / "ltx_video" / "ltxv-2b-0.9.8-distilled"),
        },
        working_dir=str(MODELS_DIR / "ltx_video"),
    ),
    "ltx_13b": ModelSpec(
        name="ltx_13b",
        conda_env=str(ENVS_DIR / "af-ltx"),
        worker_module="workers.ltx_worker",
        gpu_memory_gb=50.0,
        weight_paths={
            "checkpoint": str(WEIGHTS_DIR / "ltx_video" / "ltxv-13b-0.9.8-distilled"),
        },
        working_dir=str(MODELS_DIR / "ltx_video"),
    ),
    "wan21_1_3b": ModelSpec(
        name="wan21_1_3b",
        conda_env=str(ENVS_DIR / "af-wan21"),
        worker_module="workers.wan21_worker",
        gpu_memory_gb=8.0,
        weight_paths={
            "checkpoint": str(WEIGHTS_DIR / "wan21" / "Wan2.1-T2V-1.3B"),
        },
        working_dir=str(MODELS_DIR / "wan21"),
    ),
    "wan21_14b": ModelSpec(
        name="wan21_14b",
        conda_env=str(ENVS_DIR / "af-wan21"),
        worker_module="workers.wan21_worker",
        gpu_memory_gb=35.0,
        weight_paths={
            "checkpoint": str(WEIGHTS_DIR / "wan21" / "Wan2.1-T2V-14B"),
        },
        working_dir=str(MODELS_DIR / "wan21"),
    ),
}


@dataclass
class GPUInfo:
    index: int
    total_memory_gb: float = 80.0


@dataclass
class PlatformConfig:
    gpus: list[GPUInfo] = field(default_factory=lambda: [
        GPUInfo(index=0),
        GPUInfo(index=1),
    ])
    worker_base_port: int = 9100
    orchestrator_port: int = 8000
    outputs_dir: str = str(OUTPUTS_DIR)

    @property
    def num_gpus(self) -> int:
        return len(self.gpus)
