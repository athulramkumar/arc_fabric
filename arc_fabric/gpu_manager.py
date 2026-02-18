"""GPU allocation manager with LRU eviction."""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import GPUInfo, ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class GPUAllocation:
    model_name: str
    gpu_indices: list[int]
    worker_port: int
    worker_pid: Optional[int] = None
    loaded_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class GPUManager:
    """Manages GPU allocation with LRU eviction for model workers."""

    def __init__(self, gpus: list[GPUInfo]):
        self.gpus = {gpu.index: gpu for gpu in gpus}
        self.allocations: dict[str, GPUAllocation] = {}
        self._gpu_to_model: dict[int, str] = {}

    @property
    def free_gpus(self) -> list[int]:
        return [i for i in self.gpus if i not in self._gpu_to_model]

    @property
    def allocated_models(self) -> list[str]:
        return list(self.allocations.keys())

    def get_allocation(self, model_name: str) -> Optional[GPUAllocation]:
        alloc = self.allocations.get(model_name)
        if alloc:
            alloc.last_accessed = time.time()
        return alloc

    def can_allocate(self, model_spec: ModelSpec) -> bool:
        needed = 1 if not model_spec.multi_gpu else 2
        return len(self.free_gpus) >= needed

    def allocate(self, model_spec: ModelSpec, worker_port: int) -> GPUAllocation:
        """Allocate GPU(s) for a model. Raises if not enough free GPUs."""
        needed = 1 if not model_spec.multi_gpu else 2
        free = self.free_gpus

        if len(free) < needed:
            raise RuntimeError(
                f"Not enough free GPUs for {model_spec.name}: "
                f"need {needed}, have {len(free)}. "
                f"Call evict_lru() first."
            )

        gpu_indices = free[:needed]
        alloc = GPUAllocation(
            model_name=model_spec.name,
            gpu_indices=gpu_indices,
            worker_port=worker_port,
        )
        self.allocations[model_spec.name] = alloc
        for idx in gpu_indices:
            self._gpu_to_model[idx] = model_spec.name

        logger.info(
            f"Allocated GPU {gpu_indices} for {model_spec.name} "
            f"on port {worker_port}"
        )
        return alloc

    def release(self, model_name: str) -> Optional[GPUAllocation]:
        """Release GPU allocation for a model."""
        alloc = self.allocations.pop(model_name, None)
        if alloc:
            for idx in alloc.gpu_indices:
                self._gpu_to_model.pop(idx, None)
            logger.info(f"Released GPU {alloc.gpu_indices} from {model_name}")
        return alloc

    def get_lru_model(self) -> Optional[str]:
        """Get the least-recently-used model name."""
        if not self.allocations:
            return None
        return min(
            self.allocations,
            key=lambda m: self.allocations[m].last_accessed,
        )

    def evict_lru(self) -> Optional[GPUAllocation]:
        """Evict the least-recently-used model. Returns the evicted allocation."""
        lru = self.get_lru_model()
        if lru is None:
            return None
        logger.info(f"Evicting LRU model: {lru}")
        return self.release(lru)

    def status(self) -> dict:
        return {
            "total_gpus": len(self.gpus),
            "free_gpus": self.free_gpus,
            "allocations": {
                name: {
                    "gpu_indices": alloc.gpu_indices,
                    "port": alloc.worker_port,
                    "pid": alloc.worker_pid,
                    "loaded_at": alloc.loaded_at,
                    "last_accessed": alloc.last_accessed,
                }
                for name, alloc in self.allocations.items()
            },
        }
