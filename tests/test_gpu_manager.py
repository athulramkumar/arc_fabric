"""Tests for GPU manager."""

import time
import sys
sys.path.insert(0, "/workspace/arc_fabric")

from arc_fabric.config import GPUInfo, ModelSpec
from arc_fabric.gpu_manager import GPUManager


def test_basic_allocation():
    gm = GPUManager([GPUInfo(0), GPUInfo(1)])
    assert gm.free_gpus == [0, 1]

    spec = ModelSpec(name="test_model", conda_env="test", worker_module="test", gpu_memory_gb=30)
    alloc = gm.allocate(spec, worker_port=9100)

    assert alloc.model_name == "test_model"
    assert len(alloc.gpu_indices) == 1
    assert len(gm.free_gpus) == 1
    print("PASS: basic_allocation")


def test_lru_eviction():
    gm = GPUManager([GPUInfo(0), GPUInfo(1)])

    spec_a = ModelSpec(name="model_a", conda_env="a", worker_module="a", gpu_memory_gb=30)
    spec_b = ModelSpec(name="model_b", conda_env="b", worker_module="b", gpu_memory_gb=30)
    spec_c = ModelSpec(name="model_c", conda_env="c", worker_module="c", gpu_memory_gb=30)

    gm.allocate(spec_a, 9100)
    time.sleep(0.01)
    gm.allocate(spec_b, 9101)

    assert len(gm.free_gpus) == 0

    # model_a was loaded first, should be LRU
    lru = gm.get_lru_model()
    assert lru == "model_a"

    # Access model_a to make model_b the LRU
    gm.get_allocation("model_a")
    time.sleep(0.01)
    lru = gm.get_lru_model()
    assert lru == "model_b"

    # Evict LRU
    evicted = gm.evict_lru()
    assert evicted.model_name == "model_b"
    assert len(gm.free_gpus) == 1

    # Now we can allocate model_c
    gm.allocate(spec_c, 9102)
    assert len(gm.free_gpus) == 0
    print("PASS: lru_eviction")


def test_multi_gpu():
    gm = GPUManager([GPUInfo(0), GPUInfo(1)])
    spec = ModelSpec(name="big_model", conda_env="big", worker_module="big", gpu_memory_gb=60, multi_gpu=True)

    alloc = gm.allocate(spec, 9100)
    assert len(alloc.gpu_indices) == 2
    assert gm.free_gpus == []
    print("PASS: multi_gpu")


def test_status():
    gm = GPUManager([GPUInfo(0), GPUInfo(1)])
    spec = ModelSpec(name="test", conda_env="t", worker_module="t", gpu_memory_gb=30)
    gm.allocate(spec, 9100)

    status = gm.status()
    assert status["total_gpus"] == 2
    assert len(status["free_gpus"]) == 1
    assert "test" in status["allocations"]
    print("PASS: status")


if __name__ == "__main__":
    test_basic_allocation()
    test_lru_eviction()
    test_multi_gpu()
    test_status()
    print("\nAll GPU manager tests passed!")
