"""Manages model worker subprocess lifecycle."""

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from .config import ModelSpec, PlatformConfig, MODEL_REGISTRY
from .gpu_manager import GPUAllocation, GPUManager

logger = logging.getLogger(__name__)

WORKER_STARTUP_TIMEOUT = 300  # 5 min for large model loading
WORKER_HEALTH_INTERVAL = 2


@dataclass
class WorkerProcess:
    model_name: str
    process: subprocess.Popen
    port: int
    gpu_indices: list[int]


class WorkerManager:
    """Spawns and manages model worker subprocesses in their conda envs."""

    def __init__(self, gpu_manager: GPUManager, config: PlatformConfig):
        self.gpu_manager = gpu_manager
        self.config = config
        self.workers: dict[str, WorkerProcess] = {}
        self._next_port = config.worker_base_port

    def _get_next_port(self) -> int:
        port = self._next_port
        self._next_port += 1
        return port

    def _resolve_conda_run(self, model_spec: ModelSpec) -> list[str]:
        return ["conda", "run", "--prefix", model_spec.conda_env]

    def start_worker(self, model_name: str) -> WorkerProcess:
        """Start a model worker. Allocates GPU, spawns subprocess."""
        if model_name in self.workers:
            logger.info(f"Worker {model_name} already running")
            return self.workers[model_name]

        model_spec = MODEL_REGISTRY.get(model_name)
        if model_spec is None:
            raise ValueError(f"Unknown model: {model_name}")

        if not self.gpu_manager.can_allocate(model_spec):
            evicted = self.gpu_manager.evict_lru()
            if evicted:
                self.stop_worker(evicted.model_name)
            if not self.gpu_manager.can_allocate(model_spec):
                raise RuntimeError(f"Cannot allocate GPU for {model_name}")

        port = self._get_next_port()
        alloc = self.gpu_manager.allocate(model_spec, port)

        cuda_devices = ",".join(str(i) for i in alloc.gpu_indices)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

        worker_script = str(
            Path(__file__).parent.parent / "workers" / f"{model_name.split('_')[0]}_worker.py"
        )

        # Map model names to their actual worker scripts
        worker_map = {
            "longlive": "longlive_worker.py",
            "ltx_2b": "ltx_worker.py",
            "ltx_13b": "ltx_worker.py",
            "wan21_1_3b": "wan21_worker.py",
            "wan21_14b": "wan21_worker.py",
        }
        worker_script = str(
            Path(__file__).parent.parent / "workers" / worker_map[model_name]
        )

        cmd = [
            *self._resolve_conda_run(model_spec),
            "python", worker_script,
            "--port", str(port),
            "--model-name", model_name,
        ]

        logger.info(f"Starting worker: {' '.join(cmd)}")
        logger.info(f"CUDA_VISIBLE_DEVICES={cuda_devices}")

        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=model_spec.working_dir or None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        alloc.worker_pid = proc.pid
        worker = WorkerProcess(
            model_name=model_name,
            process=proc,
            port=port,
            gpu_indices=alloc.gpu_indices,
        )
        self.workers[model_name] = worker

        logger.info(f"Worker {model_name} started (pid={proc.pid}, port={port})")
        return worker

    def wait_for_ready(self, model_name: str, timeout: float = WORKER_STARTUP_TIMEOUT) -> bool:
        """Wait for worker to respond to health checks."""
        worker = self.workers.get(model_name)
        if not worker:
            return False

        url = f"http://localhost:{worker.port}/health"
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    logger.info(f"Worker {model_name} is ready on port {worker.port}")
                    return True
            except requests.ConnectionError:
                pass

            if worker.process.poll() is not None:
                logger.error(f"Worker {model_name} exited with code {worker.process.returncode}")
                return False

            time.sleep(WORKER_HEALTH_INTERVAL)

        logger.error(f"Worker {model_name} did not become ready within {timeout}s")
        return False

    def stop_worker(self, model_name: str) -> None:
        """Stop a worker process and release its GPU."""
        worker = self.workers.pop(model_name, None)
        if worker is None:
            return

        logger.info(f"Stopping worker {model_name} (pid={worker.process.pid})")

        try:
            worker.process.terminate()
            worker.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning(f"Worker {model_name} didn't terminate, killing")
            worker.process.kill()
            worker.process.wait(timeout=10)

        self.gpu_manager.release(model_name)

    def stop_all(self) -> None:
        for name in list(self.workers.keys()):
            self.stop_worker(name)

    def get_worker_url(self, model_name: str) -> Optional[str]:
        worker = self.workers.get(model_name)
        if worker:
            return f"http://localhost:{worker.port}"
        return None

    def status(self) -> dict:
        return {
            name: {
                "pid": w.process.pid,
                "port": w.port,
                "gpu_indices": w.gpu_indices,
                "running": w.process.poll() is None,
            }
            for name, w in self.workers.items()
        }
