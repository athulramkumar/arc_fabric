"""Arc Fabric orchestrator API server."""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..config import PlatformConfig, MODEL_REGISTRY, GPUInfo, OUTPUTS_DIR
from ..gpu_manager import GPUManager
from ..session_manager import SessionManager
from ..worker_manager import WorkerManager

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CreateSessionRequest(BaseModel):
    model_name: str


class GenerateRequest(BaseModel):
    session_id: str
    prompt: str
    num_frames: int = 97
    height: int = 480
    width: int = 704
    seed: int = 42


class GenerateResponse(BaseModel):
    session_id: str
    status: str
    output_path: Optional[str] = None
    generation_time_s: float = 0.0
    error: Optional[str] = None


def create_app(config: Optional[PlatformConfig] = None) -> FastAPI:
    if config is None:
        config = PlatformConfig()

    app = FastAPI(
        title="Arc Fabric",
        description="Stateful execution and control layer for generative video",
        version="0.1.0",
    )

    gpu_manager = GPUManager(config.gpus)
    session_manager = SessionManager()
    worker_manager = WorkerManager(gpu_manager, config)

    @app.get("/")
    async def root():
        return {
            "name": "Arc Fabric",
            "version": "0.1.0",
            "models": list(MODEL_REGISTRY.keys()),
        }

    @app.get("/models")
    async def list_models():
        return {
            name: {
                "gpu_memory_gb": spec.gpu_memory_gb,
                "loaded": name in worker_manager.workers,
            }
            for name, spec in MODEL_REGISTRY.items()
        }

    @app.get("/gpus")
    async def gpu_status():
        return gpu_manager.status()

    @app.get("/sessions")
    async def list_sessions():
        return session_manager.list_sessions()

    @app.post("/sessions")
    async def create_session(req: CreateSessionRequest):
        if req.model_name not in MODEL_REGISTRY:
            raise HTTPException(404, f"Unknown model: {req.model_name}")

        # Start worker if not running
        if req.model_name not in worker_manager.workers:
            try:
                worker_manager.start_worker(req.model_name)
                ready = worker_manager.wait_for_ready(req.model_name)
                if not ready:
                    raise HTTPException(503, f"Worker for {req.model_name} failed to start")
            except RuntimeError as e:
                raise HTTPException(503, str(e))

        session = session_manager.create_session(req.model_name)
        return {
            "session_id": session.session_id,
            "model_name": session.model_name,
            "status": session.status,
        }

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        session = session_manager.get_session(req.session_id)
        if session is None:
            raise HTTPException(404, f"Session {req.session_id} not found")

        worker_url = worker_manager.get_worker_url(session.model_name)
        if worker_url is None:
            raise HTTPException(503, f"No worker for model {session.model_name}")

        output_dir = str(OUTPUTS_DIR / req.session_id)
        os.makedirs(output_dir, exist_ok=True)

        try:
            resp = requests.post(
                f"{worker_url}/generate",
                json={
                    "prompt": req.prompt,
                    "num_frames": req.num_frames,
                    "height": req.height,
                    "width": req.width,
                    "seed": req.seed,
                    "session_id": req.session_id,
                    "output_dir": output_dir,
                },
                timeout=600,
            )
            result = resp.json()

            if result.get("output_path"):
                session.output_files.append(result["output_path"])

            return GenerateResponse(
                session_id=req.session_id,
                status=result.get("status", "unknown"),
                output_path=result.get("output_path"),
                generation_time_s=result.get("generation_time_s", 0),
                error=result.get("error"),
            )

        except requests.Timeout:
            return GenerateResponse(
                session_id=req.session_id,
                status="timeout",
                error="Generation timed out",
            )
        except Exception as e:
            return GenerateResponse(
                session_id=req.session_id,
                status="error",
                error=str(e),
            )

    @app.delete("/sessions/{session_id}")
    async def end_session(session_id: str):
        session = session_manager.end_session(session_id)
        if session is None:
            raise HTTPException(404, f"Session {session_id} not found")
        return {"session_id": session_id, "status": "ended"}

    @app.get("/workers")
    async def worker_status():
        return worker_manager.status()

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Shutting down workers...")
        worker_manager.stop_all()

    return app


def main():
    config = PlatformConfig()
    app = create_app(config)
    uvicorn.run(app, host="0.0.0.0", port=config.orchestrator_port, log_level="info")


if __name__ == "__main__":
    main()
