"""
BaseSubprocessServer — unified inference container protocol.

Solves: 5 inference containers each had independent server.py with
inconsistent error handling, no GPU health check, no output mtime
validation. CMB GPU runtime crash was reported as "ok" (AP-025).

Each model server extends this base and defines:
  - pipeline_name: str              (e.g., "aneurysm")
  - build_cmd(req) -> list          (subprocess command)
  - expected_outputs(req) -> list   (expected output file paths)
  - get_env(req) -> dict            (optional: extra env vars)
  - get_cwd() -> str|None           (optional: working directory)

Provides:
  - /health — nvidia-smi GPU check (Docker healthcheck target)
  - /predict — subprocess execution + output mtime validation
  - Fake pynvml module (bypass in-script GPU memory gates)
  - Structured logging
"""

import logging
import os
import pathlib
import subprocess
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("inference")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Fake pynvml — shared across all servers
# The scheduler manages GPU allocation; in-script pynvml checks are redundant
# and harmful (cause silent failures when GPU memory gate blocks).
# ---------------------------------------------------------------------------
_FAKE_MODULES_DIR = "/tmp/fake_pynvml_modules"
os.makedirs(_FAKE_MODULES_DIR, exist_ok=True)
with open(f"{_FAKE_MODULES_DIR}/pynvml.py", "w") as _f:
    _f.write(
        "class _FakeHandle: pass\n"
        "class _FakeMemInfo:\n"
        "    used = 0\n"
        "    total = 100\n"
        "def nvmlInit(): pass\n"
        "def nvmlDeviceGetHandleByIndex(n): return _FakeHandle()\n"
        "def nvmlDeviceGetMemoryInfo(h): return _FakeMemInfo()\n"
        "class NVMLError_Unknown(Exception): pass\n"
        "class NVMLError(Exception): pass\n"
    )

FAKE_MODULES_DIR = _FAKE_MODULES_DIR  # export for subclasses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_env_file(path: str):
    """Load key=value env file, setdefault into os.environ."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def check_gpu_health() -> dict:
    """Run nvidia-smi to check GPU availability and memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {"gpu_ok": False, "error": result.stderr.strip()}
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "name": parts[0],
                    "memory_used": parts[1],
                    "memory_total": parts[2],
                })
        return {"gpu_ok": True, "gpus": gpus}
    except Exception as e:
        return {"gpu_ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Request / Response (shared schema for all subprocess servers)
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    study_id: str
    input_paths: List[str]
    dicom_dirs: List[str] = []
    output_folder: str
    gpu_id: int = 0
    input_json: str = ""
    phase: str = ""  # CP6c: "", "preprocess", "inference", "postprocess"
    # CP1c (AP-085 step-multi-group-isolation): per-task group filter; None=Watcher legacy
    # (server.py fall back to module-level GROUP_ID env), int=retrigger 帶 --group-id 帶過來
    group_id: Optional[int] = None


class PredictResponse(BaseModel):
    status: str
    output_paths: List[str] = []
    elapsed_time: float = 0.0
    error_msg: str = ""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseSubprocessServer(ABC):
    """Base class for subprocess-based inference servers.

    Subclass must implement:
      - pipeline_name (property or class attribute)
      - build_cmd(req) -> list
      - expected_outputs(req) -> list[pathlib.Path]

    Optional overrides:
      - get_env(req) -> dict
      - get_cwd() -> str | None
    """

    @property
    @abstractmethod
    def pipeline_name(self) -> str:
        """Model name, e.g. 'aneurysm', 'cmb'."""

    @abstractmethod
    def build_cmd(self, req: PredictRequest) -> list:
        """Build the subprocess command list."""

    @abstractmethod
    def expected_outputs(self, req: PredictRequest) -> List[pathlib.Path]:
        """Return list of expected output files to verify after inference."""

    def get_env(self, req: PredictRequest) -> dict:
        """Environment variables for subprocess. Override for model-specific vars."""
        return {
            **os.environ,
            "PYTHONPATH": f"{FAKE_MODULES_DIR}:{os.environ.get('PYTHONPATH', '')}",
            "CUDA_VISIBLE_DEVICES": str(req.gpu_id),
        }

    def get_cwd(self) -> Optional[str]:
        """Working directory for subprocess. Override if needed (e.g. CMB)."""
        return None

    def create_app(self) -> FastAPI:
        """Create FastAPI app with /health and /predict endpoints."""
        app = FastAPI(title=f"{self.pipeline_name.title()} Inference Server")
        server = self

        @app.get("/health")
        def health():
            gpu_info = check_gpu_health()
            return {
                "status": "ok" if gpu_info["gpu_ok"] else "degraded",
                "pipeline": server.pipeline_name,
                "gpu": gpu_info,
            }

        @app.post("/predict", response_model=PredictResponse)
        def predict(req: PredictRequest):
            start = time.time()

            cmd = server.build_cmd(req)
            env = server.get_env(req)
            cwd = server.get_cwd()

            logger.info(
                "[%s] predict study=%s gpu=%d inputs=%d",
                server.pipeline_name, req.study_id, req.gpu_id,
                len(req.input_paths),
            )

            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, cwd=cwd,
            )
            elapsed = time.time() - start

            # --- Check returncode ---
            if result.returncode != 0:
                err = (result.stderr or result.stdout)[-3000:]
                logger.error(
                    "[%s] FAILED study=%s returncode=%d elapsed=%.1fs",
                    server.pipeline_name, req.study_id,
                    result.returncode, elapsed,
                )
                return PredictResponse(
                    status="error",
                    elapsed_time=elapsed,
                    error_msg=err,
                )

            # --- Verify expected outputs exist AND are fresh ---
            for out_path in server.expected_outputs(req):
                if not out_path.exists():
                    tail = ((result.stderr or "") + (result.stdout or ""))[-2000:]
                    logger.error(
                        "[%s] SILENT FAILURE study=%s — %s not found",
                        server.pipeline_name, req.study_id, out_path,
                    )
                    return PredictResponse(
                        status="error",
                        elapsed_time=elapsed,
                        error_msg=(
                            f"Silent failure: pipeline exited 0 but "
                            f"{out_path} not found. log={tail}"
                        ),
                    )

                # mtime validation: output must be newer than request start
                mtime = out_path.stat().st_mtime
                if mtime < start:
                    logger.error(
                        "[%s] STALE OUTPUT study=%s — %s mtime=%.0f < start=%.0f",
                        server.pipeline_name, req.study_id,
                        out_path, mtime, start,
                    )
                    return PredictResponse(
                        status="error",
                        elapsed_time=elapsed,
                        error_msg=(
                            f"Stale output: {out_path} exists but "
                            f"mtime ({mtime:.0f}) < request start ({start:.0f}). "
                            f"Pipeline may have reused cached/stale file."
                        ),
                    )

            logger.info(
                "[%s] OK study=%s elapsed=%.1fs",
                server.pipeline_name, req.study_id, elapsed,
            )
            return PredictResponse(status="ok", elapsed_time=elapsed)

        # ── Startup: auto-register to platform ──
        @app.on_event("startup")
        async def _auto_register():
            """Container 啟動時自動 POST /sync/register，讓平台知道這個模型可用。

            設定方式：docker-compose environment 加 REGISTER_URL + REGISTER_CONFIG
            現有 4 model 已在 model_registry.yml，不需設定。
            未來新模型設定範例：
              REGISTER_URL=http://host:8012/api/v1/sync/register
              REGISTER_CONFIG={"name":"VesselAge","series_requirements":[["MRA_BRAIN"]],...}
            """
            register_url = os.environ.get("REGISTER_URL", "")
            if not register_url:
                return  # 現有模型不需要 auto-register
            import json as _json
            config_str = os.environ.get("REGISTER_CONFIG", "")
            if not config_str:
                logger.warning("[%s] REGISTER_URL set but REGISTER_CONFIG missing, skipping", server.pipeline_name)
                return
            try:
                import urllib.request
                req = urllib.request.Request(
                    register_url, data=config_str.encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                resp = urllib.request.urlopen(req, timeout=5)
                logger.info("[%s] auto-register → %s (%d)", server.pipeline_name, register_url, resp.status)
            except Exception as e:
                logger.warning("[%s] auto-register failed (non-blocking): %s", server.pipeline_name, e)

        return app
