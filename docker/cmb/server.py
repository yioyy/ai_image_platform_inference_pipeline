"""CMB Three-Layer Inference Server.

CP5c + CP6c: Three-step orchestration with phase routing:
  1. preprocess.py (CPU) — copy SWAN + T1 + directory setup
  2. inference.py (GPU) — SynthSeg + 2-stage detection (ONNX/TF backend)
  3. postprocess.py (CPU) — DICOM-SEG + deliver + followup

CMB-specific: output at output_folder/study_id/Pred_CMB.nii.gz (nested).
"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import time
from typing import List, Optional

from fastapi import FastAPI

from base_server import (
    PredictRequest,
    PredictResponse,
    check_gpu_health,
    load_env_file,
    FAKE_MODULES_DIR,
)

logger = logging.getLogger("cmb.server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load env files
try:
    from dotenv import load_dotenv as _load_dotenv
    _BRAIN_ENV = "/opt/shh_aiplatform-david/brain-parcellation/.env"
    if os.path.exists(_BRAIN_ENV):
        _load_dotenv(_BRAIN_ENV, override=False)
except Exception:
    pass

load_env_file("/opt/shh_aiplatform-chuan/pipeline/chuan/code/config/radax.env")

BRAIN_PARCELLATION_ROOT = "/opt/shh_aiplatform-david/brain-parcellation"


def _env(key: str, default: str) -> str:
    v = os.getenv(key, "").strip()
    return os.path.normpath(v) if v else os.path.normpath(default)


PATH_PROCESS = _env("PATH_PROCESS", "/data/4TB/ai_pipeline/chuan/process")
PATH_LOG = _env("PATH_LOG", "/data/4TB/ai_pipeline/chuan/log")


def _build_env(*, gpu_id: int | None = None) -> dict:
    env = {
        **os.environ,
        "PYTHONPATH": f"{FAKE_MODULES_DIR}:{BRAIN_PARCELLATION_ROOT}",
    }
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["GPU_N"] = str(gpu_id)
    if "XLA_FLAGS" not in env:
        env["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/conda/envs/tf_2_14"
    return env


def _run_step(
    name: str, cmd: list, env: dict, start_time: float, timeout: int = 1800, *, cwd: str | None = None,
) -> PredictResponse | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("[cmb] %s TIMEOUT after %ds", name, timeout)
        return PredictResponse(
            status="error", elapsed_time=time.time() - start_time,
            error_msg=f"{name} timeout after {timeout}s",
        )
    if result.returncode != 0:
        err = (result.stderr or result.stdout)[-3000:]
        logger.error("[cmb] %s FAILED (rc=%d): %s", name, result.returncode, err[-500:])
        return PredictResponse(
            status="error",
            elapsed_time=time.time() - start_time,
            error_msg=f"{name} failed (rc={result.returncode}): {err}",
        )
    logger.info("[cmb] %s done", name)
    return None


app = FastAPI(title="CMB Three-Layer Inference Server")


@app.get("/health")
def health():
    gpu_info = check_gpu_health()
    if not gpu_info["gpu_ok"]:
        # nvidia-container-toolkit cgroup denial (e.g. after systemd
        # daemon-reload on host) cannot recover in-process. Schedule clean
        # exit so docker `restart: unless-stopped` resurrects with a fresh
        # nvidia hook. 2s delay lets this response return to the caller.
        import threading
        logger.error("[autoheal] NVML degraded — scheduling container exit")
        threading.Timer(2.0, lambda: os._exit(1)).start()
    return {
        "status": "ok" if gpu_info["gpu_ok"] else "degraded",
        "pipeline": "cmb",
        "gpu": gpu_info,
    }


# ── Phase functions ──────────────────────────────────────────────────────────

def _run_preprocess(req: PredictRequest, start: float) -> PredictResponse:
    if len(req.input_paths) < 2:
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg=f"CMB needs 2 inputs (SWAN, T1), got {len(req.input_paths)}")

    swan_path = req.input_paths[0]
    t1_path = req.input_paths[1]
    process_dir = os.path.join(PATH_PROCESS, "Deep_CMB", req.study_id)
    base_env = _build_env()

    logger.info("[cmb] preprocess start")
    err = _run_step(
        "Preprocess",
        ["python", "/app/preprocess.py",
         "--study_id", req.study_id,
         "--swan_path", swan_path,
         "--t1_path", t1_path,
         "--process_dir", process_dir,
         "--output_dir", req.output_folder],
        base_env, start,
    )
    if err:
        return err
    return PredictResponse(status="ok", elapsed_time=time.time() - start)


def _run_inference(req: PredictRequest, start: float) -> PredictResponse:
    swan_path = req.input_paths[0]
    t1_path = req.input_paths[1]
    process_dir = os.path.join(PATH_PROCESS, "Deep_CMB", req.study_id)
    gpu_env = _build_env(gpu_id=req.gpu_id)

    logger.info("[cmb] inference start (GPU %d)", req.gpu_id)
    err = _run_step(
        "Inference",
        ["python", "/app/inference.py",
         "--study_id", req.study_id,
         "--swan_path", swan_path,
         "--t1_path", t1_path,
         "--output_dir", req.output_folder,
         "--process_dir", process_dir,
         "--log_dir", PATH_LOG,
         "--gpu_id", str(req.gpu_id)],
        gpu_env, start,
        cwd=BRAIN_PARCELLATION_ROOT,
    )
    if err:
        return err

    pred_path = pathlib.Path(PATH_PROCESS) / "Deep_CMB" / req.study_id / "Pred_CMB.nii.gz"
    if not pred_path.exists():
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg=f"Silent failure: {pred_path} not found")
    if pred_path.stat().st_mtime < start:
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg=f"Stale Pred_CMB.nii.gz mtime < request start")

    return PredictResponse(status="ok", elapsed_time=time.time() - start)


def _run_postprocess(req: PredictRequest, start: float) -> PredictResponse:
    dicom_dir = req.dicom_dirs[0] if req.dicom_dirs else ""
    logger.info("[cmb] postprocess dicom_dir='%s' dicom_dirs=%s", dicom_dir, req.dicom_dirs)
    process_dir = os.path.join(PATH_PROCESS, "Deep_CMB", req.study_id)
    base_env = _build_env()
    input_json = req.input_json if req.input_json and req.input_json not in ("", "[]") else ""

    logger.info("[cmb] postprocess start group_id=%s", req.group_id)
    # CP1c (AP-085): per-task group_id 由 task_detector_inference payload 帶進來；
    # 沒帶時 postprocess.py 會 fall back 到 env GROUP_ID_CMB / GROUP_ID（legacy）。
    cmd = ["python", "/app/postprocess.py",
           "--study_id", req.study_id,
           "--output_dir", req.output_folder,
           "--dicom_dir", dicom_dir,
           "--process_dir", process_dir,
           "--input_json", input_json]
    if req.group_id is not None:
        cmd.extend(["--group_id", str(req.group_id)])
    err = _run_step(
        "Postprocess", cmd, base_env, start,
        cwd=BRAIN_PARCELLATION_ROOT,
    )
    if err:
        return err
    return PredictResponse(status="ok", elapsed_time=time.time() - start)


# ── Main endpoint with phase routing ─────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()

    logger.info("[cmb] predict study=%s gpu=%d phase=%s",
                req.study_id, req.gpu_id, req.phase or "full")

    if req.phase == "preprocess":
        return _run_preprocess(req, start)
    elif req.phase == "inference":
        return _run_inference(req, start)
    elif req.phase == "postprocess":
        return _run_postprocess(req, start)

    # Full pipeline (backward compatible)
    resp = _run_preprocess(req, start)
    if resp.status != "ok":
        return resp

    resp = _run_inference(req, start)
    if resp.status != "ok":
        return resp

    resp = _run_postprocess(req, start)
    if resp.status != "ok":
        return resp

    elapsed = time.time() - start
    logger.info("[cmb] OK study=%s elapsed=%.1fs", req.study_id, elapsed)
    return PredictResponse(status="ok", elapsed_time=elapsed)


# ── Registration endpoint (FSL FLIRT, CPU only) ────────────────────────────

from pydantic import BaseModel as _BaseModel


class RegisterRequest(_BaseModel):
    current_synthseg5: str
    historical_synthseg5: str
    output_dir: str
    method: str = "fsl_rigid"


class RegisterResponse(_BaseModel):
    status: str
    elapsed_time: float = 0.0
    quality_metric: Optional[dict] = None
    error_msg: str = ""


class WarpRequest(_BaseModel):
    moving: str           # NIfTI to warp
    reference: str        # reference space
    affine_mat: str       # transform matrix
    output: str           # output path


class WarpResponse(_BaseModel):
    status: str
    output: str = ""
    error_msg: str = ""


@app.post("/warp", response_model=WarpResponse)
def warp(req: WarpRequest):
    """Apply existing transform to warp a NIfTI (CPU only)."""
    fsl_path = "/usr/local/fsl/bin/flirt"
    env = {**os.environ, "FSLDIR": "/usr/local/fsl", "FSLOUTPUTTYPE": "NIFTI_GZ"}
    cmd = [
        fsl_path, "-in", req.moving, "-ref", req.reference,
        "-out", req.output, "-applyxfm", "-init", req.affine_mat,
        "-interp", "nearestneighbour",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        if result.returncode != 0:
            return WarpResponse(status="error", error_msg=result.stderr.strip()[:500])
        return WarpResponse(status="ok", output=req.output)
    except Exception as e:
        return WarpResponse(status="error", error_msg=str(e)[:500])


@app.post("/register", response_model=RegisterResponse)
def register(req: RegisterRequest):
    """FSL FLIRT 對位（CPU only，不佔 GPU）。"""
    import sys
    sys.path.insert(0, BRAIN_PARCELLATION_ROOT)
    from code_ai.pipeline.registration_engine import register_pair, RegistrationRequest

    result = register_pair(RegistrationRequest(
        current_synthseg5=pathlib.Path(req.current_synthseg5),
        historical_synthseg5=pathlib.Path(req.historical_synthseg5),
        output_dir=pathlib.Path(req.output_dir),
        method=req.method,
    ))

    return RegisterResponse(
        status="ok" if result.success else "error",
        elapsed_time=result.elapsed_seconds,
        quality_metric=result.quality_metric,
        error_msg=result.error,
    )
