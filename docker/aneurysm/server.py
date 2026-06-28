"""Aneurysm Three-Layer Inference Server.

CP5b + CP6b + CP6c: Three-step orchestration with phase routing:
  1. preprocess.py (CPU) — copy MRA_BRAIN + directory setup
  2. inference.py (GPU) — 3 nnUNet stages + Vessel 16-label + reslice + MIP (CP6b)
  3. postprocess.py (CPU) — analysis + deliver (no GPU needed since CP6b)
"""

from __future__ import annotations

import logging
import os
import pathlib
import signal
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

logger = logging.getLogger("aneurysm.server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# NVIDIA container runs as root — ensure all files are group-writable
# so worker (uid=1003, gid=1001) can access them
os.umask(0o002)

load_env_file("/opt/shh_aiplatform-chuan/pipeline/chuan/code/config/radax.env")


def _env(key: str, default: str) -> str:
    v = os.getenv(key, "").strip()
    return os.path.normpath(v) if v else os.path.normpath(default)


PATH_PROCESS = _env("PATH_PROCESS", "/data/4TB/ai_pipeline/chuan/process")
PATH_BRAIN_MODEL = _env(
    "RADX_BRAIN_MODEL",
    "/opt/shh_aiplatform-chuan/pipeline/chuan/code/nnUNet/nnUNet_results"
    "/Dataset134_DeepMRABrain/nnUNetTrainer__nnUNetPlans__3d_fullres",
)
PATH_VESSEL_MODEL = _env(
    "RADX_VESSEL_MODEL",
    "/opt/shh_aiplatform-chuan/pipeline/chuan/code/nnUNet/nnUNet_results"
    "/Dataset135_DeepMRAVessel/nnUNetTrainer__nnUNetPlans__3d_fullres",
)
PATH_ANEURYSM_MODEL = _env(
    "RADX_ANEURYSM_MODEL",
    "/opt/shh_aiplatform-chuan/pipeline/chuan/code/nnUNet/nnUNet_results"
    "/Dataset080_DeepAneurysm/nnUNetTrainer__nnUNetPlans__3d_fullres",
)
PATH_CODE = _env("RADX_CODE_ROOT", "/opt/shh_aiplatform-david/brain-parcellation")
PATH_JSON = _env("RADX_JSON_ROOT", "/data/4TB/ai_pipeline/chuan/json")
GROUP_ID = int(os.getenv("RADX_ANEURYSM_GROUP_ID", "56") or "56")


def _build_env(*, gpu_id: int | None = None) -> dict:
    env = {
        **os.environ,
        "PYTHONPATH": f"{FAKE_MODULES_DIR}:{os.environ.get('PYTHONPATH', '')}",
    }
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def _kill_process_group(process: subprocess.Popen, name: str) -> None:
    """AP-098 fix: timeout 時殺整 process group (含 fork 出來的孫子 workers)。

    `subprocess.run(timeout=...)` 內部 process.kill() 只殺 direct child,
    但 inference.py 內部 batchgenerators MTA 會 fork 出 worker subprocess,
    這些 worker 變孤兒 (主 Python process 卡 fork-CUDA deadlock 死不掉)。
    用 setsid + killpg 確保整族 (process group leader + descendants) 一起殺。
    """
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        # Process 已自行結束
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
            return  # 10s 內 graceful shutdown OK
        except subprocess.TimeoutExpired:
            logger.error("[aneurysm] %s SIGTERM 10s 後仍未死,送 SIGKILL", name)
        os.killpg(pgid, signal.SIGKILL)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error("[aneurysm] %s SIGKILL 5s 後 wait timeout (kernel-level stuck?)", name)
    except (ProcessLookupError, PermissionError) as e:
        logger.warning("[aneurysm] killpg %s 失敗: %s (process 可能已結束)", name, e)


def _run_step(
    name: str, cmd: list, env: dict, start_time: float, timeout: int = 1800
) -> PredictResponse | None:
    """Run subprocess with timeout + process group kill on timeout (AP-098 fix).

    改寫自 subprocess.run(timeout=) — 因為原寫法在 timeout 時只 kill direct child,
    inference.py 內部 fork 出來的 worker 變孤兒 → 18 hr 殭屍累積。
    新寫法:start_new_session=True 建獨立 process group + timeout 觸發後 killpg 整族。
    """
    # start_new_session=True → 子 process setsid() 成新 session/process group leader,
    # 孫子 forked workers 都會繼承同個 pgid,timeout 時 killpg(pgid) 一次殺乾淨。
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, start_new_session=True,
        )
    except OSError as e:
        logger.error("[aneurysm] %s Popen 啟動失敗: %s", name, e)
        return PredictResponse(
            status="error", elapsed_time=time.time() - start_time,
            error_msg=f"{name} popen failed: {e}",
        )

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("[aneurysm] %s TIMEOUT after %ds — 殺整 process group", name, timeout)
        _kill_process_group(process, name)
        # Drain pipes after kill (避免 leaked file descriptors)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return PredictResponse(
            status="error", elapsed_time=time.time() - start_time,
            error_msg=f"{name} timeout after {timeout}s (process group killed)",
        )

    if process.returncode != 0:
        err = (stderr or stdout)[-3000:]
        logger.error("[aneurysm] %s FAILED (rc=%d): %s", name, process.returncode, err[-500:])
        return PredictResponse(
            status="error",
            elapsed_time=time.time() - start_time,
            error_msg=f"{name} failed (rc={process.returncode}): {err}",
        )
    logger.info("[aneurysm] %s done", name)
    return None


app = FastAPI(title="Aneurysm Three-Layer Inference Server")


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
        "pipeline": "aneurysm",
        "gpu": gpu_info,
    }


# ── Phase functions ──────────────────────────────────────────────────────────

def _run_preprocess(req: PredictRequest, start: float) -> PredictResponse:
    process_dir = os.path.join(PATH_PROCESS, "Deep_Aneurysm", req.study_id)
    mra_path = req.input_paths[0] if req.input_paths else ""
    dicom_dir = req.dicom_dirs[0] if req.dicom_dirs else ""

    if not mra_path:
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg="No MRA_BRAIN input path provided")

    base_env = _build_env()
    logger.info("[aneurysm] preprocess start")
    err = _run_step(
        "Preprocess",
        ["python", "/app/preprocess.py",
         "--study_id", req.study_id,
         "--mra_path", mra_path,
         "--dicom_dir", dicom_dir,
         "--process_dir", process_dir],
        base_env, start,
    )
    if err:
        return err
    return PredictResponse(status="ok", elapsed_time=time.time() - start)


def _run_inference(req: PredictRequest, start: float) -> PredictResponse:
    process_dir = os.path.join(PATH_PROCESS, "Deep_Aneurysm", req.study_id)
    dicom_dir = req.dicom_dirs[0] if req.dicom_dirs else ""
    gpu_env = _build_env(gpu_id=req.gpu_id)

    logger.info("[aneurysm] inference start (GPU %d)", req.gpu_id)
    err = _run_step(
        "Inference",
        ["python", "/app/inference.py",
         "--process_dir", process_dir,
         "--brain_model", PATH_BRAIN_MODEL,
         "--vessel_model", PATH_VESSEL_MODEL,
         "--aneurysm_model", PATH_ANEURYSM_MODEL,
         "--gpu_id", str(req.gpu_id),
         "--code_dir", PATH_CODE,
         "--dicom_dir", dicom_dir],
        gpu_env, start,
    )
    if err:
        return err

    pred_path = pathlib.Path(process_dir) / "nnUNet" / "Pred.nii.gz"
    if not pred_path.exists():
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg="Silent failure: Pred.nii.gz not found after inference")
    if pred_path.stat().st_mtime < start:
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg="Stale Pred.nii.gz mtime < request start")

    return PredictResponse(status="ok", elapsed_time=time.time() - start)


def _run_postprocess(req: PredictRequest, start: float) -> PredictResponse:
    process_dir = os.path.join(PATH_PROCESS, "Deep_Aneurysm", req.study_id)
    base_env = _build_env()

    # CP1c (AP-085): per-task req.group_id 優先；fallback 到 module-level env GROUP_ID（legacy）
    gid = req.group_id if req.group_id is not None else GROUP_ID
    logger.info("[aneurysm] postprocess start group_id=%s (req=%s, env=%s)", gid, req.group_id, GROUP_ID)
    err = _run_step(
        "Postprocess",
        ["python", "/app/postprocess.py",
         "--study_id", req.study_id,
         "--process_dir", process_dir,
         "--output_dir", req.output_folder,
         "--code_dir", PATH_CODE,
         "--json_dir", PATH_JSON,
         "--group_id", str(gid),
         "--input_json", req.input_json],
        base_env, start,
    )
    if err:
        return err

    output_pred = pathlib.Path(req.output_folder) / "Pred_Aneurysm.nii.gz"
    if not output_pred.exists():
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg=f"Postprocess completed but {output_pred} not found")

    return PredictResponse(status="ok", elapsed_time=time.time() - start)


# ── Main endpoint with phase routing ─────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()

    logger.info("[aneurysm] predict study=%s gpu=%d phase=%s",
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
    logger.info("[aneurysm] OK study=%s elapsed=%.1fs", req.study_id, elapsed)
    return PredictResponse(status="ok", elapsed_time=elapsed)
