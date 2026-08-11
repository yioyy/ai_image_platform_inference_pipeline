"""Aneurysm In-Process Inference Server (Step 2 refactor).

原 subprocess-per-case → 改成 long-running in-process (跟 synthseg 相同 pattern):
  - 容器啟動 warmup: 3 nnUNet models → CPU RAM warm + vessel16 → GPU 常駐
  - Handler 直接 import & call aneurysm_{preprocess,inference,postprocess}
  - GPU serialization: threading.Lock (with 上游 bp_worker 的 Redis rad_gpu_0 lock)
  - Errors 走 PredictResponse(status="error"),不再 raise HTTPException 避免破 Pydantic response model
  - NNUNET_CUDNN_BENCHMARK 應該在 compose 設 =1 (in-process autotune 一次,全 case 受益)

Env flags 對加速的影響:
  NNUNET_CUDNN_BENCHMARK=1     — cudnn 自動 tune kernel (in-process 只花一次,+5-15%/case)
  NNUNET_COMPILE=2             — torch.compile max-autotune (Strategy D, 啟動 +120s,+10-20%/case)
  NNUNET_BACKEND=warm (default) — model 常駐 CPU RAM,per predict swap 到 GPU
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────
# CRITICAL: spawn method MUST be set before any torch / CUDA-touching import
# (AP-098 fix: nnUNet 內部 batchgenerators / DataLoader fork 會 deadlock)
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
# ─────────────────────────────────────────────────────────────────────────

import contextlib
import faulthandler
import json
import logging
import os
import pathlib
import sys
import threading
import time

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

# ── PYTHONPATH shim: fake_modules for aneurysm_helpers imports ────────────
if FAKE_MODULES_DIR not in sys.path:
    sys.path.insert(0, FAKE_MODULES_DIR)


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

# ── In-process phase modules (import triggers heavy PyTorch/cudnn init) ────
# 放這裡確保 spawn method 已在最上面設過。
sys.path.insert(0, "/app")
from preprocess import aneurysm_preprocess
from inference import aneurysm_inference, warmup_vessel16
from postprocess import aneurysm_postprocess
from code_ai.inference.nnunet_predict import warmup as warmup_nnunet


# ── GPU serialization lock ────────────────────────────────────────────────
# uvicorn 預設 sync def handler 走 threadpool,2 個 request 可能並行進 GPU 段
# 上游 bp_worker Redis rad_gpu_0 lock 已 serialize,但 container 內加一把
# threading.Lock 是 belt-and-suspenders (防上游 lock 失效 / batch retry race)
_GPU_LOCK = threading.Lock()

# ── Phase watchdog ────────────────────────────────────────────────────────
# A normal study is ~2.5 min end to end; the slowest observed full run was 429s
# and that included a 300s stall. 15 minutes is far outside anything healthy
# while still leaving room for an unusually large volume. Set 0 to disable.
PHASE_TIMEOUT_S = int(os.environ.get("ANEURYSM_PHASE_TIMEOUT_S", "900"))
# On a bind mount, so the host-side notifier can see what the container left.
# AI_INFERENCE_RESULT_PATH is set explicitly in compose and is a bind mount, so
# the host-side notifier can read what lands here. Deriving it from ~ instead
# would depend on HOME inside a runuser shell, and a marker written somewhere
# unmounted is a marker nobody ever sees.
STUCK_DIR = os.environ.get(
    "ANEURYSM_STUCK_DIR",
    os.path.join(os.environ.get("AI_INFERENCE_RESULT_PATH",
                                "/home/david/ai-inference-result"), "_stuck"))
EXIT_STUCK = 87


@contextlib.contextmanager
def phase_watchdog(phase: str, study_id: str):
    """Kill the process if `phase` outlives PHASE_TIMEOUT_S, loudly.

    The timer thread is a daemon and does nothing at all on the normal path, so
    the cost of this is one thread per request and no wakeups.
    """
    if PHASE_TIMEOUT_S <= 0:
        yield
        return

    finished = threading.Event()
    started = time.time()

    def _fire():
        if finished.wait(PHASE_TIMEOUT_S):
            return  # normal completion
        elapsed = time.time() - started
        stamp = time.strftime("%Y%m%dT%H%M%S")
        try:
            os.makedirs(STUCK_DIR, exist_ok=True)
            trace_path = os.path.join(STUCK_DIR, f"{stamp}_{study_id}_{phase}.stacks.txt")
            with open(trace_path, "w") as fh:
                fh.write(f"study={study_id} phase={phase} elapsed={elapsed:.0f}s\n\n")
                # Every thread, not just the stuck one — the stack that matters
                # may be a worker, and there is no second chance to collect it.
                faulthandler.dump_traceback(file=fh, all_threads=True)
            with open(os.path.join(STUCK_DIR, f"{stamp}_{study_id}_{phase}.json"), "w") as fh:
                json.dump({
                    "study_id": study_id,
                    "phase": phase,
                    "elapsed_s": round(elapsed),
                    "timeout_s": PHASE_TIMEOUT_S,
                    "stacks": trace_path,
                    "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "action": "os._exit(%d); container restart policy recovers" % EXIT_STUCK,
                }, fh, indent=2)
        except Exception:
            logger.exception("[watchdog] failed to record state; exiting anyway")

        logger.error(
            "[watchdog] study=%s phase=%s exceeded %ds (elapsed %.0fs) — "
            "dumping stacks to %s and exiting %d so the container restarts clean",
            study_id, phase, PHASE_TIMEOUT_S, elapsed, STUCK_DIR, EXIT_STUCK,
        )
        # Also to stderr: if the log handler is what is wedged, this still lands.
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush()
        os._exit(EXIT_STUCK)

    t = threading.Thread(target=_fire, name=f"watchdog-{phase}", daemon=True)
    t.start()
    try:
        yield
    finally:
        finished.set()


app = FastAPI(title="Aneurysm In-Process Inference Server")


@app.on_event("startup")
def warmup_on_startup() -> None:
    """Pre-load 3 nnUNet models to CPU RAM + vessel16 to GPU (once per container).

    First case pays 0 warm cost. NNUNET_COMPILE=2 makes this ~120s longer
    but subsequent cases get +10-20% forward speedup.
    """
    logger.info("[startup] warming up models (backend=%s, compile=%s, cudnn.benchmark=%s)...",
                os.environ.get("NNUNET_BACKEND", "warm"),
                os.environ.get("NNUNET_COMPILE", "0"),
                os.environ.get("NNUNET_CUDNN_BENCHMARK", "0"))
    t0 = time.time()
    try:
        # 3 nnUNet models → CPU RAM (LAZY_GPU_SWAP baseline)
        warmup_nnunet([
            (PATH_BRAIN_MODEL, (0,), "checkpoint_best.pth", "plans.json", 3),
            (PATH_VESSEL_MODEL, (0,), "checkpoint_best.pth", "plans.json", 3),
            (PATH_ANEURYSM_MODEL, (13,), "checkpoint_best.pth", "nnUNetPlans_5L-b900.json", 3),
        ])
        # Vessel16 → GPU 常駐 (太小不值得 CPU-swap)
        warmup_vessel16()
    except Exception as exc:
        logger.exception("[startup] warmup FAILED: %s (will lazy-load per case)", exc)
    logger.info("[startup] warmup done (%.1fs)", time.time() - t0)


@app.get("/health")
def health():
    gpu_info = check_gpu_health()
    if not gpu_info["gpu_ok"]:
        # nvidia-container-toolkit cgroup denial (e.g. after systemd
        # daemon-reload on host) cannot recover in-process. Schedule clean
        # exit so docker `restart: unless-stopped` resurrects with a fresh
        # nvidia hook. 2s delay lets this response return to the caller.
        logger.error("[autoheal] NVML degraded — scheduling container exit")
        threading.Timer(2.0, lambda: os._exit(1)).start()
    return {
        "status": "ok" if gpu_info["gpu_ok"] else "degraded",
        "pipeline": "aneurysm",
        "gpu": gpu_info,
    }


# ── Phase functions (in-process direct calls) ─────────────────────────────

def _run_preprocess(req: PredictRequest, start: float) -> PredictResponse:
    process_dir = os.path.join(PATH_PROCESS, "Deep_Aneurysm", req.study_id)
    mra_path = req.input_paths[0] if req.input_paths else ""
    dicom_dir = req.dicom_dirs[0] if req.dicom_dirs else ""

    if not mra_path:
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg="No MRA_BRAIN input path provided")

    logger.info("[aneurysm] preprocess start")
    try:
        ok = aneurysm_preprocess(req.study_id, mra_path, dicom_dir, process_dir)
    except Exception as exc:
        logger.exception("[aneurysm] preprocess exception")
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg=f"Preprocess exception: {exc}")
    if not ok:
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg="Preprocess returned False")
    logger.info("[aneurysm] Preprocess done")
    return PredictResponse(status="ok", elapsed_time=time.time() - start)


def _run_inference(req: PredictRequest, start: float) -> PredictResponse:
    process_dir = os.path.join(PATH_PROCESS, "Deep_Aneurysm", req.study_id)
    dicom_dir = req.dicom_dirs[0] if req.dicom_dirs else ""

    logger.info("[aneurysm] inference start (GPU %d) [in-process]", req.gpu_id)
    try:
        # Watchdog inside the lock, so the clock covers only real work and not
        # time spent queued behind another request.
        with _GPU_LOCK, phase_watchdog("inference", req.study_id):
            ok = aneurysm_inference(
                process_dir,
                PATH_BRAIN_MODEL,
                PATH_VESSEL_MODEL,
                PATH_ANEURYSM_MODEL,
                gpu_id=req.gpu_id,
                code_dir=PATH_CODE,
                dicom_dir=dicom_dir,
            )
    except Exception as exc:
        logger.exception("[aneurysm] inference exception")
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg=f"Inference exception: {exc}")
    if not ok:
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg="Inference returned False")

    pred_path = pathlib.Path(process_dir) / "nnUNet" / "Pred.nii.gz"
    if not pred_path.exists():
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg="Silent failure: Pred.nii.gz not found after inference")
    if pred_path.stat().st_mtime < start:
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg="Stale Pred.nii.gz mtime < request start")

    logger.info("[aneurysm] Inference done")
    return PredictResponse(status="ok", elapsed_time=time.time() - start)


def _run_postprocess(req: PredictRequest, start: float) -> PredictResponse:
    process_dir = os.path.join(PATH_PROCESS, "Deep_Aneurysm", req.study_id)

    # CP1c (AP-085): per-task req.group_id 優先；fallback 到 module-level env GROUP_ID
    gid = req.group_id if req.group_id is not None else GROUP_ID
    logger.info("[aneurysm] postprocess start group_id=%s (req=%s, env=%s)", gid, req.group_id, GROUP_ID)
    try:
        ok = aneurysm_postprocess(
            req.study_id, process_dir, req.output_folder,
            group_id=gid, input_json=req.input_json,
        )
    except Exception as exc:
        logger.exception("[aneurysm] postprocess exception")
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg=f"Postprocess exception: {exc}")
    if not ok:
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg="Postprocess returned False")

    output_pred = pathlib.Path(req.output_folder) / "Pred_Aneurysm.nii.gz"
    if not output_pred.exists():
        return PredictResponse(status="error", elapsed_time=time.time() - start,
                               error_msg=f"Postprocess completed but {output_pred} not found")

    logger.info("[aneurysm] Postprocess done")
    return PredictResponse(status="ok", elapsed_time=time.time() - start)


# ── Main endpoint with phase routing ─────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()

    logger.info("[aneurysm] predict study=%s gpu=%d phase=%s [in-process]",
                req.study_id, req.gpu_id, req.phase or "full")

    if req.phase == "preprocess":
        return _run_preprocess(req, start)
    elif req.phase == "inference":
        return _run_inference(req, start)
    elif req.phase == "postprocess":
        return _run_postprocess(req, start)

    # Full pipeline
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
