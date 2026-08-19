#!/usr/bin/env python3
"""Infarct in-process inference server — phase routing, port 5002.

Shaped after docker/aneurysm/server.py, with one deliberate difference: SynthSeg
runs in preprocess, not inference.

Aneurysm cannot do that. Its SynthSeg call is Stage C.5, inside legacy inference
code, which runs inside the Redis rad_gpu_0 lock — and the SynthSeg container
takes the same lock (LAZY_GPU_SWAP=1). SET NX is not reentrant, so C.5 blocked
until its own lock expired: 292 wasted seconds per study, and the neck filter it
gates never once applied. That was patched with a dispatcher-side prewarm whose
payload must stay byte-identical or the in-lock call misses its cache and
deadlocks again. This container is new code, so it makes the call from
preprocess where no lock is held, and skips the whole mechanism.

The consequence is that preprocess here is not CPU-only the way the aneurysm one
is: it can block on the GPU lock. The dispatcher's preprocess timeout is a
hardcoded 600s for every model, which is not enough for a SynthSeg run queued
behind another study, so it needs to become per-model before this goes live.
Tracked in the plan; the phase watchdog below is not a substitute, since it
fires inside the container and the dispatcher would already have given up.
"""

from __future__ import annotations

# Before torch, as in the aneurysm server (AP-098): a fork-started worker
# inherits a CUDA context and deadlocks on first use.
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_server import PredictRequest, PredictResponse, check_gpu_health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("infarct.server")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


PATH_PROCESS = _env("PATH_PROCESS", "/home/david/pipeline/chuan/process")
PATH_INFARCT_MODEL = _env(
    "PATH_INFARCT_MODEL",
    "/opt/shh_aiplatform-chuan/pipeline/chuan/code/nnUNet/nnUNet_results/"
    "Dataset300_DeepInfarct_v2_3ch/nnUNetTrainer__nnUNetPlans__3d_fullres",
)

from preprocess import infarct_preprocess
from inference import infarct_inference
from postprocess import infarct_postprocess

# The upstream bp_worker Redis lock already serialises GPU work; this is
# belt-and-suspenders against a retry racing itself inside one container.
_GPU_LOCK = threading.Lock()

PHASE_TIMEOUT_S = int(os.environ.get("INFARCT_PHASE_TIMEOUT_S", "900"))
# On a bind mount, so the host-side notifier can read what the container left.
# Deriving this from ~ would depend on HOME inside a runuser shell, and a marker
# written somewhere unmounted is a marker nobody ever sees.
STUCK_DIR = os.environ.get(
    "INFARCT_STUCK_DIR",
    os.path.join(os.environ.get("AI_INFERENCE_RESULT_PATH",
                                "/home/david/ai-inference-result"), "_stuck"))
EXIT_STUCK = 87


@contextlib.contextmanager
def phase_watchdog(phase: str, study_id: str):
    """Kill the process if `phase` outlives PHASE_TIMEOUT_S, loudly."""
    if PHASE_TIMEOUT_S <= 0:
        yield
        return

    finished = threading.Event()
    started = time.time()

    def _fire():
        if finished.wait(PHASE_TIMEOUT_S):
            return
        elapsed = time.time() - started
        stamp = time.strftime("%Y%m%dT%H%M%S")
        try:
            os.makedirs(STUCK_DIR, exist_ok=True)
            trace_path = os.path.join(STUCK_DIR, f"{stamp}_{study_id}_{phase}.stacks.txt")
            with open(trace_path, "w") as fh:
                fh.write(f"study={study_id} phase={phase} elapsed={elapsed:.0f}s\n\n")
                # Every thread: the stack that matters may be a worker, and
                # there is no second chance to collect it.
                faulthandler.dump_traceback(file=fh, all_threads=True)
            with open(os.path.join(STUCK_DIR, f"{stamp}_{study_id}_{phase}.json"), "w") as fh:
                json.dump({
                    "study_id": study_id, "phase": phase,
                    "elapsed_s": round(elapsed), "timeout_s": PHASE_TIMEOUT_S,
                    "stacks": trace_path,
                    "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "action": f"os._exit({EXIT_STUCK}); container restart policy recovers",
                }, fh, indent=2)
        except Exception:
            logger.exception("[watchdog] failed to record state; exiting anyway")

        logger.error(
            "[watchdog] study=%s phase=%s exceeded %ds (elapsed %.0fs) — "
            "dumping stacks to %s and exiting %d so the container restarts clean",
            study_id, phase, PHASE_TIMEOUT_S, elapsed, STUCK_DIR, EXIT_STUCK)
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


app = FastAPI(title="Infarct In-Process Inference Server")


@app.get("/health")
def health():
    gpu_info = check_gpu_health()
    if not gpu_info["gpu_ok"]:
        # An nvidia-container-toolkit cgroup denial cannot recover in-process.
        # Schedule a clean exit so the restart policy resurrects the container
        # with a fresh nvidia hook; 2s lets this response reach the caller.
        logger.error("[autoheal] NVML degraded — scheduling container exit")
        threading.Timer(2.0, lambda: os._exit(1)).start()
    return {
        "status": "ok" if gpu_info["gpu_ok"] else "degraded",
        "pipeline": "infarct",
        "gpu": gpu_info,
    }


def _process_dir(study_id: str) -> str:
    return os.path.join(PATH_PROCESS, "Deep_Infarct", study_id)


def _err(start: float, msg: str) -> PredictResponse:
    return PredictResponse(status="error", elapsed_time=time.time() - start,
                           error_msg=msg)


def _inputs_by_label(req: PredictRequest) -> dict:
    """Map input paths to ADC / DWI0 / DWI1000 by filename.

    The dispatcher passes paths positionally in the order config.yaml defines,
    and that ordering is already enforced upstream — _reorder_series_by_config
    raises for Infarct rather than proceeding in an arbitrary order. Matching on
    the filename here anyway costs nothing and means a future ordering change
    upstream cannot silently feed ADC where DWI1000 belongs.
    """
    found = {}
    for p in req.input_paths or []:
        stem = os.path.basename(p).split(".")[0].upper()
        for label in ("DWI1000", "DWI0", "ADC"):   # DWI1000 before DWI0
            if stem == label:
                found.setdefault(label, p)
                break
    return found


def _run_preprocess(req: PredictRequest, start: float) -> PredictResponse:
    process_dir = _process_dir(req.study_id)
    found = _inputs_by_label(req)
    missing = [k for k in ("ADC", "DWI0", "DWI1000") if k not in found]
    if missing:
        return _err(start, f"missing inputs {missing} in {req.input_paths}")

    dicom_dirs = {}
    for d in req.dicom_dirs or []:
        name = os.path.basename(os.path.normpath(d)).upper()
        if name in ("ADC", "DWI0", "DWI1000"):
            dicom_dirs[name] = d

    logger.info("[infarct] preprocess start")
    try:
        # SynthSeg runs here and can queue on the GPU lock, so the watchdog
        # covers this phase too — unlike the aneurysm server, where preprocess
        # is pure filesystem work.
        with phase_watchdog("preprocess", req.study_id):
            ok = infarct_preprocess(
                req.study_id, found["ADC"], found["DWI0"], found["DWI1000"],
                dicom_dirs, process_dir)
    except Exception as exc:
        logger.exception("[infarct] preprocess exception")
        return _err(start, f"Preprocess exception: {exc}")
    if not ok:
        return _err(start, "Preprocess returned False")

    merged = pathlib.Path(process_dir) / "SynthSeg_merged.nii.gz"
    if not merged.exists():
        return _err(start, "Silent failure: SynthSeg_merged.nii.gz not found")
    logger.info("[infarct] Preprocess done")
    return PredictResponse(status="ok", elapsed_time=time.time() - start)


def _run_inference(req: PredictRequest, start: float) -> PredictResponse:
    process_dir = _process_dir(req.study_id)
    logger.info("[infarct] inference start (GPU %d) [in-process]", req.gpu_id)
    try:
        # Watchdog inside the lock, so the clock covers real work rather than
        # time spent queued behind another request.
        with _GPU_LOCK, phase_watchdog("inference", req.study_id):
            ok = infarct_inference(process_dir, PATH_INFARCT_MODEL, gpu_id=req.gpu_id)
    except Exception as exc:
        logger.exception("[infarct] inference exception")
        return _err(start, f"Inference exception: {exc}")
    if not ok:
        return _err(start, "Inference returned False")

    pred = pathlib.Path(process_dir) / "nnUNet" / "Pred.nii.gz"
    if not pred.exists():
        return _err(start, "Silent failure: Pred.nii.gz not found after inference")
    if pred.stat().st_mtime < start:
        return _err(start, "Stale Pred.nii.gz mtime < request start")

    logger.info("[infarct] Inference done")
    return PredictResponse(status="ok", elapsed_time=time.time() - start)


def _run_postprocess(req: PredictRequest, start: float) -> PredictResponse:
    process_dir = _process_dir(req.study_id)
    logger.info("[infarct] postprocess start")
    try:
        ok = infarct_postprocess(req.study_id, process_dir, req.output_folder)
    except Exception as exc:
        logger.exception("[infarct] postprocess exception")
        return _err(start, f"Postprocess exception: {exc}")
    if not ok:
        return _err(start, "Postprocess returned False")

    out = pathlib.Path(req.output_folder) / "prediction.json"
    if not out.exists():
        return _err(start, f"Postprocess completed but {out} not found")

    logger.info("[infarct] Postprocess done")
    return PredictResponse(status="ok", elapsed_time=time.time() - start)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()
    logger.info("[infarct] predict study=%s gpu=%d phase=%s [in-process]",
                req.study_id, req.gpu_id, req.phase or "full")

    if req.phase == "preprocess":
        return _run_preprocess(req, start)
    if req.phase == "inference":
        return _run_inference(req, start)
    if req.phase == "postprocess":
        return _run_postprocess(req, start)

    for step in (_run_preprocess, _run_inference, _run_postprocess):
        resp = step(req, start)
        if resp.status != "ok":
            return resp

    elapsed = time.time() - start
    logger.info("[infarct] full pipeline done in %.1fs", elapsed)
    return PredictResponse(status="ok", elapsed_time=elapsed)
