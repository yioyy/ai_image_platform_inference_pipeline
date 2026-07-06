"""
SynthSeg Inference Server  (Brain Seg 1/4 — CP1-CP7a)
=====================================================
Centralised SynthSeg service.  Models are loaded once at startup; subsequent
requests reuse the loaded weights, eliminating ~50 s of per-request TF init.

Backends
--------
SYNTHSEG_BACKEND=pytorch — PyTorch (default, ~14 s warm, ~10 GB VRAM peak)  ← recommended
SYNTHSEG_BACKEND=tf      — TensorFlow 2.14 (~22-32 s per run, ~19 GB VRAM)
SYNTHSEG_BACKEND=onnx    — ONNX Runtime GPU (~22 s per run, ~30 GB VRAM)

Phases (CP13c)
--------------
phase=""       — Full pipeline: resample + GPU inference + CPU postprocess (used by CMB preprocess.py)
phase="gpu_only" — Resample + GPU inference only; caller does local CPU postprocess (task_pipeline)

Memory management
-----------------
~10MB RSS leak per request. Auto-restart via SIGTERM when RSS > SYNTHSEG_RSS_LIMIT_MB (default 4096).
Docker restart: unless-stopped brings the container back in ~30s.

TF is imported transitively by code_ai even in ONNX/PyTorch mode.
Without configuration TF pre-allocates ALL GPU VRAM.
Fix: set memory_growth=True BEFORE any code_ai import.

Endpoints
---------
GET  /health              server status (includes rss_mb, request_count)
GET  /info                model metadata
POST /predict             run SynthSeg (see PredictRequest)
DELETE /cache             clear cache for a study
"""

# ── Step 0: MUST set GPU config BEFORE any code_ai / TF import ───────────────
import os
import sys

_BP_ROOT = os.environ.get(
    "BRAIN_PARCELLATION_ROOT",
    "/opt/shh_aiplatform-david/brain-parcellation",
)
if _BP_ROOT not in sys.path:
    sys.path.insert(0, _BP_ROOT)

# Silence TF C++ log spam
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# XLA JIT fix: TF 2.14 needs to find nvvm/libdevice/libdevice.10.bc for GPU ops
# (e.g. GaussianBlur, Pow, Elu). Point --xla_gpu_cuda_data_dir to the conda env
# root so XLA looks in $root/nvvm/libdevice/.
_XLA_CUDA_ROOT = "/opt/conda/envs/tf_2_14"
os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={_XLA_CUDA_ROOT}")

import tensorflow as tf  # noqa: E402

_gpus = tf.config.list_physical_devices("GPU")
for _g in _gpus:
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except RuntimeError:
        pass  # already initialized

# ── Normal imports ────────────────────────────────────────────────────────────
import logging
import signal
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional

import nibabel as nib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [synthseg] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_BACKEND = os.environ.get("SYNTHSEG_BACKEND", "pytorch").lower()
_ONNX_DIR = Path(os.environ.get("SYNTHSEG_ONNX_DIR", "/data/4TB/ai_pipeline/onnx_models"))
_GPU_N = int(os.environ.get("GPU_N", "0"))

logger.info("Backend: %s  GPU: %d  BP_ROOT: %s", _BACKEND, _GPU_N, _BP_ROOT)

# ── RAD Phase 3a: lazy CPU<->GPU swap + redis lock ────────────────────────────
# Enabled by LAZY_GPU_SWAP=1. Model load goes to CPU RAM at startup; per /predict,
# acquire redis gpu_lock, swap to GPU, forward, move result to CPU, swap back,
# release lock. PyTorch backend only — ONNX/TF unchanged.
import contextlib

_LAZY_GPU_SWAP = os.environ.get("LAZY_GPU_SWAP", "0") == "1"
_GPU_LOCK_REDIS_URL = os.environ.get("GPU_LOCK_REDIS_URL", "redis://localhost:10079/2")
_GPU_LOCK_KEY = os.environ.get("GPU_LOCK_KEY", "rad_gpu_0")
_GPU_LOCK_TIMEOUT_S = int(os.environ.get("GPU_LOCK_TIMEOUT_S", "1800"))

if _LAZY_GPU_SWAP:
    logger.info(
        "LAZY_GPU_SWAP=1: model in CPU RAM idle, swap to GPU per request "
        "(lock=%s key=%s timeout=%ds)",
        _GPU_LOCK_REDIS_URL, _GPU_LOCK_KEY, _GPU_LOCK_TIMEOUT_S,
    )

_redis_client = None

def _get_redis():
    """Lazy redis client init (avoid import cost when LAZY_GPU_SWAP=0)."""
    global _redis_client
    if _redis_client is None:
        import redis as _redis
        _redis_client = _redis.from_url(_GPU_LOCK_REDIS_URL)
    return _redis_client


@contextlib.contextmanager
def _redis_gpu_lock():
    """Redis cross-container mutex for shared 1-GPU mode. No-op if LAZY_GPU_SWAP=0.

    Note: original _gpu_lock = threading.Lock() (line ~164) is in-process serialization
    and stays untouched. This redis lock layers ON TOP for cross-container coordination.
    """
    if not _LAZY_GPU_SWAP:
        yield
        return
    r = _get_redis()
    val = f"synthseg:{os.getpid()}"
    deadline = time.time() + _GPU_LOCK_TIMEOUT_S
    acquired = False
    while time.time() < deadline:
        if r.set(_GPU_LOCK_KEY, val, nx=True, ex=_GPU_LOCK_TIMEOUT_S):
            acquired = True
            logger.info("[gpu_lock] acquired by %s", val)
            break
        time.sleep(0.5)
    if not acquired:
        raise TimeoutError(
            f"GPU lock not acquired within {_GPU_LOCK_TIMEOUT_S}s by {val}"
        )
    try:
        yield
    finally:
        # Lua: only delete if we still own it (guard against TTL expire race)
        lua = (
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "return redis.call('del', KEYS[1]) else return 0 end"
        )
        r.eval(lua, 1, _GPU_LOCK_KEY, val)
        logger.info("[gpu_lock] released by %s", val)


# ── Lazy imports (avoid long TF init at module level) ─────────────────────────
# These are populated in load_models() on startup.
_synth_seg = None          # SynthSegOnnx (preprocessing / postprocessing)
_net_unet2 = None          # ONNX session (unet2) — onnx backend only
_net_parcellation = None   # ONNX session (parcellation) — onnx backend only
_pt_seg = None             # SynthSegUNet2 (pytorch backend)
_pt_parc = None            # SynthSegParc  (pytorch backend)
_model_info: Dict = {}

# One request at a time for GPU inference.  Released before postprocessing so
# CPU-only WM parcellation (~42 s) does not block the next GPU request.
_gpu_lock = threading.Lock()

# ── Memory watchdog — auto-restart when RSS exceeds threshold ────────────────
# Memory leak ~10MB/request; restart before OOM kills us (~4 GB threshold).
_RSS_LIMIT_MB = int(os.environ.get("SYNTHSEG_RSS_LIMIT_MB", "4096"))
_request_count = 0


def _check_rss_and_restart() -> None:
    """If process RSS exceeds threshold, initiate graceful shutdown."""
    global _request_count
    _request_count += 1
    # Check every 5 requests to avoid stat overhead
    if _request_count % 5 != 0:
        return
    try:
        # /proc/self/status is always available in Linux containers
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    rss_mb = rss_kb // 1024
                    if rss_mb > _RSS_LIMIT_MB:
                        logger.warning(
                            "RSS %d MB > limit %d MB after %d requests — shutting down for restart",
                            rss_mb, _RSS_LIMIT_MB, _request_count,
                        )
                        # Signal uvicorn to shut down; docker restart: unless-stopped will bring us back
                        os.kill(os.getpid(), signal.SIGTERM)
                    break
    except Exception:
        pass

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="SynthSeg Inference Server", version="1.0")


@app.on_event("startup")
def load_models() -> None:
    global _synth_seg, _net_unet2, _net_parcellation, _model_info

    if _BACKEND == "pytorch":
        _load_models_pytorch()
    elif _BACKEND == "onnx":
        _load_models_onnx()
    else:
        _load_models_tf()

    threading.Thread(target=_cleanup_jobs, daemon=True, name="job-cleanup").start()


def _load_models_pytorch() -> None:
    global _synth_seg, _pt_seg, _pt_parc, _model_info
    import torch
    from code_ai.utils_synthsegOnnx import SynthSegOnnx
    from code_ai.utils_synthseg_pt import load_synthseg_from_onnx, load_parcellation_from_onnx

    unet2_path = _ONNX_DIR / "synthseg_unet2.onnx"
    parc_path  = _ONNX_DIR / "synthseg_parcellation.onnx"
    if not unet2_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {unet2_path}")
    if not parc_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {parc_path}")

    # RAD Phase 3a: lazy mode loads to CPU (swap to GPU per /predict)
    if _LAZY_GPU_SWAP:
        device = torch.device("cpu")
        logger.info("Loading PyTorch SynthSeg models from %s to CPU (lazy mode) …", _ONNX_DIR)
    else:
        device = torch.device(f"cuda:{_GPU_N}" if torch.cuda.is_available() else "cpu")
        logger.info("Loading PyTorch SynthSeg models from %s (GPU %d) …", _ONNX_DIR, _GPU_N)

    _pt_seg  = load_synthseg_from_onnx(str(unet2_path)).to(device)
    _pt_parc = load_parcellation_from_onnx(str(parc_path)).to(device)

    # Warm-up only when staying on GPU (lazy mode skips — would just swap right out)
    if not _LAZY_GPU_SWAP and device.type == "cuda":
        logger.info("PyTorch warm-up pass …")
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 192, 192, 192, device=device)
            _ = _pt_seg(dummy)
            dummy3 = torch.zeros(1, 3, 192, 192, 192, device=device)
            _ = _pt_parc(dummy3)
        del dummy, dummy3
        torch.cuda.empty_cache()

    # SynthSegOnnx is still needed for pre/postprocessing
    _synth_seg = SynthSegOnnx()
    _model_info = {
        "backend": "pytorch",
        "unet2": str(unet2_path),
        "parcellation": str(parc_path),
        "device": str(device),
        "lazy_gpu_swap": _LAZY_GPU_SWAP,
    }
    logger.info("PyTorch models loaded on %s (lazy=%s).", device, _LAZY_GPU_SWAP)


def _load_models_tf() -> None:
    global _synth_seg, _model_info
    from code_ai.utils_synthseg import SynthSeg, set_gpu
    from code_ai.ext.lab2im import utils

    logger.info("Loading TF SynthSeg models (GPU %d) …", _GPU_N)
    set_gpu(str(_GPU_N))
    synth = SynthSeg(intput_size=192)
    args = synth.load_parameter()

    labels_segmentation, _ = utils.get_list_labels(label_list=args["labels_segmentation"])
    labels_segmentation, unique_idx = np.unique(labels_segmentation, return_index=True)
    labels_denoiser = np.unique(utils.get_list_labels(args["labels_denoiser"])[0])
    labels_parcellation, _ = np.unique(
        utils.get_list_labels(args["labels_parcellation"])[0], return_index=True
    )

    net_unet2, net_convert, net_parcellation = synth.build_model(
        path_model_segmentation=args["path_model_segmentation"],
        path_model_parcellation=args["path_model_parcellation"],
        labels_segmentation=labels_segmentation,
        labels_denoiser=labels_denoiser,
        labels_parcellation=labels_parcellation,
    )
    synth.net_unet2 = net_unet2
    synth.net_convert = net_convert
    synth.net_parcellation = net_parcellation

    _synth_seg = synth
    _model_info = {
        "backend": "tf",
        "unet2": args.get("path_model_segmentation", ""),
        "parcellation": args.get("path_model_parcellation", ""),
    }
    logger.info("TF models loaded.")


def _onnx_providers() -> list:
    """CUDA providers.
    kSameAsRequested: arena only grows to exactly what's needed (no doubling).
    If sessions share the device-level allocator, freed blocks from unet2 are
    reused by parcellation → peak = max(unet2, parc) instead of sum.
    """
    return [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,  # Docker remaps physical GPU_N → device 0
                "arena_extend_strategy": "kSameAsRequested",
            },
        ),
        "CPUExecutionProvider",
    ]


def _load_models_onnx() -> None:
    global _synth_seg, _net_unet2, _net_parcellation, _model_info
    import onnxruntime as ort
    from code_ai.utils_synthsegOnnx import SynthSegOnnx

    unet2_path = _ONNX_DIR / "synthseg_unet2.onnx"
    parc_path = _ONNX_DIR / "synthseg_parcellation.onnx"

    if not unet2_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {unet2_path}")
    if not parc_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {parc_path}")

    logger.info("Loading ONNX models from %s (GPU %d) …", _ONNX_DIR, _GPU_N)
    providers = _onnx_providers()
    _net_unet2 = ort.InferenceSession(str(unet2_path), providers=providers)
    _net_parcellation = ort.InferenceSession(str(parc_path), providers=providers)
    logger.info(
        "ONNX models loaded.  unet2=%s  parc=%s",
        _net_unet2.get_providers(),
        _net_parcellation.get_providers(),
    )

    _synth_seg = SynthSegOnnx()
    _model_info = {
        "backend": "onnx",
        "unet2": str(unet2_path),
        "parcellation": str(parc_path),
        "providers": _net_unet2.get_providers(),
    }


# ── Request / Response schemas ────────────────────────────────────────────────
PostProcess = Literal["none", "cmb", "dwi", "wmh", "all"]


SynthSegPhase = Literal["", "gpu_only"]


class PredictRequest(BaseModel):
    input_path: str                          # full path to input NIfTI
    output_dir: str                          # directory for all outputs
    study_id: str
    post_process: PostProcess = "none"       # extra masks to compute
    run_parcellation: bool = True            # False → seg33 only (faster)
    force: bool = False                      # True → ignore cache
    resample_path: Optional[str] = None      # provide pre-resampled file to skip resample step
    output_basename: Optional[str] = None   # override output file basename (e.g. "{study_id}_{seq}_resample")
    phase: SynthSegPhase = ""               # "" = full pipeline; "gpu_only" = resample + GPU inference (postprocess done by caller)


class PredictResponse(BaseModel):
    status: str
    output_paths: Dict[str, str] = {}        # name → absolute path
    elapsed_time: float = 0.0
    error_msg: str = ""
    cached: bool = False
    timing: Dict[str, float] = {}            # phase → seconds (resample, inference, postprocess)


class AsyncPredictResponse(BaseModel):
    job_id: str
    status: str  # "queued"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str           # queued | running | done | error
    output_paths: Dict[str, str] = {}
    elapsed_time: float = 0.0
    timing: Dict[str, float] = {}
    error_msg: str = ""
    cached: bool = False


class CacheDeleteRequest(BaseModel):
    study_id: str
    output_dir: str


# ── Async job registry ────────────────────────────────────────────────────────
# job_id → JobStatusResponse-compatible dict
_jobs: Dict[str, dict] = {}

# Remove completed jobs older than this many seconds to prevent unbounded growth
_JOB_TTL_S = 3600  # 1 hour


def _cleanup_jobs() -> None:
    """Periodic cleanup of finished jobs older than _JOB_TTL_S."""
    while True:
        time.sleep(300)
        now = time.time()
        stale = [
            jid for jid, j in list(_jobs.items())
            if j["status"] in ("done", "error") and now - j.get("created_at", now) > _JOB_TTL_S
        ]
        for jid in stale:
            _jobs.pop(jid, None)
        if stale:
            logger.info("Cleaned up %d stale jobs", len(stale))


# ── Helpers ───────────────────────────────────────────────────────────────────
def _strip_nii(name: str) -> str:
    for s in (".nii.gz", ".nii"):
        if name.endswith(s):
            return name[: -len(s)]
    return name


def _resample(input_path: Path, resample_path: Path) -> None:
    from code_ai.utils.resample import resample_one
    resample_one(str(input_path), str(resample_path))


def _run_synthseg_onnx(
    resample_path: Path,
    out_synthseg: Path,
    out_synthseg33: Path,
    run_parcellation: bool,
) -> None:
    """Run unet2 (+ optional parcellation) ONNX inference."""
    if run_parcellation:
        _synth_seg.run(
            path_images=str(resample_path),
            path_segmentations=str(out_synthseg),
            path_segmentations33=str(out_synthseg33),
            net_unet2=_net_unet2,
            net_parcellation=_net_parcellation,
        )
    else:
        _synth_seg.run_segmentations33(
            path_images=str(resample_path),
            path_segmentations33=str(out_synthseg33),
            net_unet2=_net_unet2,
        )


def _run_synthseg_tf(
    resample_path: Path,
    out_synthseg: Path,
    out_synthseg33: Path,
    run_parcellation: bool,
) -> None:
    """Run TF SynthSeg inference."""
    if run_parcellation:
        _synth_seg.run(
            path_images=str(resample_path),
            path_segmentations=str(out_synthseg),
            path_segmentations33=str(out_synthseg33),
        )
    else:
        _synth_seg.run_segmentations33(
            path_images=str(resample_path),
            path_segmentations33=str(out_synthseg33),
        )


def _run_synthseg_pytorch(
    resample_path: Path,
    out_synthseg: Path,
    out_synthseg33: Path,
    run_parcellation: bool,
) -> None:
    """Run PyTorch SynthSeg inference (~1 s warm, ~10 GB VRAM peak)."""
    import torch
    from code_ai.ext.lab2im import utils

    args = _synth_seg.load_parameter()
    labels_segmentation, _ = utils.get_list_labels(label_list=args["labels_segmentation"])
    labels_segmentation, _ = __import__("numpy").unique(labels_segmentation, return_index=True)
    labels_parcellation, _ = __import__("numpy").unique(
        utils.get_list_labels(args["labels_parcellation"])[0], return_index=True
    )
    cropping = args["crop"]
    min_pad = utils.reformat_to_list(cropping, length=3, dtype="int") if cropping else 128

    # ── CPU preprocess (outside lock) ─────────────────────────────────────────
    image, aff, h, im_res, shape, pad_idx, crop_idx = _synth_seg.preprocess(
        path_image=str(resample_path), ct=args["ct"], crop=cropping, min_pad=min_pad
    )
    image = image.astype(np.float32)
    # image: [1, H, W, D, 1] channels-last float32

    # ── GPU phase: lock + (optional) lazy swap + forward unet2 (+ parc) ──────
    parc_output = None
    with _redis_gpu_lock():
        cuda_device = torch.device(f"cuda:{_GPU_N}" if torch.cuda.is_available() else "cpu")
        if _LAZY_GPU_SWAP and cuda_device.type == "cuda":
            t0 = time.time()
            _pt_seg.to(cuda_device)
            _pt_parc.to(cuda_device)
            logger.info("[lazy_swap] CPU->GPU %.2fs", time.time() - t0)

        try:
            device = next(_pt_seg.parameters()).device

            # unet2 forward
            x = torch.from_numpy(image.transpose(0, 4, 1, 2, 3)).to(device)
            with torch.no_grad():
                seg_out = _pt_seg(x)
            unet2_output = seg_out.cpu().numpy().transpose(0, 2, 3, 4, 1)
            del x, seg_out

            # parcellation forward (optional, still on GPU under same lock)
            if run_parcellation:
                idx = unet2_output[0].argmax(-1)
                mask_1 = np.logical_or(idx == 2, idx == 20).astype(np.float32)
                mask_2 = np.logical_and(idx != 2, idx != 20).astype(np.float32)
                parc_np = np.stack([image[0, ..., 0], mask_2, mask_1], axis=0)
                xp = torch.from_numpy(parc_np).unsqueeze(0).to(device)
                with torch.no_grad():
                    parc_out = _pt_parc(xp)
                parc_output = parc_out.cpu().numpy().transpose(0, 2, 3, 4, 1)
                del xp, parc_out
        finally:
            if _LAZY_GPU_SWAP and cuda_device.type == "cuda":
                t0 = time.time()
                _pt_seg.to("cpu")
                _pt_parc.to("cpu")
                torch.cuda.empty_cache()
                logger.info("[lazy_swap] GPU->CPU %.2fs", time.time() - t0)

    # ── CPU postprocess + save (outside lock — fellow containers can use GPU now) ──
    h.set_data_dtype("int16")
    seg33 = _synth_seg.postprocess(
        post_patch_seg=unet2_output,
        post_patch_parc=None,
        shape=shape, pad_idx=pad_idx, crop_idx=crop_idx,
        labels_segmentation=labels_segmentation,
        labels_parcellation=None,
        aff=aff, im_res=im_res,
        fast=args["fast"],
        topology_classes=args["topology_classes"],
        v1=args["v1"],
        return_seg=True, return_posteriors=False,
    )
    utils.save_volume(seg33, aff, h, str(out_synthseg33), dtype="int16")

    if not run_parcellation:
        return

    seg_parc = _synth_seg.postprocess(
        post_patch_seg=unet2_output,
        post_patch_parc=parc_output,
        shape=shape, pad_idx=pad_idx, crop_idx=crop_idx,
        labels_segmentation=labels_segmentation,
        labels_parcellation=labels_parcellation,
        aff=aff, im_res=im_res,
        fast=False,
        topology_classes=args["topology_classes"],
        v1=False,
        return_seg=True, return_posteriors=False,
    )
    utils.save_volume(seg_parc, aff, h, str(out_synthseg), dtype="int16")


def _compute_post_process(
    out_synthseg: Path,
    out_synthseg33: Path,
    post_process: PostProcess,
    depth_number: int = 5,
) -> Dict[str, Path]:
    """
    Compute derivative masks (david/wm/CMB/DWI/WMH_PVS) from synthseg outputs.
    Returns dict of name → output_path.
    """
    if post_process == "none" or not out_synthseg.exists():
        return {}

    from code_ai.pipeline.synthseg.postprocess import (
        compute_cmb,
        compute_dwi,
        compute_white_matter_masks,
        compute_wmh,
        save_volume,
    )

    synthseg_nii = nib.load(str(out_synthseg))
    synthseg_arr = np.asarray(synthseg_nii.dataobj)
    synthseg33_nii = nib.load(str(out_synthseg33))
    synthseg33_arr = np.asarray(synthseg33_nii.dataobj)

    seg_arr, wm_arr = compute_white_matter_masks(synthseg_arr, synthseg33_arr, depth_number)

    base = out_synthseg.parent
    stem = _strip_nii(out_synthseg.name).replace("_synthseg", "")
    outputs: Dict[str, Path] = {}

    def _save(arr: np.ndarray, suffix: str) -> Path:
        p = base / f"{stem}{suffix}.nii.gz"
        save_volume(arr, synthseg_nii, p)
        return p

    want_cmb = post_process in ("cmb", "all")
    want_dwi = post_process in ("dwi", "all")
    want_wmh = post_process in ("wmh", "all")
    want_wm  = post_process == "all"

    if want_wm:
        outputs["david"] = _save(seg_arr, "_david")
        outputs["wm"]    = _save(wm_arr,  "_wm")
    if want_cmb:
        outputs["CMB"] = _save(compute_cmb(seg_arr), "_CMB")
    if want_dwi:
        outputs["DWI"] = _save(compute_dwi(seg_arr), "_DWI")
    if want_wmh:
        outputs["WMH_PVS"] = _save(
            compute_wmh(synthseg_arr, wm_arr, depth_number), "_WMH_PVS"
        )

    return outputs


def _cache_complete(
    out_resample: Path,
    out_synthseg: Path,
    out_synthseg33: Path,
    post_process: PostProcess,
    run_parcellation: bool,
) -> bool:
    """Return True if all expected output files already exist."""
    if not out_resample.exists():
        return False
    if not out_synthseg33.exists():
        return False
    if run_parcellation and not out_synthseg.exists():
        return False
    # Post-process outputs are NOT checked here — only core files
    return True


# ── Endpoints ─────────────────────────────────────────────────────────────────
def _check_gpu_health() -> dict:
    """Run nvidia-smi to check GPU availability."""
    import subprocess as _sp
    try:
        r = _sp.run(
            ["nvidia-smi", "--query-gpu=gpu_name,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return {"gpu_ok": False, "error": r.stderr.strip()}
        gpus = []
        for line in r.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"name": parts[0], "memory_used": parts[1], "memory_total": parts[2]})
        return {"gpu_ok": True, "gpus": gpus}
    except Exception as e:
        return {"gpu_ok": False, "error": str(e)}


@app.get("/health")
def health():
    loaded = _synth_seg is not None
    if _BACKEND == "pytorch":
        loaded = _pt_seg is not None and _pt_parc is not None
    gpu_info = _check_gpu_health()
    status = "ok" if (loaded and gpu_info.get("gpu_ok")) else "degraded"
    # RSS monitoring
    rss_mb = 0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) // 1024
                    break
    except Exception:
        pass
    return {
        "status": status,
        "pipeline": "synthseg",
        "backend": _BACKEND,
        "gpu": _GPU_N,
        "gpu_health": gpu_info,
        "models_loaded": loaded,
        "rss_mb": rss_mb,
        "rss_limit_mb": _RSS_LIMIT_MB,
        "request_count": _request_count,
    }


@app.get("/info")
def info():
    return {
        "status": "ok",
        "backend": _BACKEND,
        "model_info": _model_info,
        "gpu": _GPU_N,
    }


def _do_predict(req: PredictRequest) -> PredictResponse:
    """Core prediction logic — shared by /predict (sync) and /predict/async."""
    start = time.time()

    try:
        if _synth_seg is None:
            return PredictResponse(
                status="error",
                error_msg="Models not loaded yet — server still starting up",
                elapsed_time=time.time() - start,
            )

        input_path = Path(req.input_path)
        output_dir = Path(req.output_dir)

        if not input_path.exists():
            return PredictResponse(
                status="error",
                error_msg=f"Input file not found: {input_path}",
                elapsed_time=time.time() - start,
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        basename = _strip_nii(input_path.name)

        if req.resample_path:
            resample_path = Path(req.resample_path)
            resample_basename = _strip_nii(resample_path.name)
        elif req.output_basename:
            resample_path = output_dir / f"{req.output_basename}.nii.gz"
            resample_basename = req.output_basename
        else:
            resample_path = output_dir / f"{basename}_resample.nii.gz"
            resample_basename = f"{basename}_resample"

        out_synthseg   = output_dir / f"{resample_basename}_synthseg.nii.gz"
        out_synthseg33 = output_dir / f"{resample_basename}_synthseg33_1mm.nii.gz"

        # ── Cache check ──
        out_synthseg33_native = output_dir / f"{basename}_synthseg33_native.nii.gz"
        if not req.force and _cache_complete(
            resample_path, out_synthseg, out_synthseg33,
            req.post_process, req.run_parcellation
        ):
            logger.info("Cache hit: %s", out_synthseg33)
            outputs = {"resample": str(resample_path), "synthseg33": str(out_synthseg33)}
            if req.run_parcellation:
                outputs["synthseg"] = str(out_synthseg)
            if out_synthseg33_native.exists():
                outputs["synthseg33_native"] = str(out_synthseg33_native)
            return PredictResponse(
                status="ok",
                output_paths=outputs,
                elapsed_time=time.time() - start,
                cached=True,
            )

        # ── Resample ──────────────────────────────────────────────────────
        t_resample = time.time()
        if not resample_path.exists() or req.force:
            if req.resample_path:
                pass  # caller provided pre-resampled file
            else:
                logger.info("Resampling %s …", input_path.name)
                _resample(input_path, resample_path)
        t_resample_done = time.time() - t_resample

        # ── SynthSeg inference (GPU-exclusive) ──
        logger.info(
            "SynthSeg %s start: study=%s seq=%s parc=%s pp=%s",
            _BACKEND, req.study_id, basename, req.run_parcellation, req.post_process,
        )
        t_infer = time.time()

        with _gpu_lock:
            if _BACKEND == "pytorch":
                _run_synthseg_pytorch(resample_path, out_synthseg, out_synthseg33, req.run_parcellation)
            elif _BACKEND == "onnx":
                _run_synthseg_onnx(resample_path, out_synthseg, out_synthseg33, req.run_parcellation)
            else:
                _run_synthseg_tf(resample_path, out_synthseg, out_synthseg33, req.run_parcellation)

            if _BACKEND == "pytorch":
                import torch
                torch.cuda.empty_cache()

        t_infer_done = time.time() - t_infer
        logger.info("SynthSeg inference done in %.1f s (GPU released)", t_infer_done)

        # ── gpu_only phase: return here (caller handles postprocess locally) ──
        if req.phase == "gpu_only":
            elapsed = time.time() - start
            logger.info("gpu_only phase done in %.1f s (GPU released) → %s", elapsed, out_synthseg33)
            outputs = {"resample": str(resample_path), "synthseg33": str(out_synthseg33)}
            if req.run_parcellation:
                outputs["synthseg"] = str(out_synthseg)
            return PredictResponse(
                status="ok",
                output_paths=outputs,
                elapsed_time=elapsed,
                timing={"resample_s": round(t_resample_done, 2), "inference_s": round(t_infer_done, 2)},
            )

        # ── WM parcellation postprocessing (CPU only, GPU now free) ──────
        post_outputs: dict = {}
        t_pp_done = 0.0
        if req.post_process != "none" and req.run_parcellation:
            t_pp = time.time()
            try:
                post_outputs = _compute_post_process(
                    out_synthseg, out_synthseg33, req.post_process
                )
                post_outputs = {k: str(v) for k, v in post_outputs.items()}
                t_pp_done = time.time() - t_pp
                logger.info("Post-process done in %.1f s: %s", t_pp_done, list(post_outputs))
            except Exception:
                t_pp_done = time.time() - t_pp
                logger.exception(
                    "Post-process failed (core synthseg outputs still saved): "
                    "study=%s seq=%s pp=%s",
                    req.study_id, Path(req.input_path).name, req.post_process,
                )

        # ── Resample synthseg back to original input space (CPU, non-blocking) ──
        if req.run_parcellation and out_synthseg.exists():
            try:
                from nibabel.processing import resample_from_to
                ref_nii = nib.load(str(input_path))
                seg_nii = nib.load(str(out_synthseg))
                resampled = resample_from_to(seg_nii, ref_nii, order=0, cval=0)
                nib.save(resampled, str(out_synthseg33_native))
                logger.info("SynthSeg original space saved: %s", out_synthseg33_native.name)
                wmh_pvs_path = post_outputs.get("WMH_PVS")
                if wmh_pvs_path and Path(wmh_pvs_path).exists():
                    wmh_pvs_orig = output_dir / f"{basename}_WMH_PVS_orig.nii.gz"
                    wmh_nii = nib.load(wmh_pvs_path)
                    wmh_resampled = resample_from_to(wmh_nii, ref_nii, order=0, cval=0)
                    nib.save(wmh_resampled, str(wmh_pvs_orig))
                    post_outputs["WMH_PVS_orig"] = str(wmh_pvs_orig)
                    logger.info("WMH_PVS original space saved: %s", wmh_pvs_orig.name)
            except Exception:
                logger.warning(
                    "Failed to produce synthseg_orig for %s (detectors will run own SynthSeg)",
                    basename, exc_info=True,
                )

        elapsed = time.time() - start
        logger.info("Total %.1f s → %s", elapsed, out_synthseg33)

        outputs = {
            "resample": str(resample_path),
            "synthseg33": str(out_synthseg33),
            **post_outputs,
        }
        if req.run_parcellation:
            outputs["synthseg"] = str(out_synthseg)
        if out_synthseg33_native.exists():
            outputs["synthseg33_native"] = str(out_synthseg33_native)

        return PredictResponse(
            status="ok",
            output_paths=outputs,
            elapsed_time=elapsed,
            timing={
                "resample_s": round(t_resample_done, 2),
                "inference_s": round(t_infer_done, 2),
                "postprocess_s": round(t_pp_done, 2),
                "total_s": round(elapsed, 2),
            },
        )

    except Exception:
        logger.exception("SynthSeg predict failed: study=%s", req.study_id)
        return PredictResponse(
            status="error",
            error_msg="Internal error — check server logs",
            elapsed_time=time.time() - start,
        )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Synchronous endpoint — blocks until fully complete (backward compatible)."""
    result = _do_predict(req)
    _check_rss_and_restart()
    return result


def _run_job(job_id: str, req: PredictRequest) -> None:
    """Thread worker for /predict/async jobs."""
    _jobs[job_id]["status"] = "running"
    result = _do_predict(req)
    _jobs[job_id].update({
        "status": "done" if result.status == "ok" else "error",
        "output_paths": result.output_paths,
        "elapsed_time": result.elapsed_time,
        "timing": result.timing,
        "error_msg": result.error_msg,
        "cached": result.cached,
    })


@app.post("/predict/async", response_model=AsyncPredictResponse)
def predict_async(req: PredictRequest) -> AsyncPredictResponse:
    """Non-blocking endpoint — returns job_id immediately; poll GET /status/{job_id}."""
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status": "queued",
        "created_at": time.time(),
        "output_paths": {},
        "elapsed_time": 0.0,
        "timing": {},
        "error_msg": "",
        "cached": False,
    }
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True, name=f"job-{job_id}").start()
    logger.info("Async job %s queued: study=%s seq=%s", job_id, req.study_id, req.input_path)
    return AsyncPredictResponse(job_id=job_id, status="queued")


@app.get("/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Poll async job status. Returns status=done|running|queued|error + output_paths when done."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatusResponse(job_id=job_id, **{k: v for k, v in job.items() if k != "created_at"})


@app.delete("/cache")
def clear_cache(req: CacheDeleteRequest):
    """Remove all SynthSeg outputs for a study so they will be recomputed."""
    output_dir = Path(req.output_dir)
    deleted: List[str] = []
    for p in output_dir.glob("*_synthseg*.nii.gz"):
        p.unlink(missing_ok=True)
        deleted.append(p.name)
    for p in output_dir.glob("*_resample.nii.gz"):
        p.unlink(missing_ok=True)
        deleted.append(p.name)
    return {"status": "ok", "deleted": deleted}
