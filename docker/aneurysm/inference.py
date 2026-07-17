#!/usr/bin/env python3
"""Aneurysm GPU Inference — 3 nnUNet stages + Vessel 16-label + MIP.

Internalized from Chuan's gpu_aneurysm.py (855 lines).
Zero Chuan code dependencies.

Stages:
  A: Brain Detection (Dataset134, 3D fullres) → brain mask
  B: Vessel Segmentation (Dataset135, 3D fullres) → vessel mask
  B2: Vessel 16-Label (ONNX, ~0.4s)
  C: Aneurysm Detection (Dataset080, 3D fullres) → Pred
  D: Reslice + DICOM decompress + MIP (GPU)

GPU memory: each nnUNet predict auto-cleans via nnunet_predict.predict_3d.
"""

# AP-098 fix (CP4-C): 強制所有 multiprocessing 用 spawn,不用 fork。
# 必須在 import torch / nnUNet 之前 call。fork-after-CUDA-init 在
# PyTorch DataLoader (nnUNet predict_3d 內部用) 跟 batchgenerators MTA 內會
# deadlock — child process 繼承 broken CUDA state → torch.from_numpy() 永久卡住。
# CP4-A (v16 model load 後移) 只 delay CUDA init 到 Stage B2,但 nnUNet predict_3d
# 自己會 init CUDA + fork worker,所以 v16 後移無法解 root cause,必須改 spawn。
# 詳:pipeline-error-log.md AP-098 + step_error_recovery_ux_cp4_plan_v2.md
import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

import argparse
import gc
import logging
import os
import shutil
import sys
import time

import nibabel as nib
import numpy as np

from code_ai.inference.nnunet_predict import predict_3d
from code_ai.inference.aneurysm_helpers import (
    load_volume,
    get_vessel_seed,
    combine_vessel_brain,
    predict_vessel_16labels,
    modify_vessel_16labels,
    custom_normalize_1,
    resize_volume,
    filter_aneurysm,
)
from nifti_utils import flip_to_cnn, flip_to_native, nii_replace
# Stage D (reslice/dcm2niix/MIP) 走 subprocess,不需要 in-process import mip_utils

logger = logging.getLogger("aneurysm.inference")


# ── Vessel16 warm state (Step 2 in-process refactor) ─────────────────────
# 200MB model 常駐 GPU (per user spec: 太小不值得 CPU-swap),跨 case 復用。
# In-process server.py 啟動時呼叫 warmup_vessel16() 一次;subprocess mode
# fallback 到 aneurysm_inference 內 lazy 載入 (每 case 重載,同原行為)。
_VESSEL16_STATE = None  # (kind, model_or_sess, device_or_input_name)


def warmup_vessel16() -> None:
    """Load vessel 16-label model to GPU (常駐, ~200MB VRAM)."""
    global _VESSEL16_STATE
    if _VESSEL16_STATE is not None:
        return

    onnx_dir = os.environ.get("VESSEL_16_ONNX_DIR", "/data/4TB/ai_pipeline/onnx_models/aneurysm")
    pt_path = os.path.join(onnx_dir, "vessel_16label_resunet_traced.pt")
    prefer = os.environ.get("VESSEL_16_BACKEND", "auto").lower()

    if prefer in ("torch", "auto") and os.path.isfile(pt_path):
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.jit.load(pt_path, map_location=device).eval()
        _VESSEL16_STATE = ("torch", model, device)
        logger.info("[vessel16] warmed to %s (PyTorch traced .pt)", device)
    else:
        import onnxruntime as ort
        onnx_path = os.path.join(onnx_dir, "vessel_16label_resunet.onnx")
        sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        _VESSEL16_STATE = ("onnx", sess, sess.get_inputs()[0].name)
        logger.info("[vessel16] warmed (ONNX Runtime CUDA)")


def _get_vessel16_serve():
    """Return an object with .serve(x) callable (bound to warm model)."""
    if _VESSEL16_STATE is None:
        warmup_vessel16()  # lazy: subprocess mode falls here
    kind, model_or_sess, device_or_name = _VESSEL16_STATE

    if kind == "torch":
        import torch
        _model = model_or_sess
        _device = device_or_name
        class TorchModelWrapper:
            def serve(self, x):
                class R:
                    def __init__(s, v): s._v = v
                    def numpy(s): return s._v
                with torch.no_grad():
                    out = _model(torch.from_numpy(np.asarray(x, dtype=np.float32)).to(_device))
                return R(out.cpu().numpy())
        return TorchModelWrapper()
    else:
        _sess = model_or_sess
        _inp_name = device_or_name
        class OnnxModelWrapper:
            def serve(self, x):
                class R:
                    def __init__(s, v): s._v = v
                    def numpy(s): return s._v
                return R(_sess.run(None, {_inp_name: np.asarray(x, dtype=np.float32)})[0])
        return OnnxModelWrapper()


def _save_nii_brain(path_process: str, image_arr, out_dir: str):
    """Prepare nnUNet input for brain detection: Image + Ones mask."""
    for d in ["Image", "Ones"]:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    img = nib.load(os.path.join(path_process, "MRA_BRAIN.nii.gz"))
    img = nib.as_closest_canonical(img)
    aff, hdr = img.affine, img.header

    img_dest = os.path.join(out_dir, "Image", "DeepAneurysm_00001_0000.nii.gz")
    if not os.path.isfile(img_dest):
        shutil.copy(os.path.join(path_process, "MRA_BRAIN.nii.gz"), img_dest)

    ones = np.ones_like(image_arr, dtype=np.int16)[::-1, ::-1, :]
    nib.save(nib.nifti1.Nifti1Image(ones, affine=aff, header=hdr),
             os.path.join(out_dir, "Ones", "DeepAneurysm_00001_0000.nii.gz"))


def _save_nii_vessel(path_process: str, image_arr, vessel_mask, vessel_16labels, out_dir: str):
    """Prepare nnUNet input for aneurysm detection: Normalized Image + Vessel mask."""
    for d in ["Normalized_Image"]:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    img = nib.load(os.path.join(path_process, "MRA_BRAIN.nii.gz"))
    aff, hdr = img.affine, img.header
    original_size = tuple(img.header.get_data_shape())
    img = nib.as_closest_canonical(img)

    if image_arr.shape != original_size:
        image_arr = resize_volume(image_arr, target_size=original_size, dtype="uint8")
    if vessel_mask.shape != original_size:
        vessel_mask = resize_volume(vessel_mask, target_size=original_size, dtype="uint8")
    if vessel_16labels.shape != original_size:
        vessel_16labels = resize_volume(vessel_16labels, target_size=original_size, dtype="uint8")

    image_arr = custom_normalize_1(image_arr)
    nib.save(nib.nifti1.Nifti1Image(image_arr[::-1, ::-1, :], affine=aff, header=hdr),
             os.path.join(out_dir, "Normalized_Image", "DeepAneurysm_00001_0000.nii.gz"))
    nib.save(nib.nifti1.Nifti1Image(vessel_mask[::-1, ::-1, :].astype("uint8"), affine=aff, header=hdr),
             os.path.join(out_dir, "Vessel.nii.gz"))
    nib.save(nib.nifti1.Nifti1Image(vessel_16labels[::-1, ::-1, :].astype("uint8"), affine=aff, header=hdr),
             os.path.join(out_dir, "Vessel_16.nii.gz"))



# --- Stage A helper: keep only largest connected component of brain mask ---
# Wide-FOV TOF (e.g. 台大) sometimes has brain seg outputting stray blobs
# (neck / eye socket / cavernous sinus) as separate CCs. Keep only the
# largest so downstream vessel-gate + aneurysm filter stay clean.
# Env: ANEURYSM_BRAIN_LARGEST_CC=1 (default 1), set 0 to skip.
def _keep_largest_cc(mask):
    from scipy.ndimage import label
    m = (mask > 0).astype("uint8")
    labeled, n = label(m)
    if n <= 1:
        return mask, {"n_components": int(n), "kept_size": int(m.sum()),
                      "dropped_size": 0}
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    biggest = int(sizes.argmax())
    kept = (labeled == biggest)
    total = int(m.sum())
    kept_size = int(kept.sum())
    cleaned = np.where(kept, mask, 0).astype(mask.dtype)
    return cleaned, {"n_components": int(n),
                     "kept_size": kept_size,
                     "dropped_size": total - kept_size}


# --- Stage C.5 helper: SynthSeg call + neck-region z-filter ---
# Cross-institution TOF cases (2014-2019 Siemens Verio 3D-TOF multi-slab)
# with cervical FOV extension confuse the aneurysm model outside its
# training distribution (雙和 train set has no neck coverage). Use
# SynthSeg to determine actual brain bottom in z, discard lesions whose
# lowest voxel sits below brain_z_min + margin.
# Env:
#   ANEURYSM_SYNTHSEG_ENABLED=1        default 1 (0 to skip entirely)
#   SYNTHSEG_URL=http://127.0.0.1:5005 same host network as inference_aneurysm
#   ANEURYSM_BRAIN_Z_MARGIN=5          slices above brain bottom edge
def _call_synthseg(process_dir, study_id, url, timeout=300.0):
    import json as _json
    from urllib.request import Request, urlopen
    out_dir = os.path.join(process_dir, "synthseg")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "input_path": os.path.join(process_dir, "MRA_BRAIN.nii.gz"),
        "output_dir": out_dir,
        "study_id": study_id,
        "run_parcellation": True,
        "force": False,
    }
    req = Request(f"{url}/predict",
                  data=_json.dumps(payload).encode("utf-8"),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        body = _json.loads(resp.read().decode("utf-8"))
    if body.get("status") != "ok":
        raise RuntimeError(
            f"synthseg status={body.get('status')} err={body.get('error_msg')}")
    return body.get("output_paths", {})


def _resolve_si_axis(affine):
    """Return (voxel_axis, sign) where voxel_axis is the array axis (0/1/2)
    corresponding to S/I direction, and sign is +1 if voxel_axis+1 points to
    Superior, -1 if it points to Inferior. Determined from the NIfTI affine.
    """
    ornt = nib.orientations.io_orientation(affine)
    # ornt is (3, 2); row i = [ras_axis, sign] for voxel axis i.
    # We want the voxel axis whose ras_axis == 2 (S/I).
    for voxel_axis, (ras_axis, sign) in enumerate(ornt):
        if int(ras_axis) == 2:
            return voxel_axis, int(sign)
    raise RuntimeError("cannot resolve S/I axis from affine")


def _apply_neck_filter(process_dir, synthseg_native_path, z_margin):
    """Drop Pred.nii.gz lesions sitting at/below the SynthSeg-defined brain
    bottom (+ z_margin slices). Orientation-aware: uses the affine to
    determine which voxel axis is S/I and which end is inferior. Discards
    lesions whose minimum coordinate along the inferior direction sits within
    z_margin of the brain edge on that side.
    """
    path_nnunet = os.path.join(process_dir, "nnUNet")
    pred_path = os.path.join(path_nnunet, "Pred.nii.gz")
    if not os.path.isfile(pred_path) or not os.path.isfile(synthseg_native_path):
        return {"dropped": [], "kept": [], "reason": "inputs missing"}

    pred_img = nib.load(pred_path)
    pred_arr = np.array(pred_img.dataobj)
    syn_img = nib.load(synthseg_native_path)
    syn_arr = np.asarray(syn_img.dataobj)

    # Resolve S/I axis. Trust pred affine (it should match synthseg since
    # synthseg was run on the same MRA_BRAIN volume with same header).
    si_axis, si_sign = _resolve_si_axis(pred_img.affine)

    # Reduce synthseg to a 1-D presence array along si_axis (True where any
    # brain label present at that slice).
    other = tuple(a for a in (0, 1, 2) if a != si_axis)
    syn_slice_has_brain = (syn_arr > 0).sum(axis=other) > 0
    syn_idx = np.where(syn_slice_has_brain)[0]
    if len(syn_idx) == 0:
        return {"dropped": [], "kept": [], "reason": "synthseg empty"}

    margin = int(z_margin)
    if si_sign > 0:
        # voxel_axis+1 -> Superior. Inferior end = index 0 side.
        # brain bottom (inferior extent) = min index of syn presence.
        brain_edge = int(syn_idx.min())
        cutoff = brain_edge + margin
        # discard rule: lesion has any voxel at index < cutoff
        def drop(lesion_idx_along_si):
            return int(lesion_idx_along_si.min()) < cutoff
        direction = "index0=inferior (voxel_axis+1=S)"
    else:
        # voxel_axis+1 -> Inferior. Inferior end = max index side.
        # brain bottom (inferior extent) = max index of syn presence.
        brain_edge = int(syn_idx.max())
        cutoff = brain_edge - margin
        # discard rule: lesion has any voxel at index > cutoff
        def drop(lesion_idx_along_si):
            return int(lesion_idx_along_si.max()) > cutoff
        direction = "indexN-1=inferior (voxel_axis+1=I)"

    dropped, kept = [], []
    for lab in np.unique(pred_arr):
        if lab == 0:
            continue
        loc = np.where(pred_arr == lab)
        if len(loc[si_axis]) == 0:
            continue
        if drop(loc[si_axis]):
            dropped.append(int(lab))
        else:
            kept.append(int(lab))

    if dropped:
        for lab in dropped:
            pred_arr[pred_arr == lab] = 0
        out = nib.Nifti1Image(pred_arr.astype(pred_img.get_data_dtype()),
                              pred_img.affine, pred_img.header)
        nib.save(out, pred_path)

    return {"dropped": dropped, "kept": kept,
            "brain_edge": brain_edge, "cutoff": cutoff,
            "si_axis": si_axis, "direction": direction}


def aneurysm_inference(
    process_dir: str,
    brain_model: str,
    vessel_model: str,
    aneurysm_model: str,
    gpu_id: int = 0,
    code_dir: str = "",
    dicom_dir: str = "",
) -> bool:
    try:
        t_total = time.time()
        path_nnunet = os.path.join(process_dir, "nnUNet")
        path_img = os.path.join(process_dir, "Image")
        path_ones = os.path.join(process_dir, "Ones")
        path_brain = os.path.join(process_dir, "Brain")
        path_vessel_dir = os.path.join(process_dir, "Vessel")
        for d in [path_nnunet, path_img, path_ones, path_brain, path_vessel_dir]:
            os.makedirs(d, exist_ok=True)

        image_arr, spacing, _ = load_volume(os.path.join(process_dir, "MRA_BRAIN.nii.gz"), dtype="int16")

        # AP-098 fix: v16 model load 移到 Stage B2 開頭 (見下方)。
        # 原本在此處 (Stage A 之前) load 會 init CUDA context,使後續 nnUNet
        # predict_3d 內部 batchgenerators MTA fork 出來的 worker 繼承 broken CUDA
        # state,worker 在 utility_transforms.py:60 torch.from_numpy().contiguous()
        # 永久 deadlock。延後 load 到 Stage A/B 之後,fork 時 CUDA 未 init,worker
        # 乾淨繼承記憶體。詳: pipeline-error-log.md AP-098

        _save_nii_brain(process_dir, image_arr, out_dir=process_dir)

        # ── Stage A: Brain Detection ─────────────────────────────────────
        t = time.time()
        logger.info("[A] Brain Detection (GPU %d)", gpu_id)
        predict_3d(path_img, path_ones, path_brain, brain_model)

        prob_nii = nib.load(os.path.join(path_brain, "DeepAneurysm_00001.nii.gz"))
        prob = flip_to_cnn(np.array(prob_nii.dataobj), prob_nii, qfac_both=True)
        brain_mask = flip_to_native((prob > 0.1).astype(int), prob_nii).astype(int)
        # Keep only the largest connected component. Stray blobs in wide-FOV
        # TOF (e.g. 台大 with neck coverage) can appear as separate CCs and
        # contaminate downstream vessel-gate + aneurysm neck-filter.
        # Toggle: ANEURYSM_BRAIN_LARGEST_CC=0
        if os.environ.get("ANEURYSM_BRAIN_LARGEST_CC", "1") == "1":
            brain_mask, _cc_stat = _keep_largest_cc(brain_mask)
            if _cc_stat["n_components"] > 1:
                logger.info(
                    "[A] largest-CC: n_components=%d kept=%d dropped=%d voxels",
                    _cc_stat["n_components"],
                    _cc_stat["kept_size"],
                    _cc_stat["dropped_size"])
        nib.save(nii_replace(prob_nii, brain_mask), os.path.join(path_brain, "DeepAneurysm_00001_0000.nii.gz"))
        shutil.copy(os.path.join(path_brain, "DeepAneurysm_00001_0000.nii.gz"),
                    os.path.join(path_nnunet, "brain_mask.nii.gz"))
        os.remove(os.path.join(path_brain, "DeepAneurysm_00001.nii.gz"))
        logger.info("[A] done (%.0fs)", time.time() - t)

        # ── Stage B: Vessel Segmentation ─────────────────────────────────
        t = time.time()
        logger.info("[B] Vessel Segmentation (GPU %d)", gpu_id)
        predict_3d(path_img, path_brain, path_vessel_dir, vessel_model, batch_size=32)

        prob_nii = nib.load(os.path.join(path_vessel_dir, "DeepAneurysm_00001.nii.gz"))
        prob = flip_to_cnn(np.array(prob_nii.dataobj), prob_nii, qfac_both=True)
        vessel_pred = prob > 0.1
        vspacing = prob_nii.header.get_zooms()[:3]
        brain_nii = nib.load(os.path.join(path_nnunet, "brain_mask.nii.gz"))
        brain = flip_to_cnn(np.array(brain_nii.dataobj), brain_nii, qfac_both=True)
        vessel_seed = get_vessel_seed(vessel_pred, mask=brain, spacing=vspacing)
        vessel_mask = combine_vessel_brain(vessel_pred, seed_mask=vessel_seed, brain_mask=brain)
        vessel_native = flip_to_native(vessel_mask, prob_nii).astype(int)
        nib.save(nii_replace(prob_nii, vessel_native), os.path.join(path_vessel_dir, "DeepAneurysm_00001_0000.nii.gz"))
        shutil.copy(os.path.join(path_vessel_dir, "DeepAneurysm_00001_0000.nii.gz"),
                    os.path.join(path_nnunet, "Vessel.nii.gz"))
        os.remove(os.path.join(path_vessel_dir, "DeepAneurysm_00001.nii.gz"))
        logger.info("[B] done (%.0fs)", time.time() - t)

        # ── Stage B2: Vessel 16-Label ──────────────────────────────────
        # In-process server 於啟動時已 warm 好 vessel16 model (常駐 GPU),
        # subprocess mode 走 _get_vessel16_serve 的 lazy fallback (原行為)。
        t = time.time()
        v16, sp16, _ = load_volume(os.path.join(path_nnunet, "Vessel.nii.gz"), dtype="int16")
        model3 = _get_vessel16_serve()

        v16labels = predict_vessel_16labels(v16, model3, sp16, verbose=True)
        v16labels, v16 = modify_vessel_16labels(v16labels, sp16)
        _save_nii_vessel(process_dir, image_arr, v16, v16labels, out_dir=process_dir)
        # 不 del model3: 是 warm state 的 wrapper 引用,實際 model 常駐
        gc.collect()
        logger.info("[B2] Vessel 16-label done (%.0fs)", time.time() - t)

        # ── Stage C: Aneurysm Detection ──────────────────────────────────
        t = time.time()
        logger.info("[C] Aneurysm Detection (GPU %d)", gpu_id)
        path_normimg = os.path.join(process_dir, "Normalized_Image")
        predict_3d(path_normimg, path_vessel_dir, process_dir, aneurysm_model,
                   folds=(13,), checkpoint="checkpoint_best.pth",
                   plans_json="nnUNetPlans_5L-b900.json", batch_size=112)

        shutil.copy(os.path.join(process_dir, "DeepAneurysm_00001.nii.gz"),
                    os.path.join(path_nnunet, "Prob.nii.gz"))
        prob_nii = nib.load(os.path.join(path_nnunet, "Prob.nii.gz"))
        prob = flip_to_cnn(np.array(prob_nii.dataobj), prob_nii, qfac_both=True)
        pixdim = prob_nii.header.copy()["pixdim"]
        _, _, pred_label = filter_aneurysm(prob, [pixdim[1], pixdim[2]], conf_th=0.1, min_diameter=2, top_k=4, obj_th=0.73)
        pred_label = flip_to_native(pred_label, prob_nii).astype(int)
        nib.save(nii_replace(prob_nii, pred_label), os.path.join(path_nnunet, "Pred.nii.gz"))
        logger.info("[C] done (%.0fs)", time.time() - t)

        # --- Stage C.5: SynthSeg + neck filter ---
        # Cross-institution TOF cases (e.g. 台大 wide-FOV) with cervical
        # extension confuse the aneurysm model outside training distribution.
        # Filter using SynthSeg-determined brain bottom.
        if os.environ.get("ANEURYSM_SYNTHSEG_ENABLED", "1") == "1":
            t_syn = time.time()
            _syn_url = os.environ.get("SYNTHSEG_URL", "http://127.0.0.1:5005")
            _z_margin = int(os.environ.get("ANEURYSM_BRAIN_Z_MARGIN", "5"))
            try:
                _paths = _call_synthseg(process_dir,
                                        os.path.basename(process_dir),
                                        _syn_url)
                _native = _paths.get("synthseg33_native")
                if _native:
                    _dst = os.path.join(path_nnunet, "SynthSEG.nii.gz")
                    if not os.path.exists(_dst):
                        shutil.copy(_native, _dst)
                    _stat = _apply_neck_filter(process_dir, _native, _z_margin)
                    logger.info(
                        "[C.5] synthseg+neck-filter done (%.0fs) si_axis=%s dir=%s brain_edge=%s cutoff=%s dropped=%s kept=%s",
                        time.time() - t_syn,
                        _stat.get("si_axis"),
                        _stat.get("direction"),
                        _stat.get("brain_edge"),
                        _stat.get("cutoff"),
                        _stat.get("dropped"),
                        _stat.get("kept"))
                else:
                    logger.warning("[C.5] synthseg returned no native output; skip filter")
            except Exception as _exc:
                logger.warning("[C.5] synthseg/neck-filter failed (non-fatal, keep original Pred): %s", _exc)

        elapsed_nnunet = time.time() - t_total

        if not code_dir:
            code_dir = os.environ.get("RADX_CODE_ROOT", "")

        # ── Stage D: Reslice + MIP ───────────────────────────────────────
        # NNUNET_MIP_SUBPROCESS=1 → 走獨立 subprocess (省 VRAM,適合 TRT mode)
        # 預設 0 → in-process (原路徑,適合 warm mode; warm + subprocess 會 OOM)
        _mip_subprocess = os.environ.get("NNUNET_MIP_SUBPROCESS", "0") == "1"

        if _mip_subprocess:
            # subprocess mode: exit 時 GPU 記憶體全還 driver
            import subprocess
            t_mip = time.time()
            cmd = ["python", "/app/mip_stage.py",
                   "--process_dir", process_dir,
                   "--gpu_id", str(gpu_id),
                   "--dicom_dir", dicom_dir]
            if code_dir:
                cmd += ["--code_dir", code_dir]
            try:
                result = subprocess.run(cmd, timeout=600)
            except subprocess.TimeoutExpired:
                logger.error("[D] MIP subprocess TIMEOUT after 600s")
                return False
            if result.returncode != 0:
                logger.error("[D] MIP subprocess failed (rc=%d)", result.returncode)
                return False
            logger.info("[D] Stage D subprocess done (%.0fs)", time.time() - t_mip)
        else:
            # in-process mode (原本行為)
            from mip_utils import reslice_nifti_pred_nobrain, decompress_dicom_with_gdcm, create_MIP_pred

            path_nii = os.path.join(path_nnunet, "Image_nii")
            path_dcm = os.path.join(path_nnunet, "Dicom")
            path_reslice = os.path.join(path_nnunet, "Image_reslice")
            for d in [path_nii, path_dcm, path_reslice]:
                os.makedirs(d, exist_ok=True)

            for src, dest_dir, name in [
                (os.path.join(path_nnunet, "Pred.nii.gz"), path_nii, "Pred.nii.gz"),
                (os.path.join(process_dir, "Vessel.nii.gz"), path_nii, "Vessel.nii.gz"),
                (os.path.join(process_dir, "Vessel_16.nii.gz"), path_nnunet, "Vessel_16.nii.gz"),
            ]:
                dest = os.path.join(dest_dir, name)
                if os.path.isfile(src) and not os.path.isfile(dest):
                    shutil.copy(src, dest)

            if dicom_dir and os.path.isdir(dicom_dir):
                dcm_dest = os.path.join(path_dcm, "MRA_BRAIN")
                if not os.path.isdir(dcm_dest):
                    shutil.copytree(dicom_dir, dcm_dest)

            t = time.time()
            reslice_nifti_pred_nobrain(path_nii, path_reslice)
            logger.info("[D] Reslice done (%.0fs)", time.time() - t)

            t = time.time()
            decompress_dicom_with_gdcm(path_dcm)
            logger.info("[D] DICOM decompress done (%.0fs)", time.time() - t)

            t = time.time()
            path_png = os.path.join(code_dir, "png") if code_dir else "/tmp/png"
            create_MIP_pred(path_dcm, path_reslice, path_png, gpu_id)
            logger.info("[D] MIP done (%.0fs)", time.time() - t)

            for name in ["MIP_Pitch_pred.nii.gz", "MIP_Yaw_pred.nii.gz"]:
                src = os.path.join(path_reslice, name)
                if os.path.isfile(src):
                    shutil.copy(src, os.path.join(path_nnunet, name))

        total = time.time() - t_total
        logger.info("All done (%.0fs total: nnUNet=%.0fs, MIP=%.0fs)", total, elapsed_nnunet, total - elapsed_nnunet)
        return True

    except Exception as exc:
        logger.error("[inference] Failed: %s", exc, exc_info=True)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--process_dir", required=True)
    p.add_argument("--brain_model", required=True)
    p.add_argument("--vessel_model", required=True)
    p.add_argument("--aneurysm_model", required=True)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--code_dir", default="")
    p.add_argument("--dicom_dir", default="")
    a = p.parse_args()
    sys.exit(0 if aneurysm_inference(a.process_dir, a.brain_model, a.vessel_model, a.aneurysm_model, a.gpu_id, a.code_dir, a.dicom_dir) else 1)
