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
from mip_utils import reslice_nifti_pred_nobrain, decompress_dicom_with_gdcm, create_MIP_pred

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

        elapsed_nnunet = time.time() - t_total

        # ── Stage D: Reslice + MIP ───────────────────────────────────────
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
        if not code_dir:
            code_dir = os.environ.get("RADX_CODE_ROOT", "")
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
