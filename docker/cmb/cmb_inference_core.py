"""CMB Inference Core — sliding window detection + FP filtering.

Rewritten from scratch. Replaces CMBServiceTF (735 lines).

Algorithm:
  1. Load SWAN → resize to isotropic → normalize with brain mask
  2. Model1: 64³ sliding window → per-voxel detection probability
  3. Connected components → candidate lesions
  4. Model2: 26³ cube per candidate → FP confidence filter
  5. Remap surviving lesions → Pred label map + JSON stats
"""

import gc
import itertools
import json
import logging
import os

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import windows as scipy_windows
from skimage.measure import label, regionprops

logger = logging.getLogger("cmb.core")

# Model paths
DEFAULT_MODEL_DIR = "/data/4TB/ai_pipeline/onnx_models/cmb"
MODEL1_ONNX = "cmb_model1_unet5_32.onnx"
MODEL1_PT = "cmb_model1_unet5_32_traced.pt"
MODEL2_ONNX = "cmb_model2_resunet_fpn.onnx"
MODEL2_PT = "cmb_model2_resunet_fpn_traced.pt"

# Detection parameters (must match training settings exactly)
PATCH_SIZE = (64, 64, 64)
OVERLAP = 0.5
CANDIDATE_THRESHOLD = 0.05
MIN_CMB_PROB = 0.084
FP_REDUCTION_TH = 0.357
UNCERTAIN_TH = 0.5175
BATCH_SIZE = 8

# SynthSeg label → brain region name (matches Chuan's CMBServiceTF.label_index_name_mapping_dict)
REGION_NAMES = {
    0: "Background", 1: "CSF", 301: "brainstem",
    102: "L. cerebellum", 103: "L. basal ganglion", 104: "L. thalamus",
    105: "L. internal capsule", 106: "L. external capsule", 107: "L. corpus callosum",
    108: "L. DPWM", 109: "L. frontal", 110: "L. parietal",
    111: "L. occipital", 112: "L. temporal", 113: "L. insular", 114: "L. cingulate",
    202: "R. cerebellum", 203: "R. basal ganglion", 204: "R. thalamus",
    205: "R. internal capsule", 206: "R. external capsule", 207: "R. corpus callosum",
    208: "R. DPWM", 209: "R. frontal", 210: "R. parietal",
    211: "R. occipital", 212: "R. temporal", 213: "R. insular", 214: "R. cingulate",
}


# ── Public API ────────────────────────────────────────────────────────────────

def cmb_detect(
    swan_path: str,
    synthseg_path: str,
    output_nii_path: str,
    output_json_path: str,
    model_dir: str = "",
) -> str:
    """Run full CMB detection pipeline.

    Returns output_nii_path on success, empty string on failure.
    """
    if not model_dir:
        model_dir = os.environ.get("CMB_ONNX_MODEL_DIR", DEFAULT_MODEL_DIR)

    # ── 1. Load + preprocess ─────────────────────────────────────────
    swan_nii = nib.as_closest_canonical(nib.load(swan_path))
    swan_arr = np.squeeze(swan_nii.get_fdata()).astype(np.float32)
    swan_arr = swan_arr[::-1, ::-1, :]  # RAS → LPS (model trained in LPS)
    spacing = list(swan_nii.header.get_zooms()[:3])

    # Isotropic resize
    target_spacing = np.array([spacing[0], spacing[1], min(spacing)])
    if target_spacing[0] > 0.6:
        target_spacing = target_spacing / 2
    swan_iso = ndi.zoom(swan_arr, np.array(spacing) / target_spacing,
                        order=1, prefilter=True, grid_mode=False).astype(np.float32)

    # Brain mask from SynthSeg (also canonical + LPS)
    seg_nii = nib.as_closest_canonical(nib.load(synthseg_path))
    seg_arr = np.squeeze(seg_nii.get_fdata()).astype(np.uint16)[::-1, ::-1, :]
    brain_mask = (ndi.zoom(seg_arr, np.array(swan_iso.shape) / np.array(seg_arr.shape),
                           order=0, prefilter=False, grid_mode=False) > 0)

    swan_norm = _normalize(swan_iso, brain_mask)

    # ── 2. Model1: sliding window detection ──────────────────────────
    logger.info("Model1: sliding window %s on volume %s", PATCH_SIZE, swan_norm.shape)
    model1_fn = _load_model(model_dir, MODEL1_PT, MODEL1_ONNX)
    prob_map = _sliding_window(swan_norm, model1_fn, PATCH_SIZE, OVERLAP, brain_mask) * brain_mask
    del model1_fn
    gc.collect()
    logger.info("Model1 done. prob range: [%.3f, %.3f]", prob_map.min(), prob_map.max())

    # ── 3. Connected components → candidates ─────────────────────────
    labeled, n_candidates = label(prob_map > CANDIDATE_THRESHOLD, return_num=True)
    logger.info("Candidates: %d (threshold=%.2f)", n_candidates, CANDIDATE_THRESHOLD)

    if n_candidates == 0:
        _save_empty(swan_nii, output_nii_path, output_json_path)
        return output_nii_path

    # ── 4. Model2: per-candidate FP filtering ────────────────────────
    model2_fn = _load_model(model_dir, MODEL2_PT, MODEL2_ONNX)
    regions = regionprops(labeled, prob_map)
    tp_confs = _score_candidates(regions, swan_norm, model2_fn)
    del model2_fn
    gc.collect()

    # ── 5. Filter + relabel (matching original object_analysis exactly) ──
    # Resize seg_arr to match labeled (isotropic) for location lookup
    seg_iso = ndi.zoom(seg_arr, np.array(swan_iso.shape) / np.array(seg_arr.shape),
                       order=0, prefilter=False, grid_mode=False).astype(np.uint16)
    result_label, lesion_stats, n_kept = _filter_and_relabel(
        regions, tp_confs, labeled, target_spacing, seg_iso)

    logger.info("After FP filter: %d lesions (from %d candidates)", n_kept, n_candidates)

    # ── 6. Save results ──────────────────────────────────────────────
    result_orig = ndi.zoom(result_label.astype(np.uint16),
                           np.array(swan_arr.shape) / np.array(result_label.shape),
                           order=0, prefilter=False, grid_mode=False).astype(np.uint16)
    result_orig = result_orig[:swan_arr.shape[0], :swan_arr.shape[1], :swan_arr.shape[2]]
    result_orig = result_orig[::-1, ::-1, :]  # LPS → RAS

    os.makedirs(os.path.dirname(output_nii_path), exist_ok=True)
    nib.save(nib.Nifti1Image(result_orig.astype(np.int16), swan_nii.affine, swan_nii.header),
             output_nii_path)
    _save_json(output_json_path, lesion_stats, n_kept)
    logger.info("Saved: %s (%d lesions)", output_nii_path, n_kept)
    return output_nii_path


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(model_dir, pt_name, onnx_name):
    """Load PyTorch TorchScript (preferred) or ONNX model. Returns callable(batch_5d)."""
    pt_path = os.path.join(model_dir, pt_name)
    if os.path.isfile(pt_path):
        import torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.jit.load(pt_path, map_location=device).eval()
        def run(batch_5d):
            with torch.no_grad():
                return model(torch.from_numpy(batch_5d).float().to(device)).cpu().numpy()
        return run

    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(model_dir, onnx_name),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    def run(batch_5d):
        return sess.run(None, {input_name: batch_5d.astype(np.float32)})[0]
    return run


# ── Sliding window ────────────────────────────────────────────────────────────

def _sliding_window(volume, model_fn, patch_size, overlap, mask=None):
    """3D sliding window with Gaussian blending + batch inference.

    Matches original _sliding_window_np logic:
      - Pad volume to >= patch_size (centered)
      - overlap_inside window positioning
      - Gaussian importance weighting
      - output_weight_sum initialized to ones (original's convention)
      - Model input/output: [N, 64, 64, 64, 1] (5D)
    """
    ps = np.array(patch_size)
    padded_size = tuple(max(volume.shape[i], ps[i]) for i in range(3))
    padding = [px - ix for px, ix in zip(padded_size, volume.shape)]
    pad_before = [p // 2 for p in padding]
    pad_after = [p - p // 2 for p in padding]
    pad_width = list(zip(pad_before, pad_after))

    vol_padded = np.pad(volume, pad_width, mode="constant")
    mask_padded = np.pad(mask, pad_width, mode="constant").astype(bool) if mask is not None else None

    output_sum = np.zeros(padded_size, dtype=np.float32)
    output_weight = np.ones(padded_size, dtype=np.float32)
    importance = _gaussian_kernel(ps[0])

    positions = _window_positions(padded_size, tuple(ps), overlap)
    batch_patches, batch_positions = [], []
    n_computed = 0

    def flush():
        nonlocal n_computed
        if not batch_patches:
            return
        batch = np.stack(batch_patches, axis=0)[:, :, :, :, np.newaxis].astype(np.float32)
        preds = model_fn(batch)[:, :, :, :, 0]
        for i, (z, y, x) in enumerate(batch_positions):
            output_sum[z:z+ps[0], y:y+ps[1], x:x+ps[2]] += preds[i] * importance
            output_weight[z:z+ps[0], y:y+ps[1], x:x+ps[2]] += importance
            n_computed += 1
        batch_patches.clear()
        batch_positions.clear()

    for pos in positions:
        z, y, x = pos
        if mask_padded is not None and not mask_padded[z:z+ps[0], y:y+ps[1], x:x+ps[2]].any():
            continue
        batch_patches.append(vol_padded[z:z+ps[0], y:y+ps[1], x:x+ps[2]])
        batch_positions.append(pos)
        if len(batch_patches) >= BATCH_SIZE:
            flush()
    flush()

    logger.info("Sliding window: %d/%d computed", n_computed, len(positions))
    result = output_sum / output_weight
    crop = tuple(slice(pb, pb + s) for pb, s in zip(pad_before, volume.shape))
    return result[crop]


def _window_positions(image_size, roi_size, overlap):
    """Generate sliding window start positions with overlap_inside strategy."""
    dim_starts = []
    for img_s, roi_s in zip(image_size, roi_size):
        step = roi_s if roi_s == img_s else int(roi_s * (1 - overlap))
        starts = list(range(0, img_s - roi_s + 1, step))
        if starts[-1] + roi_s < img_s:
            starts.append(img_s - roi_s)
        dim_starts.append(starts)
    return list(itertools.product(*dim_starts))


# ── Candidate scoring ─────────────────────────────────────────────────────────

def _score_candidates(regions, swan_norm, model2_fn):
    """Score each candidate lesion with Model2 FP filter."""
    gaussian_26 = _gaussian_kernel(26).astype(np.float16)
    tp_confs = []
    for region in regions:
        x0, y0, z0, x1, y1, z1 = region.bbox
        center = [(x0 + x1) // 2, (y0 + y1) // 2, (z0 + z1) // 2]
        cube = _crop_cube_padded(swan_norm, center, 26)
        inp = np.stack([cube, gaussian_26], axis=-1)[np.newaxis].astype(np.float32)
        raw_conf = float(model2_fn(inp).ravel()[0])
        tp_confs.append(raw_conf / 2.0)
    return np.array(tp_confs, dtype=np.float32)


def _filter_and_relabel(regions, tp_confs, labeled, target_spacing, seg_arr=None):
    """Filter by CMB_prob, sort descending, relabel 1..N.

    Matches original object_analysis exactly:
    - CMB_prob > MIN_CMB_PROB → keep (including "other")
    - Sort by CMB_prob descending → relabel
    - CMB_prob < FP_REDUCTION_TH → classify as "other" but KEEP in label map
    """
    cmb_probs = []
    for i, region in enumerate(regions):
        pred_mean = float(region.intensity_mean) / 0.6
        cmb_probs.append((pred_mean + tp_confs[i]) / 2.0)

    indices = [i for i, p in enumerate(cmb_probs) if p > MIN_CMB_PROB]
    indices.sort(key=lambda i: cmb_probs[i], reverse=True)

    result_label = np.zeros_like(labeled, dtype=np.int32)
    lesion_stats = []

    for new_label, i in enumerate(indices, 1):
        region = regions[i]
        cmb_prob = cmb_probs[i]

        if cmb_prob < FP_REDUCTION_TH:
            cls = "other"
        elif tp_confs[i] * 2 > UNCERTAIN_TH:
            cls = "CMB"
        else:
            cls = "Uncertain"

        # Location from SynthSeg parcellation
        region_id = 0
        region_name = ""
        if seg_arr is not None:
            voxels = seg_arr[labeled == region.label]
            if len(voxels) > 0:
                counts = np.bincount(voxels.ravel())
                if len(counts) > 1:
                    counts[0] = 0  # skip Background label
                region_id = int(np.argmax(counts))
                region_name = REGION_NAMES.get(region_id, "")

        result_label[labeled == region.label] = new_label
        bbox = region.bbox
        diameter = ((bbox[3]-bbox[0])*target_spacing[0] + (bbox[4]-bbox[1])*target_spacing[1]) / 2

        lesion_stats.append({
            "index": new_label,
            "cmb_prob": round(float(cmb_prob), 3),
            "tp_conf": round(float(tp_confs[i]), 3),
            "prob_mean": round(float(region.intensity_mean), 3),
            "diameter_mm": round(float(diameter), 1),
            "voxels": int(region.area),
            "class": cls,
            "type": f"C{region_id}",
            "type_name": region_name,
        })

    return result_label, lesion_stats, len(indices)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _normalize(volume, mask, new_min=0.0, new_max=1.0, pmin=0.5, pmax=99.5):
    """Percentile-based normalization within brain mask."""
    out = volume.copy().astype(np.float32)
    intensities = out[mask].ravel()
    intensities = intensities[intensities > 0]
    if len(intensities) == 0:
        return np.zeros_like(out)
    lo, hi = np.percentile(intensities, pmin), np.percentile(intensities, pmax)
    out = np.clip(out, lo, hi)
    if hi <= lo:
        return np.zeros_like(out)
    return (new_min + (out - lo) / (hi - lo) * (new_max - new_min)).astype(np.float32)


def _gaussian_kernel(roi_size=64, sigma=0.125):
    """3D Gaussian importance kernel — scipy.signal.windows → outer → cube root."""
    std = sigma * roi_size  # 0.125 * 64 = 8.0
    g1d = scipy_windows.gaussian(roi_size, std=std)
    kernel = g1d.copy()
    for _ in range(2):
        kernel = np.outer(kernel, g1d)
    kernel = kernel.reshape((roi_size, roi_size, roi_size))
    kernel = np.power(kernel, 1.0 / 3.0)
    kernel /= kernel.max()
    return kernel.astype(np.float32)


def _crop_cube_padded(volume, center, size=26):
    """Crop a cube with padding fallback for border cases."""
    r = size // 2
    x0, y0, z0 = int(center[0]), int(center[1]), int(center[2])
    w, h, d = volume.shape
    if (r < x0 < w - r - 1) and (r < y0 < h - r - 1) and (r < z0 < d - r - 1):
        return volume[x0-r:x0-r+size, y0-r:y0-r+size, z0-r:z0-r+size].copy()
    padded = np.pad(volume, ((r, r), (r, r), (r, r)), "minimum")
    return padded[x0:x0+size, y0:y0+size, z0:z0+size].copy()


def _save_empty(ref_nii, nii_path, json_path):
    """Save empty Pred when no lesions found."""
    os.makedirs(os.path.dirname(nii_path), exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(ref_nii.shape[:3], dtype=np.int16),
                              ref_nii.affine, ref_nii.header), nii_path)
    _save_json(json_path, [], 0)
    logger.info("No lesions found. Saved empty Pred.")


def _save_json(path, lesion_stats, count):
    """Save CMB detection results as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"lesion_count": count, "lesions": lesion_stats}, f, indent=2)
