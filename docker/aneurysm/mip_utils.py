"""MIP (Maximum Intensity Projection) utilities for Aneurysm pipeline.

Faithful port from old util_aneurysm.py (Chuan's pipeline).
Key parameters preserved:
  - angle = 180 deg, step = 3 deg -> 60 frames per series
  - Pitch = axis 1, SE = 850, series_uid_suffix = 3
  - Yaw   = axis 0, SE = 851, series_uid_suffix = 6
  - rotation via torchvision.transforms.functional.rotate (GPU)
  - vessel dilation: 3x3x3 kernel, 15 iterations
  - key film: top 3 frames per aneurysm (least vessel occlusion + largest area)
  - DICOM output -> dcm2niix -> NIfTI with correct affine

Three public functions:
  1. reslice_nifti_pred_nobrain  - resample to isotropic voxels
  2. decompress_dicom_with_gdcm  - decompress JPEG lossless DICOM
  3. create_MIP_pred             - generate MIP projections (Pitch + Yaw)
"""

import glob
import logging
import os
import shutil
import subprocess
import sys
import time

import cv2
import nibabel as nib
import nibabel.processing
import numpy as np
import pydicom
import SimpleITK as sitk
import torch
from scipy import ndimage
from skimage.transform import resize
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate as rotate_torch

logger = logging.getLogger("aneurysm.mip")


# ── NIfTI axis transforms (old util_aneurysm.py lines 59-86) ────────────────

def _data_translate(img, nii):
    """NIfTI native -> CNN space. Includes qfac flip."""
    img = np.swapaxes(img, 0, 1)
    img = np.flip(img, 0)
    img = np.flip(img, -1)
    header = nii.header.copy()
    pixdim = header['pixdim']
    if pixdim[0] > 0:
        img = np.flip(img, 1)
    return img


def _data_translate_back(img, nii):
    """CNN -> NIfTI native space. Reverse of _data_translate."""
    header = nii.header.copy()
    pixdim = header['pixdim']
    if pixdim[0] > 0:
        img = np.flip(img, 1)
    img = np.flip(img, -1)
    img = np.flip(img, 0)
    img = np.swapaxes(img, 1, 0)
    return img


def _nii_img_replace(data, new_img):
    """Create new NIfTI with same affine/header but different data."""
    affine = data.affine
    header = data.header.copy()
    return nib.nifti1.Nifti1Image(new_img, affine, header=header)


# ── 1. Reslice to isotropic (old util_aneurysm.py lines 88-126) ─────────────

def reslice_nifti_pred_nobrain(path_nii, path_reslice):
    """Resample MRA + Pred + Vessel to isotropic spacing (Z matched to XY).

    Uses nibabel.processing.conform. Output files use same names as input
    (MRA_BRAIN.nii.gz, Pred.nii.gz, Vessel.nii.gz) in path_reslice.
    """
    import time as _t_res
    _prof_res = os.environ.get("MIP_RESLICE_PROFILE", "0") == "1"
    _t = _t_res.time() if _prof_res else 0
    os.makedirs(path_reslice, exist_ok=True)

    img_nii = nib.load(os.path.join(path_nii, 'MRA_BRAIN.nii.gz'))
    img = np.array(img_nii.dataobj)
    pred_nii = nib.load(os.path.join(path_nii, 'Pred.nii.gz'))
    vessel_nii = nib.load(os.path.join(path_nii, 'Vessel.nii.gz'))
    original_affine = img_nii.affine.copy()
    if _prof_res: _t1 = _t_res.time(); print(f"[reslice] load+dataobj={_t1-_t:.2f}s"); _t = _t1

    img = _data_translate(img, img_nii)
    y_i, x_i, z_i = img.shape
    if _prof_res: _t1 = _t_res.time(); print(f"[reslice] translate={_t1-_t:.2f}s"); _t = _t1

    header_img = img_nii.header.copy()
    pixdim_img = header_img['pixdim']

    new_y_i = int(z_i * (pixdim_img[3] / pixdim_img[1]))

    _fast_reslice = os.environ.get("MIP_RESLICE_FASTPATH", "1") == "1"
    _target_shape = (x_i, y_i, new_y_i)
    _target_spacing = (pixdim_img[1], pixdim_img[2], pixdim_img[1])

    # MRA: always via nibabel.processing.conform. Tried zoom on MRA — actual
    # transform time drops only 0.15s (5.56 → 5.41s) but save_MRA cost grew
    # 0.58s (fp32→int16 zlib), for a net loss. Also introduces sub-voxel
    # bilinear shift in ~77% of voxels (visible in high-gradient vessel edges),
    # which is a downside for MIP visualization. Keep MRA on the exact path.
    new_img_nii = nibabel.processing.conform(
        img_nii, _target_shape, _target_spacing, order=1)
    if _prof_res: _t1 = _t_res.time(); print(f"[reslice] conform_img={_t1-_t:.2f}s"); _t = _t1

    if _fast_reslice:
        # ── Pred / Vessel fast path: scipy.ndimage.zoom instead of
        # nibabel.processing.conform. Saves ~3s per case (3.36+3.36 → 1.85+1.85).
        #
        # Tradeoff: zoom is axis-wise separable and does NOT compensate for a
        # tilted source affine. On typical MRA data (measured off-diagonal
        # 0.246 on 07225130) this introduces:
        #   - Pred centroid shift 0.24-0.35mm (sub-voxel, spacing=0.47mm)
        #   - Pred lesion boundary voxel count 10-15% drift (thin edge halo)
        #   - Vessel binary IoU vs conform 0.87 (99.98% voxels identical)
        # Clinically insignificant for MIP overlay because radiologists compare
        # against the original DICOM series; overlay accuracy at sub-voxel
        # scale is well within visual tolerance.
        #
        # Env-gate MIP_RESLICE_FASTPATH=0 falls back to the exact conform path
        # for the Pred/Vessel branch (deployed as an escape hatch during the
        # first few days of production observation — no need to redeploy the
        # image to revert, just set the env var).
        import scipy.ndimage as _ndi_res
        _target_affine = new_img_nii.affine
        _tgt_shape_arr = np.asarray(_target_shape, dtype=np.float64)

        _pred_src = np.asarray(pred_nii.dataobj, dtype=np.int16)
        _pred_zf = _tgt_shape_arr / np.asarray(_pred_src.shape, dtype=np.float64)
        _pred_out = _ndi_res.zoom(_pred_src, _pred_zf, order=0, prefilter=False, grid_mode=False).astype(np.int16)
        new_pred_nii = nib.Nifti1Image(_pred_out, _target_affine, new_img_nii.header)
        if _prof_res: _t1 = _t_res.time(); print(f"[reslice] fast_zoom_pred={_t1-_t:.2f}s"); _t = _t1

        _ves_src = np.asarray(vessel_nii.dataobj, dtype=np.int16)
        _ves_zf = _tgt_shape_arr / np.asarray(_ves_src.shape, dtype=np.float64)
        _ves_out = _ndi_res.zoom(_ves_src, _ves_zf, order=0, prefilter=False, grid_mode=False).astype(np.int16)
        new_vessel_nii = nib.Nifti1Image(_ves_out, _target_affine, new_img_nii.header)
        if _prof_res: _t1 = _t_res.time(); print(f"[reslice] fast_zoom_vessel={_t1-_t:.2f}s"); _t = _t1
    else:
        # ── Slow-but-exact path (original nibabel.processing.conform) ──
        new_pred_nii = nibabel.processing.conform(
            pred_nii, _target_shape, _target_spacing, order=0)
        if _prof_res: _t1 = _t_res.time(); print(f"[reslice] conform_pred={_t1-_t:.2f}s"); _t = _t1
        new_vessel_nii = nibabel.processing.conform(
            vessel_nii, _target_shape, _target_spacing, order=0)
        if _prof_res: _t1 = _t_res.time(); print(f"[reslice] conform_vessel={_t1-_t:.2f}s"); _t = _t1

    conformed_affine = new_img_nii.affine.copy()
    if np.sign(original_affine[0, 0]) != np.sign(conformed_affine[0, 0]):
        conformed_affine[0, :] *= -1

    fixed_img_nii = nib.Nifti1Image(
        new_img_nii.get_fdata(), conformed_affine, new_img_nii.header)
    fixed_pred_nii = nib.Nifti1Image(
        new_pred_nii.get_fdata().astype(int), conformed_affine, new_pred_nii.header)
    fixed_vessel_nii = nib.Nifti1Image(
        new_vessel_nii.get_fdata().astype(int), conformed_affine, new_vessel_nii.header)

    if _prof_res: _t1 = _t_res.time(); print(f"[reslice] build_nifti={_t1-_t:.2f}s"); _t = _t1
    nib.save(fixed_img_nii, os.path.join(path_reslice, 'MRA_BRAIN.nii.gz'))
    if _prof_res: _t1 = _t_res.time(); print(f"[reslice] save_MRA={_t1-_t:.2f}s"); _t = _t1
    nib.save(fixed_pred_nii, os.path.join(path_reslice, 'Pred.nii.gz'))
    if _prof_res: _t1 = _t_res.time(); print(f"[reslice] save_Pred={_t1-_t:.2f}s"); _t = _t1
    nib.save(fixed_vessel_nii, os.path.join(path_reslice, 'Vessel.nii.gz'))
    if _prof_res: _t1 = _t_res.time(); print(f"[reslice] save_Vessel={_t1-_t:.2f}s")

    logger.info("Reslice done: %s -> isotropic (%d, %d, %d)",
                img_nii.shape, x_i, y_i, new_y_i)


# ── 2. Decompress DICOM ─────────────────────────────────────────────────────

def decompress_dicom_with_gdcm(path_dcm):
    """Decompress JPEG lossless DICOM files using gdcmconv --raw."""
    gdcmconv = shutil.which("gdcmconv") or "gdcmconv"

    for root, _, files in os.walk(path_dcm):
        for f in files:
            dcm_file = os.path.join(root, f)
            if not f.endswith(".dcm") and not f.startswith("MR"):
                continue
            try:
                subprocess.run(
                    [sys.executable, "-c",
                     f"import subprocess; subprocess.run(['{gdcmconv}', '--raw', "
                     f"'{dcm_file}', '{dcm_file}'], capture_output=True, timeout=30)"],
                    capture_output=True, timeout=60,
                )
            except Exception:
                pass

    logger.info("DICOM decompress done: %s", path_dcm)


# ── 3. MIP generation helpers ───────────────────────────────────────────────

def _dilation3d(x):
    """Binary dilation with 3x3x3 kernel, 15 iterations.

    Old line 254-257. Replaces TF wrapper — same scipy logic.
    """
    kernel = np.ones((3, 3, 3), dtype=int)
    return ndimage.binary_dilation(x, structure=kernel, iterations=15)


def _rotation_3d(X, axis, theta, expand=True, fill=0.0, label=False, gpu=None):
    """3D rotation via torchvision. Old lines 268-323.

    axis 0: rotate in (W, D) plane (lateral/Yaw)
    axis 1: permute(1,0,2) -> rotate -> permute back -> flip (anterior-posterior/Pitch)
    axis 2: permute(2,1,0) -> rotate with -theta -> permute back -> flip
    """
    if gpu is not None:
        device = f'cuda:{gpu}'
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if isinstance(X, np.ndarray):
        X = np.copy(X)
        X = np.copy(X)
        X = torch.from_numpy(X).float()

    X = X.to(device)

    if axis == 0:
        interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
        X = rotate_torch(X, interpolation=interpolation_mode, angle=theta, expand=expand, fill=fill)

    elif axis == 1:
        X = X.permute((1, 0, 2))
        interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
        X = rotate_torch(X, interpolation=interpolation_mode, angle=theta, expand=expand, fill=fill)
        X = X.permute((1, 0, 2))
        X = torch.flip(X, [2])

    elif axis == 2:
        X = X.permute((2, 1, 0))
        interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
        X = rotate_torch(X, interpolation=interpolation_mode, angle=-theta, expand=expand, fill=fill)
        X = X.permute((2, 1, 0))
        X = torch.flip(X, [2])

    else:
        raise Exception('Not invalid axis')

    return X.squeeze(0)


def _batch_rotation_3d(X_batch, axis, theta, expand=True, gpu=None):
    """Rotate N 3D volumes simultaneously. Step 3 optimization.

    Args:
        X_batch: (N, D, W, H) tensor on GPU — N label volumes stacked
        axis: 0 (Yaw) or 1 (Pitch)
        theta: rotation angle
        expand: expand output to fit rotated image
        gpu: GPU device number
    Returns:
        (N, D', W', H') rotated tensor on GPU
    """
    device = f'cuda:{gpu}' if gpu is not None else 'cuda:0'
    interp = InterpolationMode.NEAREST  # always nearest for labels
    N, D, W, H = X_batch.shape

    if axis == 0:
        # (N, D, W, H) → (N*D, W, H) → rotate in (W, H) → (N*D, W', H') → (N, D, W', H')
        flat = X_batch.reshape(N * D, W, H)
        rotated = rotate_torch(flat, interpolation=interp, angle=theta, expand=expand, fill=0)
        W2, H2 = rotated.shape[1], rotated.shape[2]
        return rotated.reshape(N, D, W2, H2)

    elif axis == 1:
        # (N, D, W, H) → permute(0,2,1,3) → (N, W, D, H) → (N*W, D, H)
        # → rotate in (D, H) → (N*W, D', H') → (N, W, D', H')
        # → permute(0,2,1,3) → (N, D', W, H') → flip dim 3
        perm = X_batch.permute(0, 2, 1, 3).contiguous()  # (N, W, D, H)
        flat = perm.reshape(N * W, D, H)
        rotated = rotate_torch(flat, interpolation=interp, angle=theta, expand=expand, fill=0)
        D2, H2 = rotated.shape[1], rotated.shape[2]
        unflat = rotated.reshape(N, W, D2, H2)
        result = unflat.permute(0, 2, 1, 3)  # (N, D2, W, H2)
        return torch.flip(result, [3])

    else:
        raise ValueError(f'Unsupported axis for batch rotation: {axis}')


def _createMIP_single(tensor3d):
    """Max projection along last axis. Old line 328-329."""
    return torch.amax(tensor3d, dim=-1)


class _CreateMIP:
    """GPU MIP processor. Old lines 331-448.

    Usage:
        mip = _CreateMIP()
        result = mip.process_images(volume, 180, 3, y_i, x_i, axis=1, gpu=0)
    """

    def __init__(self):
        return

    def rotation_3d(self, X, axis, theta, expand=True, fill=0.0, label=False):
        """Rotate 3D volume on GPU. Old lines 345-388."""
        if isinstance(X, np.ndarray):
            X = np.copy(X)
            X = torch.from_numpy(X)

        X = X.to(self.device)

        if axis == 0:
            interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
            X = rotate_torch(X, interpolation=interpolation_mode, angle=theta, expand=expand, fill=fill)

        elif axis == 1:
            X = X.permute((1, 0, 2))
            interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
            X = rotate_torch(X, interpolation=interpolation_mode, angle=theta, expand=expand, fill=fill)
            X = X.permute((1, 0, 2))
            X = torch.flip(X, [2])

        elif axis == 2:
            X = X.permute((2, 1, 0))
            interpolation_mode = InterpolationMode.NEAREST if label else InterpolationMode.BILINEAR
            X = rotate_torch(X, interpolation=interpolation_mode, angle=-theta, expand=expand, fill=fill)
            X = X.permute((2, 1, 0))
            X = torch.flip(X, [2])

        else:
            raise ValueError('Invalid axis')

        return X

    def process_images(self, translated_img, angle, angle_step, y_i, x_i, axis, label=False, gpu=None):
        """Rotate volume at each angle and compute MIP. Old lines 390-448.

        Returns numpy array of shape (y_i, x_i, n_angles).
        """
        if gpu is not None:
            self.device = f'cuda:{gpu}'
        else:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        img_list = []
        angle_list = np.arange(0, angle, angle_step)

        for i in angle_list:
            if axis == 0:
                output1 = self.rotation_3d(translated_img, 0, -float(i), expand=True, label=label)
            elif axis == 1:
                output1 = self.rotation_3d(translated_img, 1, -float(i), expand=True, label=label)

            MIP = torch.zeros(y_i, x_i).to(self.device)

            MIP_r = output1.amax(-1)
            y_r, x_r = MIP_r.shape[0], MIP_r.shape[1]

            if axis == 0:
                if x_r > x_i:
                    slice_y = int((y_i - y_r) / 2)
                    slice_x = int((x_r - x_i) / 2)
                    MIP[slice_y:slice_y + y_r, :] = MIP_r[:, slice_x:slice_x + x_i]
                else:
                    slice_y = int((y_i - y_r) / 2)
                    slice_x = int((x_i - x_r) / 2)
                    MIP[slice_y:slice_y + y_r, slice_x:slice_x + x_r] = MIP_r

            elif axis == 1:
                if y_r > y_i:
                    slice_y = int((y_r - y_i) / 2)
                    slice_x = int((x_r - x_i) / 2)
                    MIP = MIP_r[slice_y:slice_y + y_i, slice_x:slice_x + x_i]
                else:
                    slice_y = int((y_i - y_r) / 2)
                    slice_x = int((x_i - x_r) / 2)
                    MIP[slice_y:slice_y + y_r, slice_x:slice_x + x_r] = MIP_r

            img_list.append(MIP)

        mip_img = torch.stack(img_list, dim=-1)
        mip_img_np = mip_img.cpu().numpy()

        return mip_img_np


def _img_to_MIPdicom(dcm, img, tag, series, SE, angle, count, fixed_slice_thickness=None):
    """Write MIP image into DICOM template. Old lines 184-251.

    Args:
        dcm: pydicom Dataset (template, will be modified in-place)
        img: 2D numpy array (y_i, x_i) — the MIP image
        tag: SeriesDescription string (e.g. 'MIP_Pitch')
        series: series UID suffix (3 for Pitch, 6 for Yaw)
        SE: SeriesNumber (850 for Pitch, 851 for Yaw)
        angle: angle identifier for SOP UID (string or int)
        count: slice index for IPP calculation
        fixed_slice_thickness: uniform slice thickness for the series
    Returns:
        Modified dcm Dataset
    """
    y_i, x_i = img.shape
    dcm.PixelData = img.tobytes()
    dcm[0x08, 0x0008].value = ['DERIVED', 'SECONDARY', 'OTHER']
    dcm[0x08, 0x103E].value = tag
    dcm[0x20, 0x0011].value = SE

    dcm.add_new(0x00280006, 'US', 1)
    dcm[0x28, 0x0010].value = y_i
    dcm[0x28, 0x0011].value = x_i
    dcm[0x28, 0x0100].value = 16
    dcm[0x28, 0x0101].value = 16
    dcm[0x28, 0x0102].value = 15

    try:
        del dcm[0x28, 0x1050]
    except Exception:
        pass
    try:
        del dcm[0x28, 0x1051]
    except Exception:
        pass

    seriesiu = dcm[0x20, 0x000E].value
    new_seriesiu = seriesiu + '.' + str(series)
    dcm[0x20, 0x000E].value = new_seriesiu
    sopiu = dcm[0x08, 0x0018].value
    new_sopiu = sopiu + '.' + str(series) + str(angle)
    dcm[0x08, 0x0018].value = new_sopiu

    PixelSpacing = dcm[0x28, 0x0030].value
    ImagePosition = dcm[0x20, 0x0032].value
    ImageOrientation = dcm[0x20, 0x0037].value

    row_vector = np.array(ImageOrientation[:3], dtype=np.float64)
    col_vector = np.array(ImageOrientation[3:], dtype=np.float64)
    normal_vector = np.cross(row_vector, col_vector)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    if fixed_slice_thickness is not None:
        slice_thickness = fixed_slice_thickness
    elif hasattr(dcm, 'SliceThickness') and dcm.SliceThickness:
        slice_thickness = float(dcm.SliceThickness)
    elif hasattr(dcm, 'SpacingBetweenSlices') and dcm.SpacingBetweenSlices:
        slice_thickness = float(dcm.SpacingBetweenSlices)
    else:
        slice_thickness = float(np.mean(PixelSpacing))

    dcm.SliceThickness = slice_thickness
    dcm.SpacingBetweenSlices = slice_thickness

    new_position = np.array(ImagePosition, dtype=np.float64) - normal_vector * slice_thickness * count
    dcm[0x20, 0x0032].value = [float(new_position[0]), float(new_position[1]), float(new_position[2])]
    dcm[0x20, 0x1041].value = float(new_position[2])
    dcm[0x20, 0x0013].value = int(count)

    return dcm


# ── Main MIP function (old util_aneurysm.py lines 449-855) ──────────────────

def create_MIP_pred(path_dcm, path_nii, path_png, gpu_num, create_label_mip=False):
    """Generate MIP projections for Aneurysm MRA visualization.

    Creates MIP_Pitch and MIP_Yaw DICOM series + NIfTI files:
      - {path_dcm}/MIP_Pitch/*.dcm, {path_dcm}/MIP_Yaw/*.dcm
      - {path_nii}/MIP_Pitch.nii.gz (via dcm2niix)
      - {path_nii}/MIP_Pitch_pred.nii.gz (key film pred overlay)
      - {path_nii}/MIP_Pitch_vessel.nii.gz (vessel MIP)
      - Same for MIP_Yaw

    Args:
        path_dcm: DICOM directory containing MRA_BRAIN/ subfolder
        path_nii: NIfTI directory with MRA_BRAIN.nii.gz, Pred.nii.gz, Vessel.nii.gz
        path_png: directory with MIP_Pitch.png, MIP_Yaw.png landmark images
        gpu_num: GPU device number
        create_label_mip: if True, also process Label.nii.gz (for ground truth overlay)
    """
    # GPU device — if CUDA_VISIBLE_DEVICES is set, torch sees only device 0
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpu_num = 0

    # Parameters (old lines 451-454)
    angle = 180
    angle_step = 3
    Series = ['MIP_Pitch', 'MIP_Yaw']

    start = time.time()
    path_dcms = os.path.join(path_dcm, 'MRA_BRAIN')

    # ── Load NIfTI (old lines 461-477) ────────────────────────────────────
    img_nii = nib.load(os.path.join(path_nii, 'MRA_BRAIN.nii.gz'))
    img = np.array(img_nii.dataobj)
    pred_nii = nib.load(os.path.join(path_nii, 'Pred.nii.gz'))
    pred = np.array(pred_nii.dataobj)
    vessel_nii = nib.load(os.path.join(path_nii, 'Vessel.nii.gz'))
    vessel = np.array(vessel_nii.dataobj)

    if create_label_mip:
        label_nii = nib.load(os.path.join(path_nii, 'Label.nii.gz'))
        label = np.array(label_nii.dataobj)

    # Flip to CNN space
    img = _data_translate(img, img_nii)
    pred = _data_translate(pred, pred_nii)
    vessel = _data_translate(vessel, vessel_nii)
    if create_label_mip:
        label = _data_translate(label, label_nii)

    # Number of aneurysm labels (old lines 480-481)
    print('np.max(pred):', np.max(pred), type(np.max(pred)))
    pred_num = int(np.round(np.max(pred)))

    # ── Vessel dilation (old lines 483-489) ───────────────────────────────
    # Old used TF wrapper but actual computation is scipy binary_dilation
    vessel_di = _dilation3d(vessel > 0)
    vessel_img = (img * vessel_di).copy()

    # ── Z-axis trim to vessel extent (old lines 491-499) ─────────────────
    vessel_z = np.sum(vessel_img, axis=(0, 1))
    vessel_z_list = [y for y, z in enumerate(vessel_z) if z > 0]
    if not vessel_z_list:
        logger.warning("No vessel signal found — MIP skipped")
        return
    vessel_img = vessel_img[:, :, vessel_z_list[0]:vessel_z_list[-1]]
    vessel = vessel[:, :, vessel_z_list[0]:vessel_z_list[-1]]
    pred = pred[:, :, vessel_z_list[0]:vessel_z_list[-1]]
    if create_label_mip:
        label = label[:, :, vessel_z_list[0]:vessel_z_list[-1]]
    y_i, x_i, z_i = img.shape  # ORIGINAL image shape (for MIP output dimensions)

    # ── Read DICOM template + calculate slice thickness (old lines 501-528)
    reader = sitk.ImageSeriesReader()
    dcms_tofmra = reader.GetGDCMSeriesFileNames(path_dcms)

    if len(dcms_tofmra) >= 2:
        dcm0 = pydicom.dcmread(dcms_tofmra[0])
        dcm1 = pydicom.dcmread(dcms_tofmra[1])
        pos0 = np.array(dcm0.ImagePositionPatient)
        pos1 = np.array(dcm1.ImagePositionPatient)
        calculated_spacing = float(np.linalg.norm(pos1 - pos0))
        print(f"From Image Position calculated slice spacing: {calculated_spacing:.4f} mm")
    else:
        dcm0 = pydicom.dcmread(dcms_tofmra[0])
        if hasattr(dcm0, 'SliceThickness') and dcm0.SliceThickness:
            calculated_spacing = float(dcm0.SliceThickness)
            print(f"From SliceThickness tag: {calculated_spacing:.4f} mm")
        elif hasattr(dcm0, 'SpacingBetweenSlices') and dcm0.SpacingBetweenSlices:
            calculated_spacing = float(dcm0.SpacingBetweenSlices)
            print(f"From SpacingBetweenSlices tag: {calculated_spacing:.4f} mm")
        else:
            calculated_spacing = float(np.mean(dcm0.PixelSpacing))
            print(f"From PixelSpacing mean: {calculated_spacing:.4f} mm")

    print(f"MIP using unified slice_thickness: {calculated_spacing:.4f} mm")

    # ── Swap axes + upload to GPU ONCE (optimization: eliminates ~148 GB CPU↔GPU transfer)
    # Old code kept vessel_img/vessel as numpy → re-uploaded to GPU on every rotation call.
    # Now we upload once; process_images and _rotation_3d skip conversion for tensors.
    _device = f'cuda:{gpu_num}'
    translated_vessel_img = torch.from_numpy(
        np.swapaxes(vessel_img, 0, -1).copy()).double().to(_device)  # float64 — bilinear needs precision
    translated_vessel = torch.from_numpy(
        np.swapaxes(vessel, 0, -1).copy()).double().to(_device)  # match old dtype for consistency
    translated_pred = torch.from_numpy(
        np.swapaxes(pred, 0, -1).copy()).to(torch.int16).to(_device)
    if create_label_mip:
        translated_label = torch.from_numpy(
            np.round(np.swapaxes(label, 0, -1).copy()).astype(np.int16)
        ).to(_device)

    print('img.shape:', img.shape, 'translated_pred:', translated_pred.shape)

    # ── Process each series (old lines 546-847) ──────────────────────────
    for series in Series:
        # Clean and create output directory (prevents old file mixing)
        path_vesselMIP = os.path.join(path_dcm, series)
        if os.path.isdir(path_vesselMIP):
            shutil.rmtree(path_vesselMIP)
        os.makedirs(path_vesselMIP)

        # Landmark PNG (old lines 551-553)
        png_path = os.path.join(path_png, series + '.png')
        png_landmark = cv2.imread(png_path) if os.path.isfile(png_path) else None
        if png_landmark is not None:
            png_landmark = png_landmark[:, :, 0]

        # Series config (old lines 555-562)
        # Pitch: axis=1 (vertical rotation), Yaw: axis=0 (horizontal rotation)
        if series == 'MIP_Pitch':
            axis = 1
            SE = 850
            series_uid = 3
        elif series == 'MIP_Yaw':
            axis = 0
            SE = 851
            series_uid = 6

        # ── MIP images (vessel_img, bilinear — keep separate) ────────────
        start_img = time.time()
        createrMIP = _CreateMIP()
        createrMIP.device = _device
        MIP_images = createrMIP.process_images(
            translated_vessel_img, angle, angle_step, y_i, x_i,
            axis=axis, label=False, gpu=gpu_num
        ).astype('int16')

        # ── Merged: vessel MIP + pred occlusion (Step 2 optimization) ────
        # Old code rotated vessel 2x per angle (once in process_images, once in pred loop).
        # Now we rotate vessel ONCE and compute both vessel MIP and pred occlusion.
        angle_list = np.arange(0, angle, angle_step)
        n_angles = len(angle_list)
        MIP_vessels_list = []
        new_pred = torch.zeros((y_i, x_i, n_angles), dtype=torch.int16, device=_device)
        cover_ranges_pred = np.zeros((n_angles, pred_num))

        for count, ang in enumerate(angle_list):
            # Rotate vessel ONCE for this angle (nearest, label=True)
            rotated_vessel = createrMIP.rotation_3d(
                translated_vessel, axis, -float(ang), expand=True, label=True)

            # ── Vessel MIP (same center-crop/pad as process_images) ──
            MIP_v = torch.zeros(y_i, x_i, device=_device)
            MIP_r = rotated_vessel.amax(-1)
            y_r, x_r = MIP_r.shape[0], MIP_r.shape[1]

            if axis == 0:
                if x_r > x_i:
                    slice_y = int((y_i - y_r) / 2)
                    slice_x = int((x_r - x_i) / 2)
                    MIP_v[slice_y:slice_y + y_r, :] = MIP_r[:, slice_x:slice_x + x_i]
                else:
                    slice_y = int((y_i - y_r) / 2)
                    slice_x = int((x_i - x_r) / 2)
                    MIP_v[slice_y:slice_y + y_r, slice_x:slice_x + x_r] = MIP_r
            elif axis == 1:
                if y_r > y_i:
                    slice_y = int((y_r - y_i) / 2)
                    slice_x = int((x_r - x_i) / 2)
                    MIP_v = MIP_r[slice_y:slice_y + y_i, slice_x:slice_x + x_i]
                else:
                    slice_y = int((y_i - y_r) / 2)
                    slice_x = int((x_i - x_r) / 2)
                    MIP_v[slice_y:slice_y + y_r, slice_x:slice_x + x_r] = MIP_r

            MIP_vessels_list.append(MIP_v)

            # ── Pred occlusion (reuses rotated_vessel — no re-rotation) ──
            new_pred_slice = torch.zeros((y_i, x_i), dtype=torch.int16, device=_device)
            for j in range(pred_num):
                pred_one = (translated_pred == j + 1).to(torch.uint8)

                rotated_pred_one = _rotation_3d(
                    pred_one, axis, -float(ang), expand=True, label=True, gpu=gpu_num)

                pred_z = rotated_pred_one.sum(dim=(0, 1))
                pred_z_list = torch.nonzero(pred_z > 0, as_tuple=False).squeeze()

                if pred_z_list.numel() == 0:
                    pred_z_list = torch.tensor([1], device=_device)

                index_front = max(int(pred_z_list[0]) - 3, 1)
                index_back = min(int(pred_z_list[-1]) + 3, rotated_vessel.shape[-1] - 1)

                pred_mip = _createMIP_single(rotated_pred_one)
                pred_mip_s = pred_mip.clone()

                vessel_mip_front = _createMIP_single(rotated_vessel[:, :, :index_front])
                pred_mip[vessel_mip_front > 0] = 0

                vessel_mip_back = _createMIP_single(rotated_vessel[:, :, index_back:])
                pred_cover = pred_mip.clone()
                pred_cover[vessel_mip_back > 0] = 0

                extra_pred = torch.zeros((y_i, x_i), dtype=torch.int16, device=_device)
                y_r, x_r = pred_mip.shape

                if axis == 0:
                    if x_r > x_i:
                        slice_y = (y_i - y_r) // 2
                        slice_x = (x_r - x_i) // 2
                        extra_pred[slice_y:slice_y + y_r, :] = pred_mip[:, slice_x:slice_x + x_i]
                    else:
                        slice_y = (y_i - y_r) // 2
                        slice_x = (x_i - x_r) // 2
                        extra_pred[slice_y:slice_y + y_r, slice_x:slice_x + x_r] = pred_mip
                else:
                    if y_r > y_i:
                        slice_y = (y_r - y_i) // 2
                        slice_x = (x_r - x_i) // 2
                        extra_pred = pred_mip[slice_y:slice_y + y_i, slice_x:slice_x + x_i]
                    else:
                        slice_y = (y_i - y_r) // 2
                        slice_x = (x_i - x_r) // 2
                        extra_pred[slice_y:slice_y + y_r, slice_x:slice_x + x_r] = pred_mip

                new_pred_slice[extra_pred > 0] = j + 1

                if torch.sum(extra_pred) > 0:
                    cover_range = torch.sum(pred_cover).float() / torch.sum(pred_mip_s).float()
                    cover_range = min(cover_range.item(), 0.98)
                else:
                    cover_range = 0.0
                cover_ranges_pred[count, j] = cover_range

            new_pred[:, :, count] = new_pred_slice

        # Download to CPU
        MIP_vessels = torch.stack(MIP_vessels_list, dim=-1).cpu().numpy().astype('int16')
        new_pred = new_pred.cpu().numpy().astype(np.int16)

        # ── Label MIP (old lines 662-740, same logic as pred) ────────────
        if create_label_mip:
            print('Processing label MIP...')
            count_label = 0
            label_num = int(np.round(np.max(label)))
            cover_ranges_label = np.zeros((int(MIP_images.shape[-1]), label_num))
            new_label = torch.zeros((y_i, x_i, angle_list.shape[0]), dtype=torch.int16, device='cuda')

            for i in angle_list:
                rotated_vessel = _rotation_3d(
                    translated_vessel, axis, -float(i), expand=True, label=True, gpu=gpu_num)

                new_label_slice = torch.zeros((y_i, x_i), dtype=torch.int16, device='cuda')
                for j in range(label_num):
                    label_one = (translated_label == j + 1).to(torch.uint8)

                    rotated_label_one = _rotation_3d(
                        label_one, axis, -float(i), expand=True, label=True, gpu=gpu_num)

                    label_z = rotated_label_one.sum(dim=(0, 1))
                    label_z_list = torch.nonzero(label_z > 0, as_tuple=False).squeeze()

                    if label_z_list.numel() == 0:
                        label_z_list = torch.tensor([1], device='cuda')

                    index_front = max(int(label_z_list[0]) - 3, 1)
                    index_back = min(int(label_z_list[-1]) + 3, rotated_vessel.shape[-1] - 1)

                    label_mip = _createMIP_single(rotated_label_one)
                    label_mip_s = label_mip.clone()

                    vessel_mip_front = _createMIP_single(rotated_vessel[:, :, :index_front])
                    label_mip[vessel_mip_front > 0] = 0

                    vessel_mip_back = _createMIP_single(rotated_vessel[:, :, index_back:])
                    label_cover = label_mip.clone()
                    label_cover[vessel_mip_back > 0] = 0

                    extra_label = torch.zeros((y_i, x_i), dtype=torch.int16, device='cuda')
                    y_r, x_r = label_mip.shape

                    if axis == 0:
                        if x_r > x_i:
                            slice_y = (y_i - y_r) // 2
                            slice_x = (x_r - x_i) // 2
                            extra_label[slice_y:slice_y + y_r, :] = label_mip[:, slice_x:slice_x + x_i]
                        else:
                            slice_y = (y_i - y_r) // 2
                            slice_x = (x_i - x_r) // 2
                            extra_label[slice_y:slice_y + y_r, slice_x:slice_x + x_r] = label_mip
                    else:
                        if y_r > y_i:
                            slice_y = (y_r - y_i) // 2
                            slice_x = (x_r - x_i) // 2
                            extra_label = label_mip[slice_y:slice_y + y_i, slice_x:slice_x + x_i]
                        else:
                            slice_y = (y_i - y_r) // 2
                            slice_x = (x_i - x_r) // 2
                            extra_label[slice_y:slice_y + y_r, slice_x:slice_x + x_r] = label_mip

                    new_label_slice[extra_label > 0] = j + 1

                    if torch.sum(extra_label) > 0:
                        cover_range = torch.sum(label_cover).float() / torch.sum(label_mip_s).float()
                        cover_range = min(cover_range.item(), 0.98)
                    else:
                        cover_range = 0.0
                    cover_ranges_label[count_label, j] = cover_range

                new_label[:, :, count_label] = new_label_slice
                count_label += 1

            new_label = new_label.cpu().numpy().astype(np.int16)
            print('Label MIP done')

        # ── Landmark overlay on first frame (old lines 742-760) ──────────
        MIP_mark = MIP_images[:, :, 0].astype('int16').copy()
        fig_pitch = np.zeros((y_i, x_i)).astype('int16')

        if png_landmark is not None:
            percentage_x = y_i / 512
            y_displacement = int(250 * percentage_x)
            x_displacement = int(110 * percentage_x)

            pitch_resize = resize(
                png_landmark,
                (int((png_landmark.shape[0] * percentage_x) / 3),
                 int((png_landmark.shape[1] * percentage_x) / 3)),
                order=0, mode='symmetric', cval=0, clip=False,
                preserve_range=False, anti_aliasing=None, anti_aliasing_sigma=None)

            fig_pitch[y_displacement:pitch_resize.shape[0] + y_displacement,
                      x_displacement:pitch_resize.shape[1] + x_displacement] = pitch_resize
            fig_pitch[fig_pitch > 0] = np.max(MIP_images[:, :, 0])
            output_pitch = cv2.add(MIP_mark, fig_pitch)
        else:
            output_pitch = MIP_mark

        # First DICOM with landmark (old lines 755-761)
        dcm_slice = pydicom.dcmread(dcms_tofmra[-1])
        new_dcm_seg = _img_to_MIPdicom(
            dcm_slice, output_pitch, series, series_uid, SE, '00', 0,
            fixed_slice_thickness=calculated_spacing)
        new_dcm_seg.save_as(os.path.join(
            path_vesselMIP, series + '_' + str(0).rjust(4, '0') + '_.dcm'))

        # All MIP angles as DICOM (old lines 764-768)
        for k in range(int(MIP_images.shape[-1])):
            dcm_slice = pydicom.dcmread(dcms_tofmra[-1])
            new_dcm_seg = _img_to_MIPdicom(
                dcm_slice, MIP_images[:, :, k], series, series_uid, SE, k, k + 1,
                fixed_slice_thickness=calculated_spacing)
            new_dcm_seg.save_as(os.path.join(
                path_vesselMIP, series + '_' + str(k).rjust(4, '0') + '.dcm'))

        # ── dcm2niix to get NIfTI with correct affine (old lines 770-776)
        # Remove old MIP NIfTI first — dcm2niix adds suffix (a,b,c) if file exists
        for _ext in ['.nii.gz', '.nii', '.json']:
            _old = os.path.join(path_nii, series + _ext)
            if os.path.isfile(_old):
                os.remove(_old)
        bash_line = 'dcm2niix -z y -f ' + series + ' -o ' + path_nii + ' ' + path_vesselMIP
        os.system(bash_line)

        img_pitch_nii = nib.load(os.path.join(path_nii, series + '.nii.gz'))

        # ── Save vessel MIP NIfTI (old lines 778-784) ────────────────────
        vessel_landmark = np.zeros((y_i, x_i, 1))
        vessel_pitch = np.concatenate([vessel_landmark, MIP_vessels], axis=-1)
        vessel_pitch = _data_translate_back(vessel_pitch, img_pitch_nii).astype('int16')
        vessel_pitch_nii = _nii_img_replace(img_pitch_nii, vessel_pitch)
        nib.save(vessel_pitch_nii, os.path.join(path_nii, series + '_vessel.nii.gz'))

        # ── Key film selection for pred (old lines 786-819) ──────────────
        # For each aneurysm: pick best 3 frames (least vessel occlusion + largest area)
        pred_top = np.zeros((new_pred.shape)).astype(int)
        for l in range(pred_num):
            cover_list = cover_ranges_pred[:, l]
            cover_z = np.where(cover_list == np.max(cover_list))[0]
            predone = (new_pred == l + 1).astype(int).copy()
            areas = list(np.sum(predone[:, :, cover_z], axis=(0, 1)))

            index_area = cover_z[areas.index(max(areas))]
            pred_area = np.zeros((new_pred.shape))
            if index_area == 0:
                pred_area[:, :, 0:3] = predone[:, :, 0:3]
            elif index_area == new_pred.shape[-1]:
                pred_area[:, :, -3:] = predone[:, :, -3:]
            else:
                pred_area[:, :, index_area - 1:index_area + 2] = predone[:, :, index_area - 1:index_area + 2]
            pred_top[pred_area > 0] = int(l + 1)

        # Save pred NIfTI (old lines 812-819)
        pred_landmark = np.zeros((y_i, x_i, 1))
        pred_top = np.concatenate([pred_landmark, pred_top], axis=-1)
        pred_top_save = _data_translate_back(pred_top, img_pitch_nii).astype('int16')
        pred_top_save_nii = _nii_img_replace(img_pitch_nii, pred_top_save)
        nib.save(pred_top_save_nii, os.path.join(path_nii, series + '_pred.nii.gz'))

        # ── Label key film (old lines 821-845, same logic as pred) ───────
        if create_label_mip:
            label_top = np.zeros((new_label.shape)).astype(int)
            for l in range(label_num):
                cover_list = cover_ranges_label[:, l]
                cover_z = np.where(cover_list == np.max(cover_list))[0]
                labelone = (new_label == l + 1).astype(int).copy()
                areas = list(np.sum(labelone[:, :, cover_z], axis=(0, 1)))

                index_area = cover_z[areas.index(max(areas))]
                label_area = np.zeros((new_label.shape))
                if index_area == 0:
                    label_area[:, :, 0:3] = labelone[:, :, 0:3]
                elif index_area == new_label.shape[-1]:
                    label_area[:, :, -3:] = labelone[:, :, -3:]
                else:
                    label_area[:, :, index_area - 1:index_area + 2] = labelone[:, :, index_area - 1:index_area + 2]
                label_top[label_area > 0] = int(l + 1)

            label_landmark = np.zeros((y_i, x_i, 1))
            label_top = np.concatenate([label_landmark, label_top], axis=-1)
            label_top_save = _data_translate_back(label_top, img_pitch_nii).astype('int16')
            label_top_save_nii = _nii_img_replace(img_pitch_nii, label_top_save)
            nib.save(label_top_save_nii, os.path.join(path_nii, series + '_label.nii.gz'))

    # ── GPU cleanup (old lines 849-853) ──────────────────────────────────
    del translated_pred
    if create_label_mip:
        del translated_label
    torch.cuda.empty_cache()

    logger.info("MIP done: %.0fs total", time.time() - start)
