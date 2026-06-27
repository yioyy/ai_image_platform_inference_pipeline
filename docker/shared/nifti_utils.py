"""Shared NIfTI coordinate transforms for all models.

Two modes based on how the model was trained:
  - WMH: qfac flip only in reverse direction (flip_to_cnn ignores qfac)
  - Infarct/Aneurysm: qfac flip in BOTH directions

pixdim[0] is the NIfTI qfac (+1 or -1), controlling left-right orientation.
"""

import nibabel as nib
import numpy as np


def flip_to_cnn(arr: np.ndarray, nii: nib.Nifti1Image = None, qfac_both: bool = False) -> np.ndarray:
    """NIfTI native → CNN axis order.

    Args:
        arr: NIfTI array.
        nii: NIfTI image (needed if qfac_both=True).
        qfac_both: If True, apply qfac flip in forward direction too
                   (Infarct/Aneurysm models). WMH uses False.
    """
    arr = np.swapaxes(arr, 0, 1)
    arr = np.flip(arr, 0)
    arr = np.flip(arr, -1)
    if qfac_both and nii is not None and nii.header.copy()["pixdim"][0] > 0:
        arr = np.flip(arr, 1)
    return arr


def flip_to_native(arr: np.ndarray, nii: nib.Nifti1Image) -> np.ndarray:
    """CNN → NIfTI native axis order. Always applies qfac correction."""
    if nii.header.copy()["pixdim"][0] > 0:
        arr = np.flip(arr, 1)
    arr = np.flip(arr, -1)
    arr = np.flip(arr, 0)
    arr = np.swapaxes(arr, 1, 0)
    return arr


def nii_replace(nii: nib.Nifti1Image, data: np.ndarray) -> nib.Nifti1Image:
    """Create new NIfTI with same affine/header but different data."""
    return nib.nifti1.Nifti1Image(data, nii.affine, header=nii.header.copy())
