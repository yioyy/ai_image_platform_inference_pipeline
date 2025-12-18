from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import nibabel.orientations as nio
import numpy as np


def _meta(path: Path) -> Dict[str, object]:
    img = nib.load(str(path))
    a = img.affine
    return {
        "path": str(path),
        "shape": tuple(int(x) for x in img.shape),
        "zooms": tuple(float(x) for x in img.header.get_zooms()[:3]),
        "axcodes": tuple(nio.aff2axcodes(a)),
        "det": float(np.linalg.det(a[:3, :3])),
        "qform_code": int(img.header.get("qform_code", 0)),
        "sform_code": int(img.header.get("sform_code", 0)),
        "affine": a.tolist(),
    }


def _overlap(a: Path, b: Path) -> Tuple[int, int, int]:
    A = np.asanyarray(nib.load(str(a)).dataobj) > 0
    B = np.asanyarray(nib.load(str(b)).dataobj) > 0
    inter = int(np.logical_and(A, B).sum())
    return inter, int(A.sum()), int(B.sum())


def _overlap_flip_z(a: Path, b: Path) -> int:
    A = np.asanyarray(nib.load(str(a)).dataobj) > 0
    B = np.asanyarray(nib.load(str(b)).dataobj) > 0
    Bf = np.flip(B, axis=2)
    return int(np.logical_and(A, Bf).sum())


def main() -> None:
    root_new = Path(os.environ["ROOT_NEW"])
    root_old = Path(os.environ["ROOT_OLD"])

    bpred_new = root_new / "baseline" / "Pred_CMB.nii.gz"
    bsyn_new = root_new / "baseline" / "SynthSEG_CMB.nii.gz"
    preg_new = root_new / "registration" / "Pred_CMB_registration.nii.gz"

    bpred_old = root_old / "baseline" / "Pred_CMB.nii.gz"
    preg_old = root_old / "registration" / "Pred_CMB_registration.nii.gz"

    print("== NEW metas ==")
    for p in [bpred_new, bsyn_new, preg_new]:
        print(_meta(p))

    print("== OLD overlap ==")
    print(_overlap(bpred_old, preg_old))

    print("== NEW overlap ==")
    print(_overlap(bpred_new, preg_new))
    print("== NEW overlap (flip z) ==")
    print(_overlap_flip_z(bpred_new, preg_new))

    # sanity: are baseline Pred and SynthSEG affines identical?
    a_pred = np.asarray(nib.load(str(bpred_new)).affine, dtype=np.float64)
    a_syn = np.asarray(nib.load(str(bsyn_new)).affine, dtype=np.float64)
    print("== baseline pred vs synthseg affine allclose ==")
    print(bool(np.allclose(a_pred, a_syn, atol=1e-6)))


if __name__ == "__main__":
    main()


