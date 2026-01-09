"""
Follow-up registration utilities.

責任：
- 使用 FSL FLIRT 將 followup（moving）對位到 baseline（reference）空間
- 輸出配準後的 SynthSEG 與 Pred（label 用 nearest-neighbour）

注意：
- 本模組只負責「呼叫外部配準工具」與檔案輸出，不做影像內容運算。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from nibabel import processing as nibproc


@dataclass(frozen=True)
class RegistrationOutputs:
    mat_file: Path
    synthseg_reg_file: Path
    pred_reg_file: Path


def run_registration_followup_to_baseline(
    *,
    baseline_synthseg_file: Path,
    baseline_pred_file: Optional[Path] = None,
    followup_synthseg_file: Path,
    followup_pred_file: Path,
    out_dir: Path,
    mat_file: Path,
    fsl_flirt_path: str = "/usr/local/fsl/bin/flirt",
    dof: int = 6,
    overwrite: bool = False,
    out_synthseg_reg_file: Optional[Path] = None,
    out_pred_reg_file: Optional[Path] = None,
) -> RegistrationOutputs:
    """
    將 followup 對位到 baseline（預設方向：Follow-up -> Baseline）。

    輸出：
    - SynthSEG_<model>_registration.nii.gz
    - Pred_<model>_registration.nii.gz
    - registration .mat（由呼叫端命名）
    """
    _ensure_file(baseline_synthseg_file, "baseline_synthseg_file")
    if baseline_pred_file:
        _ensure_file(baseline_pred_file, "baseline_pred_file")
    _ensure_file(followup_synthseg_file, "followup_synthseg_file")
    _ensure_file(followup_pred_file, "followup_pred_file")

    out_dir.mkdir(parents=True, exist_ok=True)
    mat_file.parent.mkdir(parents=True, exist_ok=True)

    synthseg_reg_file = out_synthseg_reg_file or _derive_registration_filename(
        out_dir, followup_synthseg_file, suffix="_registration"
    )
    pred_reg_file = out_pred_reg_file or _derive_registration_filename(
        out_dir, followup_pred_file, suffix="_registration"
    )

    if not overwrite and synthseg_reg_file.exists() and pred_reg_file.exists() and mat_file.exists():
        return RegistrationOutputs(
            mat_file=mat_file,
            synthseg_reg_file=synthseg_reg_file,
            pred_reg_file=pred_reg_file,
        )

    if _need_min_1mm_resample([baseline_synthseg_file, followup_synthseg_file]):
        tmp_dir = out_dir / "_tmp_min1mm"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 注意：baseline/followup 常見檔名相同（例如都叫 SynthSEG_CMB.nii.gz），必須加前綴避免覆寫。
        baseline_min1mm = tmp_dir / _append_suffix(f"baseline_{baseline_synthseg_file.name}", "_min1mm")
        followup_synthseg_min1mm = tmp_dir / _append_suffix(f"followup_{followup_synthseg_file.name}", "_min1mm")

        _resample_label_min_1mm_per_axis(
            src=baseline_synthseg_file,
            dst=baseline_min1mm,
            threshold_mm=1.0,
        )
        _resample_label_min_1mm_per_axis(
            src=followup_synthseg_file,
            dst=followup_synthseg_min1mm,
            threshold_mm=1.0,
        )

        # 額外保存：將 resample 後的 baseline/followup 檔案分別保存到 registration/resample/ 方便人工對照。
        resample_dir = out_dir / "resample"
        resample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline_min1mm, resample_dir / baseline_min1mm.name)
        shutil.copy2(followup_synthseg_min1mm, resample_dir / followup_synthseg_min1mm.name)

        tmp_mat_1mm = tmp_dir / _append_suffix(mat_file.name, "_min1mm")
        tmp_reg_1mm = tmp_dir / _append_suffix(synthseg_reg_file.name, "_min1mm")
        _run_flirt_calc_matrix(
            fsl_flirt_path=fsl_flirt_path,
            moving=followup_synthseg_min1mm,
            reference=baseline_min1mm,
            out_image=tmp_reg_1mm,
            out_mat=tmp_mat_1mm,
            dof=dof,
        )
        # 也保存「對位後（min1mm 空間）」的 registered 影像與矩陣，便於人工 QA
        shutil.copy2(tmp_reg_1mm, resample_dir / tmp_reg_1mm.name)
        shutil.copy2(tmp_mat_1mm, resample_dir / tmp_mat_1mm.name)

        # NOTE:
        # 既然 min1mm 空間的 registration 影像（tmp_reg_1mm）是正確的，
        # 代表「矩陣估計」沒問題；歪掉的來源通常是「把矩陣換算回原空間」這一步。
        # 因此這裡不再做矩陣換算，直接使用 FLIRT 在 min1mm 空間估計的 mat 作為輸出 mat。
        shutil.copy2(tmp_mat_1mm, mat_file)
        shutil.copy2(mat_file, resample_dir / mat_file.name)

        _run_flirt_apply_matrix(
            fsl_flirt_path=fsl_flirt_path,
            moving=followup_synthseg_file,
            reference=baseline_synthseg_file,
            out_image=synthseg_reg_file,
            mat_file=mat_file,
            interp="nearestneighbour",
        )
        _run_flirt_apply_matrix(
            fsl_flirt_path=fsl_flirt_path,
            moving=followup_pred_file,
            reference=baseline_pred_file or baseline_synthseg_file,
            out_image=pred_reg_file,
            mat_file=mat_file,
            interp="nearestneighbour",
        )
        # 也保存「最終對位後」影像到 resample/（避免被後續流程覆蓋時難追）
        shutil.copy2(synthseg_reg_file, resample_dir / synthseg_reg_file.name)
        shutil.copy2(pred_reg_file, resample_dir / pred_reg_file.name)
    else:
        resample_dir = out_dir / "resample"
        resample_dir.mkdir(parents=True, exist_ok=True)

        _run_flirt_calc_matrix(
            fsl_flirt_path=fsl_flirt_path,
            moving=followup_synthseg_file,
            reference=baseline_synthseg_file,
            out_image=synthseg_reg_file,
            out_mat=mat_file,
            dof=dof,
        )

        _run_flirt_apply_matrix(
            fsl_flirt_path=fsl_flirt_path,
            moving=followup_pred_file,
            reference=baseline_pred_file or baseline_synthseg_file,
            out_image=pred_reg_file,
            mat_file=mat_file,
            interp="nearestneighbour",
        )
        # 保存對位後影像到 resample/
        shutil.copy2(synthseg_reg_file, resample_dir / synthseg_reg_file.name)
        shutil.copy2(pred_reg_file, resample_dir / pred_reg_file.name)

    _ensure_file(synthseg_reg_file, "synthseg_reg_file (output)")
    _ensure_file(pred_reg_file, "pred_reg_file (output)")
    _ensure_file(mat_file, "mat_file (output)")

    return RegistrationOutputs(
        mat_file=mat_file,
        synthseg_reg_file=synthseg_reg_file,
        pred_reg_file=pred_reg_file,
    )


def _run_flirt_calc_matrix(
    *,
    fsl_flirt_path: str,
    moving: Path,
    reference: Path,
    out_image: Path,
    out_mat: Path,
    dof: int,
) -> None:
    cmd = [
        fsl_flirt_path,
        "-in",
        str(moving),
        "-ref",
        str(reference),
        "-out",
        str(out_image),
        "-omat",
        str(out_mat),
        "-bins",
        "64",
        "-searchrx",
        "-60",
        "60",
        "-searchry",
        "-60",
        "60",
        "-searchrz",
        "-60",
        "60",
        "-cost",
        "corratio",
        "-dof",
        str(dof),
        "-interp",
        "nearestneighbour",
    ]
    _run_cmd(cmd, hint="flirt (calc matrix)")


def _resample_label_min_1mm_per_axis(*, src: Path, dst: Path, threshold_mm: float) -> None:
    """
    label 影像逐軸 resample：任一軸 spacing < threshold 則該軸改成 threshold，其餘軸不動。

    - 只支援 3D
    - 最近鄰（nearest-neighbour）
    - 避免引入額外依賴（scipy）
    """
    """
    改用 `nibabel.processing.conform` 進行 resample（參考 util_aneurysm.py 作法）。

    重點：
    - conform 會同步更新 affine/header，避免 resample 後出現幾何不一致造成的平移/錯位
    - label 使用 order=0（nearest neighbour）
    """
    img = nib.load(str(src))
    data = np.asanyarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"resample 只支援 3D NIfTI：{src} (ndim={data.ndim})")

    zooms = tuple(float(x) for x in img.header.get_zooms()[:3])
    target_zooms = tuple(max(z, float(threshold_mm)) for z in zooms)
    if all(z >= threshold_mm for z in zooms):
        if src.resolve() == dst.resolve():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        return

    out_shape = tuple(int(np.ceil(s * oz / tz)) for s, oz, tz in zip(img.shape[:3], zooms, target_zooms))
    conformed = nibproc.conform(img, out_shape=out_shape, voxel_size=target_zooms, order=0)

    # 確保 dtype 維持與輸入一致（label）
    out = np.asanyarray(conformed.dataobj).astype(data.dtype, copy=False)
    header = conformed.header.copy()
    header.set_data_dtype(out.dtype)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, conformed.affine, header=header), str(dst))


#
# NOTE:
# 先前使用 `_affine_preserve_center` + `resample_from_to` 的做法已改為 conform。
# 若未來需要比較不同 resample 方法，再考慮恢復。
#


def _run_flirt_apply_matrix(
    *,
    fsl_flirt_path: str,
    moving: Path,
    reference: Path,
    out_image: Path,
    mat_file: Path,
    interp: str,
) -> None:
    cmd = [
        fsl_flirt_path,
        "-in",
        str(moving),
        "-ref",
        str(reference),
        "-out",
        str(out_image),
        "-applyxfm",
        "-init",
        str(mat_file),
        "-interp",
        interp,
    ]
    _run_cmd(cmd, hint="flirt (apply matrix)")


def _run_cmd(cmd: list[str], *, hint: Optional[str] = None) -> None:
    cmd_str = " ".join(cmd)
    if hint:
        logging.info("RUN %s: %s", hint, cmd_str)
        print(f"RUN {hint}: {cmd_str}", flush=True)
    else:
        logging.info("RUN: %s", cmd_str)
        print(f"RUN: {cmd_str}", flush=True)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info("DONE %s", hint or cmd[0])
        print(f"DONE {hint or cmd[0]}", flush=True)
    except FileNotFoundError as exc:
        detail = f"找不到外部指令：{cmd[0]}"
        raise RuntimeError(detail) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        message = hint or "external command failed"
        raise RuntimeError(f"{message}: {stderr or stdout or 'unknown error'}") from exc


def _ensure_file(path: Path, name: str) -> None:
    if path.exists():
        return
    raise FileNotFoundError(f"缺少檔案 ({name}): {path}")


def _derive_registration_filename(out_dir: Path, src: Path, *, suffix: str) -> Path:
    """
    從輸入檔名推導 registration 輸出檔名，保留 model 的大小寫（例如 Pred_CMB.nii.gz）。

    範例：
    - Pred_CMB.nii.gz -> Pred_CMB_registration.nii.gz
    - SynthSEG_CMB.nii -> SynthSEG_CMB_registration.nii
    """
    name = src.name
    if name.endswith(".nii.gz"):
        return out_dir / (name[: -len(".nii.gz")] + suffix + ".nii.gz")
    if name.endswith(".nii"):
        return out_dir / (name[: -len(".nii")] + suffix + ".nii")
    return out_dir / f"{src.stem}{suffix}{src.suffix}"


def _need_min_1mm_resample(paths: list[Path], *, threshold_mm: float = 1.0) -> bool:
    for p in paths:
        x, y, z = _load_zooms_3d_mm(p)
        if x < threshold_mm or y < threshold_mm or z < threshold_mm:
            return True
    return False


def _load_zooms_3d_mm(path: Path) -> tuple[float, float, float]:
    img = nib.load(str(path))
    zooms = img.header.get_zooms()
    if len(zooms) < 3:
        raise ValueError(f"NIfTI zooms 不足 3 維：{path} (zooms={zooms})")
    x, y, z = zooms[:3]
    return (abs(float(x)), abs(float(y)), abs(float(z)))


def _load_affine(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    aff = np.asarray(img.affine, dtype=np.float64)
    if aff.shape != (4, 4):
        raise ValueError(f"NIfTI affine 必須為 4x4：{path} (shape={aff.shape})")
    return aff


def _load_fsl_mat(path: Path) -> np.ndarray:
    mat = np.loadtxt(str(path), dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"FSL mat 必須為 4x4：{path} (shape={mat.shape})")
    return mat


def _save_fsl_mat(path: Path, mat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.asarray(mat, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"FSL mat 必須為 4x4：{path} (shape={m.shape})")
    np.savetxt(str(path), m, fmt="%.10f")


def _convert_flirt_mat_between_grids(
    *,
    mat_vox: np.ndarray,
    in_affine_from: np.ndarray,
    ref_affine_from: np.ndarray,
    in_affine_to: np.ndarray,
    ref_affine_to: np.ndarray,
) -> np.ndarray:
    """
    將 FLIRT 的 voxel-space 轉換矩陣，從「from grid」換算到「to grid」。

    假設 FLIRT mat 為 voxel mapping：
    - v_ref = M_vox @ v_in

    NIfTI affine 定義 voxel->mm：
    - x_mm = A @ v

    世界座標轉換：
    - T_mm = A_ref_from @ M_vox @ inv(A_in_from)

    轉回 to grid 的 voxel-space：
    - M_vox_to = inv(A_ref_to) @ T_mm @ A_in_to
    """
    m = np.asarray(mat_vox, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"mat_vox 必須為 4x4 (shape={m.shape})")

    a_in_from = np.asarray(in_affine_from, dtype=np.float64)
    a_ref_from = np.asarray(ref_affine_from, dtype=np.float64)
    a_in_to = np.asarray(in_affine_to, dtype=np.float64)
    a_ref_to = np.asarray(ref_affine_to, dtype=np.float64)

    t_mm = a_ref_from @ m @ np.linalg.inv(a_in_from)
    return np.linalg.inv(a_ref_to) @ t_mm @ a_in_to


def _append_suffix(filename: str, suffix: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[: -len(".nii.gz")] + suffix + ".nii.gz"
    if filename.endswith(".nii"):
        return filename[: -len(".nii")] + suffix + ".nii"
    p = Path(filename)
    return f"{p.stem}{suffix}{p.suffix}"


def _derive_pred_mat_file(*, out_dir: Path, mat_file: Path, suffix: str) -> Path:
    """
    給 followup_pred 套用用的 mat（對 baseline_pred grid）。
    放在 registration/ 旁，避免污染 _tmp_*。
    """
    name = mat_file.name
    if name.endswith(".mat"):
        return out_dir / (name[: -len(".mat")] + suffix + ".mat")
    return out_dir / (name + suffix)


def _maybe_write_predgrid_mat(
    *,
    baseline_synthseg_file: Path,
    followup_synthseg_file: Path,
    baseline_pred_file: Optional[Path],
    followup_pred_file: Path,
    mat_file: Path,
    out_mat_file: Path,
) -> None:
    """
    若提供 baseline_pred_file，則將「SynthSEG 配準矩陣」換算成「Pred grid 可用的矩陣」。

    目標：確保 followup_pred registration 的輸出 voxel grid 與 baseline_pred 一致，
    以符合 followup_report 的 overlap（用 voxel index 交集）。
    """
    if not baseline_pred_file:
        return

    m_ss = _load_fsl_mat(mat_file)
    t_mm = _load_affine(baseline_synthseg_file) @ m_ss @ np.linalg.inv(_load_affine(followup_synthseg_file))
    m_pred = np.linalg.inv(_load_affine(baseline_pred_file)) @ t_mm @ _load_affine(followup_pred_file)
    _save_fsl_mat(out_mat_file, m_pred)

