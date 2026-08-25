#!/usr/bin/env python3
"""Infarct postprocessing — territory stats, DICOM-SEG, RADAX bundle, notify.

Input (written by preprocess.py / inference.py, all under process_dir):
  - SynthSEG.nii.gz              david 1xx/2xx/3xx territory vocabulary
  - nnUNet/Pred.nii.gz           uint8, connected components already numbered
  - nnUNet/Prob.nii.gz           float32 softmax
  - nnUNet/Image_nii/ADC.nii.gz  raw ADC (not skull-stripped), for mean ADC
  - nnUNet/Dicom/{DWI1000,ADC}/  source DICOM series

Output — the same flat bundle written to two places:
  - prediction.json
  - <seg_series_instance_uid>_<label>.dcm, two per detection

  1. <RADX_UPLOAD_DIR|AI_INFERENCE_RESULT_PATH>/<StudyInstanceUID>/infarct_model/
     <inference_id>/ — this is the deliverable. It is the tree the RAD backend
     scans, the same layout docker/aneurysm and docker/cmb write and the same
     layout as the reference bundles under process/Deep_Infarct_radax.
  2. output_folder/ — where the dispatcher keeps this study's NIfTI predictions
     and where server.py gates the phase ({output_folder}/prediction.json). A
     copy here is not a delivery; it is the local record and the phase gate.

Two SEG sets per lesion is not redundancy: the platform renders the DWI1000 and
the ADC series side by side, and a SEG can only reference the frames of one
series. Both sets go in the top-level `detections` list, each row naming its
own source in annotated_series_instance_uid; `reformatted_series` stays empty.
That key means something narrower than "a second series": the platform reads
every entry as a derived image series it must ingest, requiring a directory of
DICOM images named by its series_instance_uid inside the bundle (that is where
aneurysm ships its MIP), and it stamps those detections AI_DERIVED_ANEURYSM_SEG
unconditionally. ADC is an acquired series already in the archive, so it belongs
in neither role.

Deliberately NOT carried over from _reference/ (needFollowup era, old platform):
orthanc_zip_upload, upload_json_aiteam, the $9 followup json, and the PNG report
burn-in. What is carried over is the clinical arithmetic — per-territory volume,
mean ADC, probability, the CSF/Background reassignment — and the DICOM-SEG
geometry.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

logger = logging.getLogger("infarct.postprocess")

MODEL_NAME = "infarct_model"

# Last-resort upload root. Both containers that already deliver (aneurysm, cmb)
# use this same literal, and compose sets AI_INFERENCE_RESULT_PATH explicitly on
# a bind mount, so this value only ever applies to a bare CLI invocation.
UPLOAD_ROOT_DEFAULT = "/home/david/ai-inference-result"

# Assigned by the platform 2026-08, fixed for every site rather than minted
# per deployment: aneurysm and CMB were already hardcoded and infarct now
# matches them, so site-config can carry the value instead of the two sides
# copying it to each other on install day.
#
# The same constant is bound worker-side, in task_pipeline's model
# resolution. Setting one without the other is the failure to watch for:
# the worker would route the job correctly while the payload it produced
# still claimed the all-zero placeholder.
INFARCT_MODEL_ID = "908f2f2f-4774-4652-91f9-e9b2674de9c6"

# Territory cut-off. _reference used volume_ml >= 0.3, which on a thin-z DWI
# (~0.9 x 0.9 x 6 mm, roughly 5 uL per voxel) means ~62 voxels — it silently
# scaled with acquisition geometry and dropped a whole size class of real
# lesions. Counting voxels decouples the threshold from voxel size.
MIN_REGION_VOXELS = 6

# Present in Infarct_labels_OHIF only so label -> colour lookups resolve; they
# are not clinical territories and must not become detections.
SKIP_REGIONS = frozenset({"Background", "CSF"})

# _reference/dicomseg/infarct.py paints the whole-lesion mask hotpink and falls
# back to lawngreen for a territory missing from Infarct_colors_OHIF. Kept so
# the platform's colours do not shift for readers used to the old output.
TOTAL_COLOR = "hotpink"
REGION_FALLBACK_COLOR = "lawngreen"

# Matches only the filenames this module itself produces. Used to clean a
# previous run without touching other stages' files in the same directory.
BUNDLE_DCM_RE = re.compile(r"^[0-9.]+_(?:Total|A\d+)\.dcm$")


def _utc_iso_now_ms() -> str:
    """2026-08-19T03:18:17.151Z — the format the platform parses."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _dicomseg_utils():
    """Import code_ai's DICOM-SEG helpers lazily.

    Kept out of module scope on purpose: server.py imports this module at
    process start, so a missing or broken code_ai would kill the container
    before it can serve /health, instead of failing one study with a log line.
    """
    from code_ai.pipeline.dicomseg import utils

    return utils


def _dataset_json_path() -> str:
    """Locate dataset.json (the Infarct_labels_OHIF / Infarct_colors_OHIF source).

    _reference/ is documented as reference material that does not go into the
    container, so resolving the territory vocabulary there would work on a
    developer checkout and fail in the image — the worst kind of difference.
    The copy next to this file is the one that ships; INFARCT_DATASET_JSON
    exists so a vocabulary change can be tried without a rebuild.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # Order matters. `here` is /app inside the container, and /app is a staging
    # directory filled with `cp docker/<model>/*.py` -- data files are not
    # copied, so the sibling candidate only ever resolves on a dev checkout
    # (runtime/aneurysm holds nine files and all nine are .py). The chuan tree
    # is bind-mounted into the container at its own path, so that is the one
    # that actually works in the image.
    candidates = [
        os.environ.get("INFARCT_DATASET_JSON", ""),
        os.path.join(here, "dataset.json"),
        "/home/david/pipeline/chuan/code/docker/infarct/dataset.json",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "dataset.json not found; tried: %s" % [c for c in candidates if c]
    )


def _load_dataset_config() -> Tuple[Dict[str, List[int]], Dict[str, str]]:
    path = _dataset_json_path()
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    labels = data.get("Infarct_labels_OHIF") or {}
    if not labels:
        raise ValueError(f"Infarct_labels_OHIF missing or empty in {path}")
    logger.info("[postprocess] territory vocabulary: %s (%d entries)", path, len(labels))
    return labels, (data.get("Infarct_colors_OHIF") or {})


def _upload_root() -> str:
    """Root of the tree the RAD backend scans for finished bundles.

    RADX_UPLOAD_DIR wins over AI_INFERENCE_RESULT_PATH, matching aneurysm and
    cmb: the testing stack points it at a separate tree so a test study never
    appears on the production worklist.
    """
    return os.environ.get(
        "RADX_UPLOAD_DIR",
        os.environ.get("AI_INFERENCE_RESULT_PATH", UPLOAD_ROOT_DEFAULT),
    )


def _label_order(label: str) -> Tuple[int, int]:
    """Sort key for lesion labels: Total first, then A1, A2, ... A9, A10, A11.

    _reference build_infarct.py:107 ordered the same labels with a bare
    sorted(), which is lexicographic — "A10" sorts between "A1" and "A2". Under
    the old volume_ml >= 0.3 cut a study almost never reached ten territories,
    so numeric and lexicographic order agreed and the bug never fired. With
    MIN_REGION_VOXELS = 6 ten-plus territories are routine, and the failure is
    silent: detection N gets paired with a different lesion's SEG, so the
    platform draws the wrong mask underneath the right measurements.

    Deliberately strict — an unexpected label raises rather than sorting to some
    arbitrary position, because a mis-ordered bundle is not detectable
    downstream.
    """
    if label == "Total":
        return (0, 0)
    return (1, int(label[1:]))


def _stats_for_mask(
    location: str,
    mask: np.ndarray,
    voxel_ml: float,
    adc: np.ndarray,
    prob: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Per-lesion clinical numbers. Adapted from _reference infarct_pipeline.py.

    Two changes from that source: voxel_count is returned (the territory cut-off
    is now expressed in voxels, so the number it is compared against has to be
    visible), and volume keeps 3 decimals instead of 1 — at 6 voxels a 1-decimal
    volume rounds to 0.0 and the finding reads as empty.
    """
    voxel_count = int(mask.sum())
    volume_ml = round(voxel_count * voxel_ml, 3)

    mean_adc = float(np.mean(adc[mask])) if voxel_count > 0 else 0.0

    if prob is not None and voxel_count > 0:
        prob_values = prob[mask]
        prob_max = float(np.max(prob_values))
        prob_mean = float(np.mean(prob_values))
    else:
        prob_max = 0.0
        prob_mean = 0.0

    return {
        "location": location,
        "voxel_count": voxel_count,
        "volume_ml": volume_ml,
        "mean_adc": round(mean_adc),
        "prob_max": round(prob_max, 2),
        "prob_mean": round(prob_mean, 2),
        "type": "",
    }


def _assign_territories(
    synth: np.ndarray,
    total_mask: np.ndarray,
    region_labels: Dict[str, List[int]],
    slice_axis: int = 0,
) -> np.ndarray:
    """Territory id per lesion voxel, with CSF/Background voxels pulled back in.

    Ported from _reference/post_infarct.py::parcellation and the slice loop that
    calls it. Why it has to exist: SynthSEG labels the sulcal CSF a swollen
    cortical infarct displaces, and the ventricle margin beside a deep infarct,
    as CSF or Background. Those voxels are inside Pred — they are infarct — but
    they carry no territory, so without reassignment they count towards Total
    and towards nothing else. The per-territory volumes then under-report, and
    they under-report worst on the large lesions where the territory split is
    the thing the reader actually wants.

    Nearest parenchymal voxel *within the same axial slice*, not in 3D. That is
    the original's choice and it is the right one for this data: DWI slice
    spacing is ~6 mm against ~0.9 mm in plane, so a 3D nearest neighbour would
    usually be found one slice away, in whatever territory happens to sit above
    or below — a geometric answer to an anatomical question.

    Two deliberate departures from the original. It padded each class's
    coordinate table starting at `c_coord.shape[1]+1`, leaving one slot at
    (0, 0), so an image-corner phantom could win the nearest-neighbour vote for
    a territory the lesion never touches; that is dropped. And exact ties resolve
    to whatever the distance transform reports rather than to the lowest
    SynthSEG id, because between two equidistant territories the original's
    loop order was an implementation detail, not a clinical rule.
    """
    territory_ids = np.asarray(
        sorted({int(i) for name, ids in region_labels.items()
                if name not in SKIP_REGIONS for i in ids}),
        dtype=np.int32,
    )
    skip_ids = np.asarray(
        sorted({int(i) for name in SKIP_REGIONS for i in region_labels.get(name, [])}),
        dtype=np.int32,
    )

    # Attractors are restricted to the reported vocabulary, mirroring the
    # original's `intersect1d(outside_labels, ...)`: a voxel reassigned to a
    # label no territory claims would be no better off than it started.
    parenchyma = np.where(np.isin(synth, territory_ids), synth, 0).astype(np.int32)
    assigned = np.where(total_mask, parenchyma, 0).astype(np.int32)

    unparcellated = total_mask & np.isin(synth, skip_ids)
    if not unparcellated.any():
        return assigned

    # A plane here has to be an acquired axial slice -- the plane the original
    # iterated over -- and which axis that is depends on the caller's array
    # space. Postprocess works after get_array_to_dcm_axcodes, where it is
    # axis 0; the report draws in the reference display orientation, where it
    # is axis 2. Reassigning across the wrong plane would pull voxels to a
    # territory they never touch, so the axis is stated rather than assumed.
    work = np.moveaxis(unparcellated, slice_axis, 0)
    ref_all = np.moveaxis(parenchyma, slice_axis, 0)
    out = np.moveaxis(assigned, slice_axis, 0)

    moved = 0
    for z in np.unique(np.nonzero(work)[0]):
        source = work[z]
        reference = ref_all[z]
        if not reference.any():
            # No parenchyma anywhere in this slice — above the vertex, or a
            # slice SynthSEG gave up on. There is nothing to attach these
            # voxels to, and inventing a territory from a neighbouring slice is
            # exactly what the in-plane rule above exists to avoid.
            continue
        # distance_transform_edt on the complement returns, for every voxel
        # outside the parenchyma, the index of the nearest parenchymal voxel.
        # Same answer as the original's explicit O(n*m) loop over per-class
        # coordinate lists, in one pass instead of one pass per territory.
        _, nearest = ndimage.distance_transform_edt(reference == 0, return_indices=True)
        out[z][source] = reference[nearest[0][source], nearest[1][source]]
        moved += int(source.sum())

    logger.info("[postprocess] reassigned %d CSF/Background lesion voxels to the "
                "nearest in-plane territory", moved)
    # moveaxis returns a view, so writes through `out` already landed in
    # `assigned`.
    return assigned


def _compute_region_stats(
    region_labels: Dict[str, List[int]],
    territory: np.ndarray,
    total_mask: np.ndarray,
    voxel_ml: float,
    adc: np.ndarray,
    prob: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    """Split the lesion across SynthSEG territories, in dataset.json order.

    _reference read a precomputed Result/Pred_synthseg.nii.gz, which was just
    SynthSEG multiplied by the prediction. Deriving the territory volume here
    instead keeps one less volume on disk that can go stale against Pred.nii.gz.

    The 15-territory cap in the original is gone: it existed to keep the old
    platform's finding list short, and it dropped real infarct without saying so.

    Order is Infarct_labels_OHIF's declaration order, not volume descending.
    Infarct is typically one confluent lesion crossing several adjoining
    territories, and the declaration order walks the brain in a fixed anatomical
    sequence — lobes, then deep grey, then cerebellum and brainstem, left
    hemisphere before right. So A1..An read down the lesion in an order that
    holds still: the same territory keeps the same rank between two studies of
    the same patient, and A-numbers do not jump around inside one lesion because
    two territories traded a few voxels. The three RADAX reference bundles under
    process/Deep_Infarct_radax are all in declaration order — e5fde05d-... has
    A3 L. Temporal Lobe at 19.0 mL sitting behind A1 at 4.8 and A2 at 4.6.
    """
    stats: List[Dict[str, Any]] = []
    for region, label_ids in region_labels.items():
        if region in SKIP_REGIONS:
            continue
        mask = np.isin(territory, np.asarray(label_ids, dtype=np.int32)) & total_mask
        if not mask.any():
            continue
        item = _stats_for_mask(region, mask, voxel_ml, adc, prob)
        if item["voxel_count"] < MIN_REGION_VOXELS:
            continue
        item["mask"] = mask
        stats.append(item)

    return stats


def _main_seg_slice(mask: np.ndarray) -> int:
    """Median slice index of the lesion, in the sorted-DICOM slice order.

    After get_array_to_dcm_axcodes axis 0 is ('S'), i.e. the same axis the
    DICOM stack is sorted along, so this index addresses a source instance
    directly. Median over voxels rather than over occupied slices, matching
    _reference dicomseg/infarct.py::_calc_main_slice.
    """
    indices = np.where(mask)
    if indices[0].size == 0:
        return 0
    return int(np.median(indices[0]))


def _clear_stale_bundle(output_folder: str) -> None:
    """Remove a previous run's bundle from output_folder.

    The dispatcher derives output_folder from the pipeline's output path, so it
    is the *same* directory on every retrigger of a study — unlike the upload
    tree, which gets a fresh <inference_id>/ each time. pydicom_seg mints new
    SeriesInstanceUIDs on every run, so without this the old SEGs survive under
    filenames no detection in the new prediction.json refers to, and the
    platform imports orphan masks. Only this module's own naming is matched;
    anything else in the directory belongs to another stage.

    Called from _publish_to_output_folder and nowhere else. That placement is
    the point: this used to run first, before the DICOM load and the SEG
    generation, so any failure downstream of it left the study with no result at
    all — a rerun that hit a missing input turned a complete previous bundle
    into an empty directory.
    """
    stale = os.path.join(output_folder, "prediction.json")
    if os.path.isfile(stale):
        os.remove(stale)
    for name in os.listdir(output_folder):
        if BUNDLE_DCM_RE.match(name):
            os.remove(os.path.join(output_folder, name))


def _deliver_to_platform(staging_dir: str, study_uid: str, inference_id: str) -> str:
    """Copy the staged bundle into the tree the RAD backend reads. Returns its path.

    <upload_root>/<StudyInstanceUID>/infarct_model/<inference_id>/, the layout
    the reference bundles under process/Deep_Infarct_radax have and the one
    pipeline_infarct_torch.py:425 builds inline.

    Published by renaming a hidden sibling rather than by copying into the final
    name. The backend scans this tree on a timer and has no way to tell a
    directory that is still being written from a finished one, so a study copied
    in place can be imported with only the SEGs that had landed by then — and
    nothing downstream would ever report that the rest were missing.
    """
    model_root = os.path.join(_upload_root(), study_uid, MODEL_NAME)
    final_dir = os.path.join(model_root, inference_id)
    partial_dir = os.path.join(model_root, f".{inference_id}.partial")

    os.makedirs(model_root, exist_ok=True)
    if os.path.isdir(partial_dir):
        shutil.rmtree(partial_dir)
    shutil.copytree(staging_dir, partial_dir)

    if os.path.isdir(final_dir):
        # inference_id is a fresh uuid4 per run, so this is only reachable if a
        # previous run died between this rename and its return — the contents
        # are unreferenced by any prediction.json anyone has seen.
        shutil.rmtree(final_dir)
    os.rename(partial_dir, final_dir)
    return final_dir


def _publish_to_output_folder(staging_dir: str, output_folder: str) -> None:
    """Move the staged bundle into output_folder, replacing the previous run's.

    Not a directory swap: output_folder also holds this study's NIfTI
    predictions from the dispatcher, so the bundle files are replaced one at a
    time. staging_dir is a subdirectory of output_folder precisely so every one
    of those moves is a same-filesystem os.replace, which cannot leave a
    half-written file behind.
    """
    _clear_stale_bundle(output_folder)
    for name in sorted(os.listdir(staging_dir)):
        os.replace(os.path.join(staging_dir, name), os.path.join(output_folder, name))


def _emit_series(
    utils,
    series_name: str,
    bundle: Tuple[Any, Any, List[Any]],
    lesions: List[Dict[str, Any]],
    colors: Dict[str, str],
    dest_dir: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Write one DICOM-SEG per lesion for one series; return its detections.

    Key names and their order in each detection follow the RADAX samples under
    process/Deep_Infarct_radax verbatim. Note what is absent there and so absent
    here: no sub_location, even though build_infarct.py emits one.
    """
    image, first_dcm, source_images = bundle
    annotated_uid = str(getattr(first_dcm, "SeriesInstanceUID", ""))
    if not annotated_uid:
        raise RuntimeError(f"{series_name}: source DICOM has no SeriesInstanceUID")
    n_slices = len(source_images)

    detections: List[Dict[str, Any]] = []
    for lesion in lesions:
        location = lesion["location"]
        color = TOTAL_COLOR if lesion["label"] == "Total" else colors.get(
            location, REGION_FALLBACK_COLOR)
        template = utils.get_dicom_seg_template(
            f"Infarct_{series_name}", {1: {"SegmentLabel": location, "color": color}})
        dcm_seg = utils.make_dicomseg_file(
            lesion["mask"].astype("uint8"), image, first_dcm, source_images, template)

        seg_series_uid = str(getattr(dcm_seg, "SeriesInstanceUID", ""))
        seg_sop_uid = str(getattr(dcm_seg, "SOPInstanceUID", ""))
        if not (seg_series_uid and seg_sop_uid):
            raise RuntimeError(
                f"{series_name}/{lesion['label']}: DICOM-SEG has no "
                f"SeriesInstanceUID/SOPInstanceUID — it cannot be named or referenced")

        # '<uid>_<label>.dcm'. The underscore is contract, not decoration: since
        # RADAX-615 the platform pairs SEGs to detections with
        # startsWith(series_instance_uid + "_"), so a bare '<uid>.dcm' is
        # skipped — and skipped silently, leaving a finding with no mask.
        dcm_seg.save_as(os.path.join(dest_dir, f"{seg_series_uid}_{lesion['label']}.dcm"))

        # The aggregate row is marked with role, not with a location. The
        # platform's rule is positional-free but strict about absence: at most
        # one detection carries role, and an ordinary finding must not carry the
        # key at all -- there is no role: "finding", so a present-but-empty role
        # would read as a malformed total rather than as a normal lesion.
        detection = {
            "annotated_series_instance_uid": annotated_uid,
            "series_instance_uid": seg_series_uid,
            "sop_instance_uid": seg_sop_uid,
            "label": lesion["label"],
            "type": lesion["type"],
        }
        if lesion["label"] == "Total":
            detection["role"] = "total"
        else:
            detection["location"] = location
        detection.update({
            "volume_ml": lesion["volume_ml"],
            "mean_adc": lesion["mean_adc"],
            # Clamped per series: the lesion index is computed once from the
            # prediction grid, but ADC and DWI1000 need not have the same
            # instance count, and an out-of-range index is a viewer crash.
            "main_seg_slice": max(0, min(int(lesion["main_seg_slice"]), n_slices - 1)),
            "probability": lesion["prob_max"],
            "mask_index": lesion["mask_index"],
        })
        detections.append(detection)

    n_total = sum(1 for d in detections if d.get("role") == "total")
    if n_total != 1:
        raise ValueError(
            f"expected exactly one detection with role='total', got {n_total}. "
            f"The platform treats a second one as a malformed result."
        )

    return annotated_uid, detections


def _write_excel(path_excel: str, study_id: str, lesions: List[Dict[str, Any]]) -> str:
    """Infarct_Pred_list.xlsx — the clinical record _reference produced.

    study_id is <patient>_<date>_<modality>_<accession>, same convention as
    aneurysm. Voxels is an addition: the territory cut-off is now a voxel count,
    so the audit trail has to show the number that was compared against it.
    """
    parts = study_id.split("_")
    row: Dict[str, Any] = {
        "PatientID": parts[0] if parts else study_id,
        "StudyDate": parts[1] if len(parts) > 1 else "",
        "AccessionNumber": parts[3] if len(parts) > 3 else "",
        "Lesion_Number": len(lesions),
    }
    for lesion in lesions:
        prefix = f"{lesion['mask_index']}_"
        row[f"{prefix}size"] = lesion["volume_ml"]
        row[f"{prefix}Voxels"] = lesion["voxel_count"]
        row[f"{prefix}Prob_max"] = lesion["prob_max"]
        row[f"{prefix}Prob_mean"] = lesion["prob_mean"]
        row[f"{prefix}Confirm"] = ""
        row[f"{prefix}type"] = lesion["type"]
        row[f"{prefix}Location"] = lesion["location"]
        row[f"{prefix}MeanADC"] = lesion["mean_adc"]

    out_path = os.path.join(path_excel, "Infarct_Pred_list.xlsx")
    pd.DataFrame([row]).to_excel(out_path, index=False)
    return out_path


def _notify_platform_complete(prediction_json_path: str, model_name: str) -> None:
    """POST inference completion to the RAD backend (AI_APP_INFERENCE_COMPLETE).

    Same shape as docker/aneurysm/postprocess.py. Best-effort: the bundle on
    disk is the deliverable, and failing the whole study because the backend was
    briefly unreachable would throw away good results.

    The caller must not reach this until the bundle is in the upload tree. The
    POST is what flips the case to "done" on the worklist, so sending it while
    the result is still only in output_folder tells a reader to go look at
    something that is not there.
    """
    url = os.environ.get("AI_APP_INFERENCE_COMPLETE")
    if not url:
        logger.warning("AI_APP_INFERENCE_COMPLETE not set — skip platform notify")
        return
    try:
        with open(prediction_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        study_uid = (data.get("input_study_instance_uid") or [""])[0]
        inference_id = str(data.get("inference_id") or "")
        if not (study_uid and inference_id):
            logger.warning("notify: missing study_uid or inference_id in %s",
                           prediction_json_path)
            return
        import requests

        # The backend's InferenceSuccessRequest schema requires the literal
        # result: "success"; without it the POST 400s and the case stays
        # "running" forever on the worklist.
        r = requests.post(
            url,
            json={
                "studyInstanceUid": study_uid,
                "modelName": model_name,
                "result": "success",
                "inferenceId": inference_id,
            },
            timeout=30,
        )
        if r.ok:
            logger.info("[notify] %s -> %s (%d)", model_name, url, r.status_code)
        else:
            # The body carries the reason and the status code does not. A 500
            # reading "Model not found: infarct" is a missing AiModel row on the
            # platform and nothing to fix here; a 400 is our payload. Logging
            # only the number makes those two look identical, which cost a
            # manual re-POST to tell apart.
            logger.warning("[notify] %s -> %s (%d): %s",
                           model_name, url, r.status_code, r.text[:400])
    except Exception as exc:
        logger.warning("[notify] %s failed: %s", model_name, exc)


def infarct_postprocess(study_id: str, process_dir: str, output_folder: str) -> bool:
    try:
        t0 = time.time()
        logger.info("[postprocess] %s start", study_id)

        utils = _dicomseg_utils()

        path_nnunet = os.path.join(process_dir, "nnUNet")
        path_nii = os.path.join(path_nnunet, "Image_nii")
        path_dcm = os.path.join(path_nnunet, "Dicom")
        path_excel = os.path.join(path_nnunet, "excel")

        pred_path = os.path.join(path_nnunet, "Pred.nii.gz")
        prob_path = os.path.join(path_nnunet, "Prob.nii.gz")
        adc_path = os.path.join(path_nii, "ADC.nii.gz")
        # process_dir/SynthSEG.nii.gz, not SynthSeg_merged: the merged infarct10
        # volume is the model's third input channel and only carries 0-9. The
        # 1xx/2xx/3xx territory vocabulary the platform reports lives here, and
        # neither volume can be derived from the other.
        synth_path = os.path.join(process_dir, "SynthSEG.nii.gz")

        missing = [p for p in (pred_path, prob_path, adc_path, synth_path)
                   if not os.path.isfile(p)]
        if missing:
            logger.error("[postprocess] missing inputs: %s", missing)
            return False

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(path_excel, exist_ok=True)

        # The bundle is assembled in a private directory and only moved into
        # place once every step that can fail has succeeded. Inside
        # output_folder, not under process_dir, so the final publish is a
        # same-filesystem os.replace per file. Everything before that point
        # leaves the previous run's result untouched.
        staging_dir = os.path.join(
            output_folder, f".staging-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        os.makedirs(staging_dir)

        try:
            region_labels, colors = _load_dataset_config()

            # Everything is reoriented to ('S','P','L') up front so statistics
            # and DICOM-SEG share one array space. The alternative — stats in
            # NIfTI order, masks in DICOM order — is how a lesion ends up
            # measured in one place and drawn in another, with nothing to flag
            # the mismatch.
            pred = utils.get_array_to_dcm_axcodes(pred_path)
            prob = utils.get_array_to_dcm_axcodes(prob_path)
            adc = utils.get_array_to_dcm_axcodes(adc_path)
            synth = np.asanyarray(
                utils.get_array_to_dcm_axcodes(synth_path)).astype(np.int32)

            # Zooms come from the header, not the reoriented array: reorientation
            # permutes axes but the product of the three spacings is invariant.
            # Sliced to 3 because a 4D header reports a fourth (time) zoom that
            # would otherwise multiply into the volume.
            voxel_ml = float(np.prod(nib.load(pred_path).header.get_zooms()[:3]) / 1000.0)

            for name, arr in (("Prob", prob), ("ADC", adc), ("SynthSEG", synth)):
                if arr.shape != pred.shape:
                    logger.error("[postprocess] %s shape %s != Pred %s",
                                 name, arr.shape, pred.shape)
                    return False

            series_bundles: Dict[str, Tuple[Any, Any, List[Any]]] = {}
            for name in ("DWI1000", "ADC"):
                series_dir = os.path.join(path_dcm, name)
                if not os.path.isdir(series_dir):
                    logger.error("[postprocess] missing DICOM series dir: %s", series_dir)
                    return False
                _, image, first_dcm, source_images = utils.load_and_sort_dicom_files(
                    series_dir)
                if not source_images:
                    logger.error("[postprocess] no DICOM instances in %s", series_dir)
                    return False
                series_bundles[name] = (image, first_dcm, source_images)

            dwi_first_dcm = series_bundles["DWI1000"][1]
            study_uid = str(getattr(dwi_first_dcm, "StudyInstanceUID", ""))
            patient_id = str(getattr(dwi_first_dcm, "PatientID", ""))
            if not study_uid:
                # The delivery path is <upload_root>/<StudyInstanceUID>/... and
                # the completion POST is keyed on it too, so without it there is
                # no bundle to hand over and nothing to notify about. Fail here
                # rather than produce a result nobody can find.
                logger.error("[postprocess] %s: DWI1000 DICOM carries no "
                             "StudyInstanceUID — the bundle has no address",
                             study_id)
                return False

            # Whole-lesion union across every connected component, then the
            # per-territory split. Pred is already CC-numbered by inference.py;
            # the components are not re-derived here, so the numbering stays
            # consistent with what inference logged.
            total_mask = pred > 0
            lesions: List[Dict[str, Any]] = []
            if total_mask.any():
                total = _stats_for_mask("Total", total_mask, voxel_ml, adc, prob)
                total["mask"] = total_mask
                lesions.append(total)
                territory = _assign_territories(synth, total_mask, region_labels)
                lesions.extend(_compute_region_stats(
                    region_labels, territory, total_mask, voxel_ml, adc, prob))
            else:
                logger.info("[postprocess] %s: no infarct predicted — empty bundle",
                            study_id)

            for index, lesion in enumerate(lesions, start=1):
                lesion["mask_index"] = index
                # mask_index 1 is the union and is labelled Total; the
                # territories start at A1, so label number lags mask_index by one.
                lesion["label"] = "Total" if index == 1 else f"A{index - 1}"
                lesion["main_seg_slice"] = _main_seg_slice(lesion["mask"])

            # Already in this order by construction; sorting explicitly is what
            # makes that a guarantee rather than an accident. See _label_order
            # for the lexicographic-A10 failure this replaces.
            lesions.sort(key=lambda item: _label_order(item["label"]))

            # Before any of the delivery work, mirroring aneurysm's step 3. The
            # Excel is a side record; a pandas or openpyxl failure while writing
            # it must not be able to veto a bundle, and once the bundle has been
            # handed to the platform returning False would be a lie anyway.

            dwi_series_uid, dwi_detections = _emit_series(
                utils, "DWI1000", series_bundles["DWI1000"], lesions, colors, staging_dir)
            adc_series_uid, adc_detections = _emit_series(
                utils, "ADC", series_bundles["ADC"], lesions, colors, staging_dir)

            inference_id = str(uuid.uuid4())
            payload: Dict[str, Any] = {
                "inference_id": inference_id,
                "inference_timestamp": _utc_iso_now_ms(),
                "input_study_instance_uid": [study_uid],
                # Both series the top-level detections annotate. The model reads
                # DWI1000 and ADC together, so listing only one understates what
                # the result depends on.
                "input_series_instance_uid": [u for u in (dwi_series_uid, adc_series_uid) if u],
                "model_id": INFARCT_MODEL_ID,
                "patient_id": patient_id,
                # One flat list across both source series. Each row already
                # carries its own annotated_series_instance_uid, which is what
                # the platform records as the source, so a second series needs
                # no second container. See the module docstring for why
                # reformatted_series is the wrong home for the ADC rows.
                "detections": dwi_detections + adc_detections,
                "reformatted_series": [],
            }

            prediction_path = os.path.join(staging_dir, "prediction.json")
            with open(prediction_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=4)

            # The bundle is only usable if every detection has both its SEGs. A
            # UID collision or a save that lost a file would otherwise ship a
            # half-populated study that looks fine in the json.
            # One SEG file per detection row now that both series share the
            # list, so the count is a direct equality rather than a factor of
            # two. The platform resolves each row's file by the
            # "<seg_series_uid>_" prefix, so a row without its file is a 404 at
            # upload time, not a rendering glitch.
            n_dcm = len([n for n in os.listdir(staging_dir) if BUNDLE_DCM_RE.match(n)])
            n_det = len(payload["detections"])
            if n_dcm != n_det:
                logger.error("[postprocess] %s: %d SEG files staged, expected %d "
                             "(one per detection row)", study_id, n_dcm, n_det)
                return False

            # Written here rather than before the SEG loop: a run that dies during
            # SEG generation discards its bundle, and must not leave behind an
            # Infarct_Pred_list.xlsx describing results nobody received. Still
            # ahead of delivery, so an Excel failure aborts before the platform
            # sees anything.
            # Visual report. Written under the process directory only -- not
            # staged, not delivered, not declared in reformatted_series. The
            # RADAX worked examples carry no report series, so publishing one
            # would put a key in prediction.json the platform has not agreed to
            # parse; that is a conversation to have before it ships, not a
            # decision to make here.
            #
            # A failure here does not fail the run. The bundle is the clinical
            # payload and it is already complete by this point; losing a
            # visualisation should not discard a correct result. It is logged at
            # warning so it cannot pass unnoticed.
            try:
                import report as _report
                produced = _report.generate_report(
                    process_dir, patient_id, lesions, region_labels, colors)
                logger.info("[postprocess] report: %d PNG, %d DICOM",
                            len(produced["png"]), len(produced["dcm"]))
            except Exception as exc:
                logger.warning("[postprocess] report generation failed: %s",
                               exc, exc_info=True)

            excel_path = _write_excel(path_excel, study_id, lesions)
            delivered_dir = _deliver_to_platform(staging_dir, study_uid, inference_id)
            _publish_to_output_folder(staging_dir, output_folder)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

        logger.info("[postprocess] %s: %d detections, %d SEG files -> %s "
                    "(+ %s), excel %s",
                    study_id, len(dwi_detections), n_dcm, delivered_dir,
                    output_folder, excel_path)

        # Only now. The bundle is in the tree the backend reads, so the case
        # flipping to "done" points at something a reader can open.
        _notify_platform_complete(
            os.path.join(delivered_dir, "prediction.json"), MODEL_NAME)

        logger.info("[postprocess] %s done (%.0fs)", study_id, time.time() - t0)
        return True

    except Exception as exc:
        logger.error("[postprocess] %s failed: %s", study_id, exc, exc_info=True)
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Infarct postprocessing")
    p.add_argument("--study_id", required=True)
    p.add_argument("--process_dir", required=True)
    p.add_argument("--output_folder", required=True)
    a = p.parse_args()
    sys.exit(0 if infarct_postprocess(a.study_id, a.process_dir, a.output_folder) else 1)
