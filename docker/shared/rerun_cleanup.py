"""Start a rerun from a clean slate.

Until now a rerun wrote into whatever the previous run had left behind, and
each model cleared a hand-picked subset of that: aneurysm removed
DeepAneurysm_00001.nii.gz, infarct removed three prediction files, cmb removed
nothing. Everything else stayed -- Pred.nii.gz, the MIP volumes, Dicom/MIP_*,
Dicom/Dicom-Seg, excel/, JSON/ -- and every later stage reads those by name
without being able to tell this run's output from the last one's.

That is not theoretical. The aneurysm pre-flight added on 2026-09-23 checks
that Dicom/MIP_Pitch and Dicom/MIP_Yaw exist before delivering, and a previous
run's MIP folders satisfy it perfectly: a rerun that failed before regenerating
them would have shipped the old MIPs beside new predictions.

So the directory goes, whole, and preprocess rebuilds it. The cost is that
SynthSeg recomputes (~30-40 s, and cmb loses the pre-computed parcellation it
would otherwise reuse). That is the right trade: the reason to rerun is usually
that something about the inputs changed, and a SynthSeg output carried over
from the previous inputs is the same class of bug one layer down.
"""
from __future__ import annotations

import os
import shutil


def reset_process_dir(process_dir: str, study_id: str, logger) -> None:
    """Delete and recreate one study's working directory.

    Refuses anything that does not look like a study folder, because the caller
    builds this path from values that arrive over HTTP. The shape it insists on
    is the one every model uses: <root>/Deep_<Model>/<study_id>.
    """
    path = os.path.abspath(process_dir)
    parent = os.path.dirname(path)

    if not study_id or os.path.basename(path) != study_id:
        logger.error("[reset] refusing %s: basename is not the study id (%s)",
                     path, study_id)
        return
    if not os.path.basename(parent).startswith("Deep_"):
        logger.error("[reset] refusing %s: parent %s is not a Deep_* folder",
                     path, os.path.basename(parent))
        return
    if path.count(os.sep) < 3:
        logger.error("[reset] refusing %s: too close to the filesystem root", path)
        return

    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
            logger.info("[reset] cleared previous run: %s", path)
        except Exception as exc:
            # Root-owned leftovers from an older deployment are the known case.
            # Say so and carry on: a stale file is worse than a slow run, but a
            # refused rerun is worse than both.
            logger.error("[reset] could not clear %s (%s) — continuing with "
                         "whatever is there", path, exc)

    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(path, 0o775)  # worker (gid=1001) writes here too
    except Exception:
        pass


def prune_previous_results(model_root: str, keep_inference_id: str, logger) -> None:
    """Drop this model's earlier run folders, keeping the one just written.

    ⛔ NOT for aneurysm_model. The platform's comparison backfill
    (artifact-backfill.service.ts) resolves an OLD prediction's folder from its
    own inferenceId and, when the folder is gone, records the candidate
    retryable rather than absent -- so it retries every tick, for ever, and
    never succeeds. COMPARISON_TRACKED_MODEL is Aneurysm, so only that model's
    folders carry that obligation; vessel, cmb and infarct are read once at
    import and never again.
    """
    if not (model_root and keep_inference_id):
        return
    if os.path.basename(model_root) == "aneurysm_model":
        logger.error("[prune] refusing %s: the comparison backfill reads this "
                     "model's older runs", model_root)
        return
    if not os.path.isdir(model_root):
        return

    for name in sorted(os.listdir(model_root)):
        if name == keep_inference_id:
            continue
        old = os.path.join(model_root, name)
        if not os.path.isdir(old):
            continue
        try:
            shutil.rmtree(old)
            logger.info("[prune] removed superseded run: %s", old)
        except Exception as exc:
            logger.error("[prune] could not remove %s: %s", old, exc)
