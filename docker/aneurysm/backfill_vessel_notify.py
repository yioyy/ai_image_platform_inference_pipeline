"""Backfill vessel_model notify for all cases from 2026-06-29 onwards.

Reads /home/david/ai-inference-result/<study_uid>/vessel_model/<infer_id>/prediction.json
and POSTs to the platform's inference-complete endpoint.

Usage:
    DRY_RUN=1 python3 backfill_vessel_notify.py   # count + preview only
    python3 backfill_vessel_notify.py             # actually POST
"""
import json
import os
import glob
import sys
from datetime import datetime, timezone, timedelta

try:
    import requests
except ImportError:
    print("requests missing; install with pip", file=sys.stderr)
    sys.exit(1)

URL = os.environ.get("AI_APP_INFERENCE_COMPLETE",
                     "http://localhost:4000/v1/ai-inference/inference-complete")
ROOT = "/home/david/ai-inference-result"
# Include cases from 2026-06-29 00:00 Asia/Taipei (== 2026-06-28 16:00 UTC).
SINCE = datetime(2026, 6, 28, 16, 0, 0, tzinfo=timezone.utc)
MODEL_NAME = "vessel_model"
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

pattern = os.path.join(ROOT, "*", "vessel_model", "*", "prediction.json")
files = sorted(glob.glob(pattern))
print(f"scanned {len(files)} vessel prediction.json files under {ROOT}")

todo = []
skipped_old = 0
skipped_bad = 0
for path in files:
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  ! parse fail {path}: {exc}", file=sys.stderr)
        skipped_bad += 1
        continue
    ts_str = data.get("inference_timestamp")
    study_uid = (data.get("input_study_instance_uid") or [""])[0]
    inference_id = str(data.get("inference_id") or "")
    if not (ts_str and study_uid and inference_id):
        skipped_bad += 1
        continue
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        skipped_bad += 1
        continue
    if ts < SINCE:
        skipped_old += 1
        continue
    todo.append((ts, study_uid, inference_id, path))

print(f"  skipped_bad (parse or missing fields) : {skipped_bad}")
print(f"  skipped_old (< 2026-06-29 UTC)        : {skipped_old}")
print(f"  to notify                             : {len(todo)}")

if DRY_RUN:
    print("\n[DRY_RUN] first 10 to notify:")
    for ts, s, i, p in todo[:10]:
        print(f"  ts={ts.isoformat()}  study={s[-20:]}  infer={i[:8]}...")
    print("\nSet DRY_RUN=0 to actually POST.")
    sys.exit(0)

print(f"\nPOSTing to {URL} ...")
ok, bad = 0, 0
for ts, study_uid, inference_id, path in todo:
    payload = {
        "studyInstanceUid": study_uid,
        "modelName": MODEL_NAME,
        "result": "success",
        "inferenceId": inference_id,
    }
    try:
        r = requests.post(URL, json=payload, timeout=30)
        if r.status_code // 100 == 2:
            ok += 1
        else:
            bad += 1
            print(f"  ! HTTP {r.status_code} for study={study_uid[-20:]} infer={inference_id[:8]}: {r.text[:200]}",
                  file=sys.stderr)
    except Exception as exc:
        bad += 1
        print(f"  ! POST fail study={study_uid[-20:]}: {exc}", file=sys.stderr)

print(f"\ndone: ok={ok} bad={bad}")
