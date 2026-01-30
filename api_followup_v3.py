#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Follow-up v3 API (radax).

POST /api/radax/followup
Input: JSON body (follow_up_input.json structure)
Output: JSON body (followup_manifest with items[])
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request

from pipeline_followup_v3 import pipeline_followup_v3

APP = Flask(__name__)

PATH_PROCESS = Path(os.environ.get("RADAX_PATH_PROCESS", "/home/david/pipeline/sean/rename_nifti/"))
PATH_FOLLOWUP_ROOT = Path(os.environ.get("RADAX_PATH_FOLLOWUP_ROOT", "/home/david/pipeline/chuan/process"))
LOG_DIR = Path(os.environ.get("RADAX_LOG_DIR", "/home/david/pipeline/chuan/log"))
FSL_FLIRT_PATH = os.environ.get("RADAX_FSL_FLIRT_PATH", "/usr/local/fsl/bin/flirt")

_LOGGER = logging.getLogger("radax.followup.api")
_CURRENT_LOG_DATE = None
_FILE_HANDLER = None


def _ensure_log_handler() -> None:
    global _CURRENT_LOG_DATE, _FILE_HANDLER
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    date_key = time.strftime("%Y%m%d", time.localtime())
    if _CURRENT_LOG_DATE == date_key and _FILE_HANDLER:
        return
    if _FILE_HANDLER:
        logging.getLogger().removeHandler(_FILE_HANDLER)
        _FILE_HANDLER.close()
    log_path = LOG_DIR / f"{date_key}.log"
    _FILE_HANDLER = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    _FILE_HANDLER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    _FILE_HANDLER.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(_FILE_HANDLER)
    _CURRENT_LOG_DATE = date_key


def _log_event(level: int, message: str, **fields: Any) -> None:
    if fields:
        pairs = " ".join(f"{k}={v}" for k, v in fields.items())
        message = f"{message} {pairs}"
    _LOGGER.log(level, message)


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    current = payload.get("current_study")
    prior_list = payload.get("prior_study_list")
    if not isinstance(current, dict):
        raise ValueError("current_study must be an object")
    if not isinstance(prior_list, list):
        raise ValueError("prior_study_list must be a list")
    root_model_id = str(payload.get("model_id") or "").strip()
    if root_model_id:
        current.setdefault("model_id", root_model_id)
        for item in prior_list:
            if isinstance(item, dict):
                item.setdefault("model_id", root_model_id)
    _validate_required(current, prior_list)
    return payload


def _validate_required(current: Dict[str, Any], prior_list: List[Any]) -> None:
    missing = []
    if not str(current.get("patient_id") or "").strip():
        missing.append("current_study.patient_id")
    if not str(current.get("study_date") or "").strip():
        missing.append("current_study.study_date")
    if not prior_list:
        missing.append("prior_study_list")
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))


def _write_temp_payload(payload: Dict[str, Any]) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        return Path(fp.name)


def _load_manifest(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def _error_response(status_code: int, code: str, message: str, details: Optional[Any] = None):
    body = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return jsonify(body), status_code


def _map_value_error(message: str) -> Tuple[int, str]:
    if "同日找到多個 case 資料夾" in message or "model 不一致" in message:
        return 409, "CONFLICT"
    return 400, "BAD_REQUEST"


@APP.route("/api/radax/followup", methods=["POST"])
def followup_v3():
    _ensure_log_handler()
    request_id = request.headers.get("X-Request-Id") or f"req-{uuid.uuid4().hex[:12]}"
    _log_event(logging.INFO, "followup_request_received", request_id=request_id)

    payload = request.get_json(silent=True)
    if payload is None:
        _log_event(logging.WARNING, "invalid_json", request_id=request_id)
        return _error_response(400, "BAD_REQUEST", "Invalid JSON body")

    temp_path = None
    try:
        payload = _normalize_payload(payload)
        temp_path = _write_temp_payload(payload)
        manifest_path = pipeline_followup_v3(
            input_json=temp_path,
            path_process=PATH_PROCESS,
            model="Aneurysm",
            prediction_json_name=None,
            path_followup_root=PATH_FOLLOWUP_ROOT,
            fsl_flirt_path=FSL_FLIRT_PATH,
        )
        manifest = _load_manifest(Path(manifest_path))
        _log_event(logging.INFO, "followup_request_success", request_id=request_id, status="success")
        return jsonify(manifest)
    except FileNotFoundError as exc:
        _log_event(logging.ERROR, "followup_missing_files", request_id=request_id, error=str(exc))
        return _error_response(404, "NOT_FOUND", "Required files missing", [str(exc)])
    except TimeoutError:
        _log_event(logging.ERROR, "followup_timeout", request_id=request_id)
        return _error_response(408, "TIMEOUT", "Follow-up processing timeout", {"timeout_seconds": 300})
    except ValueError as exc:
        status, code = _map_value_error(str(exc))
        _log_event(logging.ERROR, "followup_bad_request", request_id=request_id, error=str(exc))
        return _error_response(status, code, str(exc))
    except Exception as exc:  # noqa: BLE001
        _log_event(logging.ERROR, "followup_internal_error", request_id=request_id, error=str(exc))
        return _error_response(500, "INTERNAL_ERROR", "Unexpected error", {"error": str(exc)})
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                _log_event(logging.WARNING, "temp_cleanup_failed", request_id=request_id, path=str(temp_path))


def main() -> None:
    host = os.environ.get("RADAX_API_HOST", "0.0.0.0")
    port = int(os.environ.get("RADAX_API_PORT", "5000"))
    _ensure_log_handler()
    _log_event(logging.INFO, "followup_api_start", host=host, port=port)
    APP.run(host=host, port=port)


if __name__ == "__main__":
    main()
