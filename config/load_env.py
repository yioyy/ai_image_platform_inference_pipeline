from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def load_env_file(path: str | Path) -> Dict[str, str]:
    """
    Load KEY=VALUE lines into os.environ (no override if already set).
    - ignores blank lines and comments starting with '#'
    - strips surrounding quotes for values
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"env file not found: {p}")

    loaded: Dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip("\"'")  # trim simple quotes
        if not key:
            continue
        if key in os.environ:
            continue
        os.environ[key] = val
        loaded[key] = val
    return loaded


def load_env_from_default_locations() -> Optional[Path]:
    """
    Load env file from:
    - RADX_ENV_FILE (if set)
    - <repo>/config/radax.env (if exists)
    """
    env_file = os.getenv("RADX_ENV_FILE")
    if env_file:
        load_env_file(env_file)
        return Path(env_file)

    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "config" / "radax.env"
    if candidate.exists():
        load_env_file(candidate)
        return candidate

    return None
