import pathlib
import os


def load_dotenv():
    """
    嘗試載入 .env（若環境沒有安裝 python-dotenv 則跳過）。
    這樣可以避免在純推理/純腳本環境中因缺少依賴而無法 import 本專案。
    """
    try:
        from dotenv import load_dotenv as _load_dotenv  # type: ignore
    except Exception:
        return

    _ = os.getenv("ENV_STATE", "dev")
    _load_dotenv()
    # load_dotenv(f'.env.{env_state}',override=True)

load_dotenv()

PYTHON3 = os.getenv("PYTHON3")