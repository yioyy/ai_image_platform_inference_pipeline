"""
dicomseg utils 匯出入口。

注意：`code_ai.pipeline.dicomseg.base` 會依賴 pydantic 等較重的套件。
為了讓純工具函數（例如讀 DICOM / NIfTI 的 utils.base）在精簡環境也能被 import，
這裡對 `dicomseg.base` 採用可選匯入。
"""

try:
    from code_ai.pipeline.dicomseg.base import *  # noqa: F401,F403
except Exception:
    # 缺少 pydantic 等依賴時允許跳過
    pass

from code_ai.pipeline.dicomseg.utils.base import *  # noqa: F401,F403
