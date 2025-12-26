
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:18:23 2020

這一版會在1個group預測君彥的結果，另一個group預測nnU-Net的結果

@author: chuan
"""

import warnings
warnings.filterwarnings("ignore") # 忽略警告输出

# 所有import移到最上方，刪除重複與未使用的import
import os
import time
import numpy as np
import logging
import pathlib
import requests
import json

study_instance_uid = '1.2.840.113820.7000602646.918.821412170060.8'
aneurysm_inference_id = '704fc56a-a55a-46fe-9478-0c04a1046292'
vessel_inference_id = '14f5ac0d-b8bc-4dfc-87b8-7ed1777f6bdb'
            
# 發送POST請求到 /v1/ai-inference/inference-complete
# 請修改為正確的 API 端點，例如：
# api_url = 'http://localhost:8080/v1/ai-inference/inference-complete'
# api_url = 'http://10.103.1.193:3000/v1/ai-inference/inference-complete'
api_url = 'http://localhost:24000/v1/ai-inference/inference-complete'  # TO: 請修改為正確的 API 端點

# 需求：aneurysm_model / vessel_model 各呼叫一次
api_payloads = [
    {
        "studyInstanceUid": study_instance_uid,
        "modelName": "aneurysm_model",
        "inferenceId": aneurysm_inference_id,
    },
    {
        "studyInstanceUid": study_instance_uid,
        "modelName": "vessel_model",
        "inferenceId": vessel_inference_id,
    },
]
for payload in api_payloads:
    try:
        response = requests.post(api_url, json=payload)
        print(f"[API POST] payload={payload} Status Code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"[API POST Error] payload={payload} err={str(e)}")

