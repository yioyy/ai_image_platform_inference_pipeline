# 臨床試驗功能待決問題（Meeting 決策清單）

## P0（會影響資料模型/輸出，務必先定）
1. 去識別策略
   - 是否允許在下載檔保留 `PatientID`？或必須改用 `subject_id`？
   - DICOM SEG 是否需要去識別（移除 PatientName/ID/Institution 等）？
2. 病灶 taxonomy
   - `type/location` 要自由輸入、固定字典、還是 code system（SNOMED/自訂）？
3. DICOM SEG 粒度策略
   - 每個 series 一個 SEG（包含多個 segment） vs 每個 lesion 一個 SEG？
   - `SegmentNumber` 要不要固定沿用 `source_mask_index`？人工新增如何編號？
4. 最小化標註的輸出格式定稿（anomaly box / center point）
   - anomaly box 的 `geometry_payload` 欄位（座標系、slice 定義、2D/3D）怎麼定？
   - center point 「固定 1mm」是直徑還是半徑？與 GT 的重疊計算規則如何定義？
5. 多 series/多 reformatted series 的處理
   - AI JSON 可能含 `reformatted_series[]`；醫師標註要落在哪個 series？是否允許跨 series？

## P1（可先做 MVP，但要先留擴充）
6. 多讀者流程（已定：不共裁）
   - 當前決策：同一 case 可多醫師閱讀（例如資深/young/住院醫師組），但互不仲裁；保留各自結果供效益分析。
7. 版本策略
   - `schema_version` 與 `pipeline_version/model_version` 來源與命名規則
8. 匯出檔命名與封裝
   - 單案下載：zip 內含 JSON + CSV/Excel + SEG？
   - 批次下載：以 Study/Subject 分資料夾？
9. 權限與審計保存期限
   - 當前決策：`export_annotations` 限制後端/Ops/AI team；`export_excel` 為特殊權限（通常給寫計劃/分析醫師或試驗管理者）。
   - 稽核紀錄保存多久、是否需要不可竄改（WORM）？
10. AIAA 回傳與標註資料回流
   - 目標是把「人工標註」轉成哪一種訓練資料格式？（NIfTI/SEG/COCO-like）
   - 是否需要回傳到既有 AIAA 介面？介面規格與責任歸屬？

