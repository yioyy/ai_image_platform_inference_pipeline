## 1. 規格與輸入輸出對齊
- [ ] 1.1 定義 `00130846_followup_input.json` 欄位需求（ids / needFollowup / models / mask）
- [ ] 1.2 定義 `study_date` 正規化與排序規則（YYYYMMDD）
- [ ] 1.3 模型判定規則：model_type 對應模型（Aneurysm/CMB/WMH/Infarct）
- [ ] 1.4 主體規則：一律以日期新者為主體產出 follow-up
- [ ] 1.5 `ids` 與 `needFollowup` 的對應來源說明（後續輸入）
- [ ] 1.6 定義輸入根目錄（rename_nifti）與輸出根目錄（Deep_FollowUp）
- [ ] 1.7 定義 case 資料夾定位規則（patient_id + study_date）
- [ ] 1.8 明確輸出檔案與覆寫策略（Pred_*_platform_json → Followup_*_platform_json_new）
- [ ] 1.9 若指定 model/model_type，需驗證存在於 needFollowup.models
- [ ] 1.10 驗證失敗時需記錄 log 並跳過執行

## 2. NIfTI 對位流程（沿用 v1）
- [ ] 2.1 規範必要檔名（Pred_<Model> / SynthSEG_<Model> / Pred_<Model>_A1.dcm / Pred_<Model>_platform_json.json）
- [ ] 2.2 取得 forward / inverse matrix
- [ ] 2.3 以 inverse matrix 回推 main_seg_slice
- [ ] 2.4 生成 sorted_slice（current/prior 對應）
- [ ] 2.5 確認 sorted_slice 索引從 1 開始

## 3. 病灶配對與狀態
- [ ] 3.1 overlap>0 的一對一配對規則
- [ ] 3.2 3 狀態輸出（new/stable/regress）
- [ ] 3.3 new/stable/regress 欄位對應（mask_index / diameter / volume / UID）
- [ ] 3.4 未標註側 main_seg_slice 回推規則

## 4. 輸出格式（platform_json）
- [ ] 4.1 `study.model[].followup[]` 欄位定義與來源
- [ ] 4.2 `mask.model[].series[].instances[].followup[]` 欄位定義與來源
- [ ] 4.3 `sorted_slice` 置於根層級，與 `sorted` 同階級
- [ ] 4.4 `sorted_slice` 以 UID 區分每一組 current/prior
- [ ] 4.5 空資料輸出規則（mask[] 空、followup 空）

## 5. 通知與錯誤處理
- [ ] 5.1 多主體情境的多次通知規則
- [ ] 5.2 缺檔或同日多筆 case 的錯誤回報格式


