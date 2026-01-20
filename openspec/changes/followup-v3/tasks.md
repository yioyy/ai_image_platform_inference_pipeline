## 1. 規格與輸入輸出對齊
- [ ] 1.1 定義 `follow_up_input.json` 欄位需求與排序規則（study_date + model_id 判定 model）
- [ ] 1.5 定義輸入根目錄（rename_nifti）與輸出根目錄（Deep_FollowUp）
- [ ] 1.2 定義 case 資料夾定位規則（patient_id + study_date）
- [ ] 1.3 明確輸出檔案與覆寫策略（prediction.json + follow_up）
- [ ] 1.4 主體規則：一律以日期新者為主體產出 follow_up

## 2. NIfTI 對位流程（沿用 v1）
- [ ] 2.1 規範必要檔名（Aneurysm：Pred_Aneurysm / SynthSEG_Aneurysm / Pred_Aneurysm_rdx_aneurysm_pred_json.json；CMB：Pred_CMB / SynthSEG_CMB / Pred_CMB_rdx_cmb_pred_json.json）
- [ ] 2.2 取得 forward / inverse matrix
- [ ] 2.3 以 inverse matrix 回推 main_seg_slice 與 sorted

## 3. 病灶配對與狀態
- [ ] 3.1 overlap>0 的一對一配對
- [ ] 3.2 3 狀態輸出（new/stable/regress）
- [ ] 3.3 欄位簡化（new/regress 只記單邊 mask_index + slice）

## 4. 輸出格式（grouped）
- [ ] 4.1 prediction.json 加 follow_up 欄位（grouped 格式）
- [ ] 4.2 sorted 欄位輸出格式確定（含轉換需求）

## 5. Manifest 與單次通知
- [ ] 5.1 定義 `followup_manifest.json` schema
- [ ] 5.2 單次通知 payload / endpoint 與狀態欄位

