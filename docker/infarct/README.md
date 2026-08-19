# Infarct container — 建置中

RADAX 平台的 infarct 模型上線工作。計畫與驗收條件見
`Desktop/infarct_pipeline_plan.md`(第 2 版)。

## `_reference/` 是什麼

從 ws2040 `/home/chuan/orthanc_combine_code/pipeline/code/` 取回的 needFollowup 時期
infarct 實作。**這是參考資料,不是要執行的程式。** 納入版控的理由是它先前只存在於單一台
機器上、未受任何版本控制,而它是唯一一份記錄著臨床計算細節(分區統計、體積、mean ADC、
DICOM-SEG 產生)的來源。

搬過來時原封不動,沒有修改。實際要跑的程式會另外寫在本目錄下,結構比照
`../aneurysm/`(preprocess / inference / postprocess / server 四段式)。

`_reference/` 裡**必須丟掉**的部分:`orthanc_zip_upload`、`upload_json_aiteam`(舊平台 API)、
`$9` needFollowup json、PNG 報告燒錄。**必須保留**的是臨床計算本身。

## 權重

已放入 docker volume `compose_volume_weights_nnunet`:

```
nnUNet_results/Dataset300_DeepInfarct_v2_3ch/nnUNetTrainer__nnUNetPlans__3d_fullres/
    plans.json          8,226 B
    dataset.json        1,429 B
    fold_0/checkpoint_best.pth   1,109,402,434 B
    sha256 81f2fe394db723ce1b7a533c8a55d493a13d454157c14485ca41af95cf1342b6
```

來源:ws2040 `mutp/experiments/infarct_25d_s2_deepconcat_maskaware_seed3`。
雜湊與來源端逐位元一致,權限已改為 uid 1000 可讀。

## 模型的三個輸入通道

| 通道 | 內容 | 正規化 |
|---|---|---|
| `_0000` | DWI1000,**已去顱骨** | `ZScoreBrainNormalization` |
| `_0001` | ADC,**已去顱骨** | `ADCNormalization`(除以 600) |
| `_0002` | `SynthSeg_merged` 0–9 | `NoNormalization` |

去顱骨遮罩必須是 **`SynthSeg_merged > 0`**(infarct10 合併規則的輸出),
不是 david 1xx 詞彙那條衍生。兩條衍生來自同一次 SynthSeg,互相不可推導:

```
SynthSeg 一次跑（aparc+aseg）
  ├─ resampleSynthSEG2original(..., 'DWI')      → david 1xx/2xx/3xx → 臨床 location（34 territories）
  └─ resampleSynthSEG2original(..., 'synthseg') → aparc+aseg
       └─ apply_merge_rule(arr, "infarct10")    → 0–9 → 模型通道 2 + 去顱骨遮罩
```

⚠ 不可使用 synthseg33 做 merge:FreeSurfer 3/42(左右大腦皮質)不在 33 類裡,
整片 cortex 會靜默掉進背景,腦體積少 35%。

## 尚未決定 / 尚未實作

- **SynthSeg 放哪個階段。** 若在 inference 階段呼叫,會在 `rad_gpu_0` 鎖內再取鎖 → 死結
  (aneurysm C.5 每個 study 白等 292 秒就是這個)。放 preprocess 最乾淨,但
  `_call("preprocess", 600)` 對所有模型寫死 600 秒,SynthSeg 排隊時可能不夠。
- **`resampleSynthSEG2original` 的目錄約定。** 兩個版本簽名不同
  (`util.py` 4 參數需要 `path/dicom/nii/` 佈局;`util_aneurysm.py` 3 參數),
  容器要用哪個、以及檔案要擺成什麼樣子,還沒定案。
- preprocess / inference / postprocess / server 四支程式本身。
