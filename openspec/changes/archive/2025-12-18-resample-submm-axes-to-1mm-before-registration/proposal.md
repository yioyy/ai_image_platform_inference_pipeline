# 配準前逐軸將 sub-mm spacing 升至 1mm（只改 sub-mm 軸）

## Why
前一版規格與實作使用「1mm isotropic（全軸改成 1mm）」；但實務上我們只需要針對 spacing < 1mm 的軸做下採樣即可，其他軸保留原始 spacing 可避免不必要的資訊損失與額外插值誤差。

## What Changes
- 當任一軸 spacing < 1mm 時，僅將該軸重採樣到 1mm（例如 0.5→1.0），其餘軸不變（例如 5.0→5.0）。
- 保持 label 影像使用最近鄰插值（nearest neighbour）。
- 配準矩陣仍需換算回原始影像 grid 的 `.mat`，輸出保持 baseline 原空間。


