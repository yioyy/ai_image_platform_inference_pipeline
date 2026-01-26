## 1. 規格更新
- [ ] 1.1 定義 `sorted_slice` 新格式（雙向嚴格對照）
- [ ] 1.2 定義 `sorted` / `reverse-sorted` 的使用情境（current / prior 滾輪主導）
- [ ] 1.3 定義對照補齊規則（空白區段與中段補值）
- [ ] 1.4 定義矩陣方向規則（reverse-sorted 使用原始矩陣）

## 2. 文件同步
- [ ] 2.1 更新 output_schema.md 的 `sorted_slice` schema 與範例
- [ ] 2.2 補充新欄位名稱 `prior_slice` / `current_slice` 定義

## 3. 程式調整
- [x] 3.1 更新 sorted_slice 生成邏輯為一對一對照
- [x] 3.2 加入補齊邏輯（前段、中段、尾段）
- [x] 3.3 reverse-sorted 使用原始矩陣方向計算

