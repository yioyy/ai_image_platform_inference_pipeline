#!/bin/bash
# Program:
#測試用sh呼叫新的環境執行model stroke predict的運算 0:檔名，1開始為參數
#測試用sh呼叫conda env
# History:
# 2021/09/23 3rd
# 1. 進入你的 wheels 資料夾
cd wheels

# 2. 建 whl 檔
for f in *.tar.gz; do
    echo "🔨 正在轉換: $f"
    pip wheel "$f" -w . --no-deps
done

# 3. 檢查哪些還是 tar.gz (代表轉換失敗)
echo "✅ 檢查結果:"
for f in *.tar.gz; do
    pkg=$(echo "$f" | sed 's/-[0-9].*//')   # 抓套件名稱
    if ls "$pkg"-*.whl 1>/dev/null 2>&1; then
        echo "✔ 已成功轉換: $f"
    else
        echo "❌ 尚未生成 whl: $f"
    fi
done

