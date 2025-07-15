#!/bin/bash

# Mac M1 專用啟動腳本
# 針對 MPS 記憶體限制進行優化

echo "🍎 Mac M1 MPS 記憶體優化啟動腳本"
echo "=================================="

# 設定 MPS 記憶體優化環境變數 (Apple Silicon 專用)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 設定 Python 模組路徑
export PYTHONPATH="$(dirname "$0")/..:$PYTHONPATH"

# 切換到專案根目錄
cd "$(dirname "$0")/.."

# 顯示當前設定
echo "📊 Apple Silicon MPS 記憶體優化設定："
echo "   PYTORCH_MPS_HIGH_WATERMARK_RATIO: $PYTORCH_MPS_HIGH_WATERMARK_RATIO (按需分配)"
echo "   PYTORCH_ENABLE_MPS_FALLBACK: $PYTORCH_ENABLE_MPS_FALLBACK"
echo "   PYTHONPATH: $PYTHONPATH"
echo "   當前目錄: $(pwd)"
echo ""
echo "💡 說明：WATERMARK_RATIO=0.0 表示 PyTorch 將按需分配 MPS 記憶體，"
echo "   這是解決 SDXL 在 Apple Silicon 上記憶體衝突的標準方法。"

# 檢查可用記憶體
echo ""
echo "💾 系統記憶體資訊："
system_profiler SPHardwareDataType | grep "Memory:"

echo ""
echo "🚀 啟動選項："
echo "1. 主程式 (批次生成)"
echo "2. API 服務器"
echo "3. 測試 SD 1.5 模型"
echo "4. 測試 SD XL 模型 (512x512)"
echo "5. 退出"

read -p "請選擇 (1-5): " choice

case $choice in
    1)
        echo "🔄 啟動主程式..."
        python src/main.py
        ;;
    2)
        echo "🌐 啟動 API 服務器..."
        python src/render/api.py
        ;;
    3)
        echo "🧪 測試 SD 1.5 模型..."
        # 暫時修改配置使用 SD 1.5
        sed -i.bak 's/model: .*/model: AnimefullFinalPrunedFp16Model/' src/config/config.yaml
        python src/main.py
        # 恢復原始配置
        mv src/config/config.yaml.bak src/config/config.yaml
        ;;
    4)
        echo "🧪 測試 SD XL 模型 (512x512 記憶體優化)..."
        # 暫時修改配置使用較小尺寸
        python -c "
import yaml
with open('src/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['model'] = 'StableDiffusionXLModel'
if 'image_size' not in config:
    config['image_size'] = {}
config['image_size']['height'] = 512
config['image_size']['width'] = 512
with open('src/config/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
"
        python src/main.py
        ;;
    5)
        echo "👋 退出"
        exit 0
        ;;
    *)
        echo "❌ 無效選項"
        exit 1
        ;;
esac

echo ""
echo "✅ 執行完成"