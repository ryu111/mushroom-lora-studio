# 蘑菇角色生成器 (Mushroom LoRA Studio)

這是一個基於Stable Diffusion和LoRA技術的卡通蘑菇角色創作工作室，能夠生成具有一致性外觀、多種動作和表情的角色圖像。

![蘑菇角色示例](assets/reference_images/dgu_01.png)

## 功能特點

- 使用LoRA微調模型生成特定風格的卡通蘑菇角色
- 支持多種動作和表情組合
- 自動去除背景，生成透明PNG圖像
- 支持測試模式，評估模型在不同角度和表情下的表現
- 可配置的生成參數，包括引導尺度、推理步數等
- 支持批量生成多張圖像

## 系統要求

- Python 3.8+
- CUDA兼容的GPU (推薦用於加速生成)
- 至少8GB RAM

## 安裝說明

1. 克隆此倉庫：

```bash
git clone https://github.com/yourusername/mushroom-lora-studio.git
cd mushroom-lora-studio
```

2. 運行生成腳本，它會自動創建虛擬環境並安裝所需依賴：

```bash
./scripts/run_generate.sh
```

## 使用方法

### 基本使用

使用默認動作和表情生成圖像：

```bash
./scripts/run_generate.sh
```

### 使用隨機動作和表情

```bash
./scripts/run_generate.sh -r
```

### 使用測試模式

測試模式用於評估模型在不同場景下的表現：

```bash
# 基本外觀測試
./scripts/run_generate.sh -t basic

# 側面視圖測試
./scripts/run_generate.sh -t side

# 背面視圖測試
./scripts/run_generate.sh -t back

# 動作測試
./scripts/run_generate.sh -t action

# 表情測試
./scripts/run_generate.sh -t expression
```

## 配置說明

主要配置文件位於 `src/config/config.yaml`，您可以修改以下參數：

- **模型配置**：選擇使用的基礎模型和LoRA權重
- **生成配置**：設置推理步數、生成圖片數量等
- **參數配置**：調整引導尺度、強度等參數
- **圖像尺寸**：設置生成圖像的高度和寬度
- **提示詞配置**：修改正向和負向提示詞

## 項目結構

```
mushroom-lora-studio/
├── assets/                  # 資源文件
│   ├── models/              # 模型文件
│   ├── reference_images/    # 參考圖像
│   └── weights/             # LoRA權重文件
├── outputs/                 # 生成的圖像輸出目錄
├── scripts/                 # 腳本文件
│   └── run_generate.sh      # 主運行腳本
├── src/                     # 源代碼
│   ├── config/              # 配置文件
│   ├── core/                # 核心功能模塊
│   ├── models/              # 模型定義
│   └── utils/               # 工具函數
└── README.md                # 本文件
```

## 自定義動作和表情

您可以通過修改 `src/utils/prompts.py` 文件來添加或修改動作和表情：

```python
# 動作字典
_actions = {
    "standing": "standing still with balanced posture, consistent proportions",
    "sitting": "sitting relaxed with legs crossed, consistent proportions",
    # 添加更多動作...
}

# 表情字典
_expressions = {
    "smiling": "gently smiling with curved lips, symmetrical features",
    "cheerful": "cheerfully smiling with wide eyes, symmetrical features",
    # 添加更多表情...
}
```

## 訓練自己的LoRA模型

本項目使用預訓練的LoRA權重文件。如果您想訓練自己的模型，建議：

1. 準備10-20張高質量的角色參考圖片
2. 為每張圖片創建詳細的文本描述
3. 使用Kohya SS等工具訓練LoRA模型
4. 將生成的權重文件放入 `assets/weights/` 目錄
5. 在配置文件中更新權重文件名

## 常見問題

**Q: 生成的圖像質量不佳或不穩定怎麼辦？**

A: 嘗試增加推理步數（30-50）和引導尺度（7.5-9.0）。也可以使用測試模式評估模型在不同場景下的表現。

**Q: 如何生成不同角度的圖像？**

A: 使用測試模式的side和back選項，或在提示詞中明確指定視角。

**Q: 可以使用其他基礎模型嗎？**

A: 是的，您可以在 `src/config/config.yaml` 中修改模型配置。目前支持 StableDiffusionV15Model 和 AnimefullFinalPrunedFp16Model。

## 許可證

[MIT License](LICENSE)

## 致謝

- 感謝Stable Diffusion和LoRA技術的開發者
- 特別感謝所有貢獻參考圖像和測試的朋友們