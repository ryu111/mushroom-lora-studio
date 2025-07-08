# Render 部署指南

## 部署步驟

### 1. 準備 GitHub 倉庫
```bash
# 確保代碼已推送到 GitHub
git add .
git commit -m "準備 Render 部署"
git push origin main
```

### 2. 在 Render 創建服務
1. 訪問 [render.com](https://render.com)
2. 註冊/登入帳號
3. 點擊 "New +" → "Web Service"
4. 連接你的 GitHub 倉庫
5. 選擇 `mushroom-lora-studio` 倉庫

### 3. 配置設定
- **Name**: `mushroom-lora-api`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python -m src.api`
- **Plan**: `Free`

### 4. 環境變數設定
```
HOST=0.0.0.0
PORT=10000
RELOAD=false
PYTHONPATH=/opt/render/project/src
```

### 5. 部署
點擊 "Create Web Service" 開始部署

## 部署後
- 你的 API 將可在 `https://mushroom-lora-api.onrender.com` 訪問
- 健康檢查: `https://mushroom-lora-api.onrender.com/health`

## n8n 整合
```json
{
  "method": "POST",
  "url": "https://mushroom-lora-api.onrender.com/generate",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
    "weight_name": "mushroom-28.safetensors",
    "steps": 50,
    "action_key": "standing",
    "expression_key": "smiling"
  }
}
```

## 注意事項
- 免費方案會在 15 分鐘無活動後休眠
- 冷啟動需要 30-60 秒
- 每月 750 小時免費運行時間
- 固定域名不會改變