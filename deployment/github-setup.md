# GitHub 倉庫創建指南

## 方法一：使用 GitHub 網站創建（推薦）

### 1. 創建新倉庫
1. 訪問 [github.com](https://github.com)
2. 登入你的 GitHub 帳號
3. 點擊右上角的 "+" → "New repository"
4. 填寫倉庫資訊：
   - **Repository name**: `mushroom-lora-studio`
   - **Description**: `蘑菇角色生成 API - 使用 LoRA 技術的 AI 圖像生成服務`
   - **Visibility**: Public 或 Private（根據需求選擇）
   - **不要勾選** "Add a README file"（我們已經有了）
   - **不要勾選** "Add .gitignore"（我們已經有了）
5. 點擊 "Create repository"

### 2. 連接本地倉庫
創建完成後，GitHub 會顯示連接指令，複製 HTTPS URL，然後執行：

```bash
# 添加遠端倉庫（替換為你的實際 URL）
git remote add origin https://github.com/YOUR_USERNAME/mushroom-lora-studio.git

# 推送到 GitHub
git branch -M main
git push -u origin main
```

## 方法二：安裝 GitHub CLI（可選）

如果你想使用命令列創建：

```bash
# 安裝 GitHub CLI
brew install gh

# 登入 GitHub
gh auth login

# 創建倉庫並推送
gh repo create mushroom-lora-studio --public --description "蘑菇角色生成 API - 使用 LoRA 技術的 AI 圖像生成服務"
git remote add origin https://github.com/YOUR_USERNAME/mushroom-lora-studio.git
git branch -M main
git push -u origin main
```

## 推送完成後

倉庫創建並推送成功後，你就可以：

1. **部署到 Render**：
   - 使用 GitHub 倉庫 URL 在 Render 創建服務
   - 參考 `deployment/render-deploy.md` 的詳細步驟

2. **與 n8n 整合**：
   - 使用 Render 提供的固定域名
   - API 端點：`https://your-app-name.onrender.com`

## 注意事項

- 大型模型文件已在 `.gitignore` 中排除
- 生成的圖像輸出也已排除
- 只推送源代碼和配置文件
- 模型文件需要在部署時另外處理