"""
主啟動文件 - 用於 Render 部署
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 導入 FastAPI 應用
from src.render.api import app

if __name__ == "__main__":
    import uvicorn
    
    # 從環境變數讀取配置
    host = "0.0.0.0"
    port = int(os.getenv("PORT", 10000))  # Render 預設端口為 10000
    
    print(f"🍄 蘑菇角色生成 API 啟動中...")
    print(f"📡 監聽地址: {host}:{port}")
    
    uvicorn.run(app, host=host, port=port)