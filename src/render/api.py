# ==================== 偵錯版本，請完整複製 ====================
print("--- 步驟 0：Python 腳本開始執行 ---", flush=True)

import os
import sys
import time
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 在導入 torch 前後加上日誌，因為它非常消耗資源
print("--- 步驟 1：準備導入 torch 函式庫... ---", flush=True)
try:
    import torch
    print("--- 步驟 2：torch 函式庫導入成功 ---", flush=True)
except Exception as e:
    print(f"!!!!!!!! 致命錯誤：導入 torch 時發生問題 !!!!!!!!", flush=True)
    print(f"錯誤類型: {type(e).__name__}, 錯誤詳情: {e}", flush=True)
    sys.exit(1) # 導入失敗，直接退出

# 包圍所有自訂模塊的導入和初始化
try:
    # ==================== 請用這段程式碼替換原有的步驟 3 和 4 ====================

    print("--- 步驟 3：開始逐一導入自訂模塊... ---", flush=True)

    # 導入第一個模塊
    print("--- 步驟 3.1：準備導入 Config... ---", flush=True)
    from src.core.config_manager import Config
    print("--- 步驟 3.2：Config 導入成功。 ---", flush=True)

    # 導入第二個模塊
    print("--- 步驟 3.3：準備導入 ModelManager... ---", flush=True)
    from src.core.model_manager import ModelManager
    print("--- 步驟 3.4：ModelManager 導入成功。 ---", flush=True)

    # 導入第三個模塊
    print("--- 步驟 3.5：準備導入 ImageGenerator... ---", flush=True)
    from src.core.image_generator import ImageGenerator
    print("--- 步驟 4：所有自訂模塊導入成功 ---", flush=True)

    # ========================================================================

    # 全局變量初始化
    print("--- 步驟 5：準備初始化 Config() 物件... ---", flush=True)
    config = Config()
    print("--- 步驟 6：Config() 物件初始化成功 ---", flush=True)

    print("--- 步驟 7：準備初始化 ModelManager(config) 物件... ---", flush=True)
    model_manager = ModelManager(config)
    print("--- 步驟 8：ModelManager(config) 物件初始化成功 ---", flush=True)
    
    loaded_models = {}  # 緩存已加載的模型

    # 創建 FastAPI 應用
    print("--- 步驟 9：準備建立 FastAPI App 實例... ---", flush=True)
    app = FastAPI(
        title="蘑菇角色生成 API",
        description="用於生成蘑菇角色圖像的 API",
        version="1.0.0"
    )
    print("--- 步驟 10：FastAPI App 實例建立成功 ---", flush=True)

except Exception as e:
    print(f"!!!!!!!! 致命錯誤：在初始化自訂模塊或 FastAPI 時發生問題 !!!!!!!!", flush=True)
    import traceback
    # 將詳細的錯誤追蹤印到標準錯誤輸出，這樣在 Render 日誌中更容易看到
    traceback.print_exc(file=sys.stderr)
    sys.exit(1) # 初始化失敗，直接退出

# --- 後續的程式碼保持不變，但現在我們知道上面的初始化都成功了 ---

print("--- 步驟 11：準備設定 CORS 中間件... ---", flush=True)
# 添加 CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源，生產環境中應該限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("--- 步驟 12：CORS 中間件設定完成 ---", flush=True)

# 定義請求模型
class GenerateImageRequest(BaseModel):
    weight_name: str
    steps: int = 50
    action_key: Optional[str] = "standing"
    expression_key: Optional[str] = "smiling"
    original_image_path: Optional[str] = None
    prompt_template: Optional[str] = None
    negative_prompt: Optional[str] = None
    guidance_scale: float = 7.0
    strength: float = 0.25
    noise_level: float = 0.0
    height: int = 512
    width: int = 512

# 定義響應模型
class GenerateImageResponse(BaseModel):
    image_path: str
    generation_time: float
    parameters: Dict[str, Any]

print("--- 步驟 13：API 端點 (Endpoint) 準備定義... ---", flush=True)

# 健康檢查端點
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}

# 獲取可用模型端點
@app.get("/models")
async def get_models():
    models_config = config.get('models', {}) or {}
    return {"models": list(models_config.keys())}

# 獲取可用動作端點
@app.get("/actions")
async def get_actions():
    from src.utils.prompts import _actions
    return {"actions": _actions}

# 獲取可用表情端點
@app.get("/expressions")
async def get_expressions():
    from src.utils.prompts import _expressions
    return {"expressions": _expressions}

# 生成圖像端點
@app.post("/generate", response_model=GenerateImageResponse)
async def generate_image(request: GenerateImageRequest):
    try:
        # 獲取或加載模型
        if request.weight_name not in loaded_models:
            pipe = model_manager.load_model(request.weight_name)
            loaded_models[request.weight_name] = pipe
        else:
            pipe = loaded_models[request.weight_name]
        
        # 創建自定義配置
        custom_config = {
            'prompt_template': request.prompt_template or config.get('prompt_template', ''),
            'negative_prompt': request.negative_prompt or config.get('negative_prompt', ''),
            'parameters': {
                'guidance_scale': request.guidance_scale,
                'strength': request.strength,
                'noise_level': request.noise_level
            },
            'image_size': {
                'height': request.height,
                'width': request.width
            }
        }
        
        # 如果提供了原始圖像路徑
        if request.original_image_path:
            if os.path.exists(request.original_image_path):
                custom_config['original_image'] = {'path': request.original_image_path}
            else:
                raise HTTPException(status_code=404, detail=f"原始圖像不存在: {request.original_image_path}")
        
        # 創建臨時配置對象
        temp_config = Config()
        temp_config.config = {**config.config, **custom_config}
        
        # 初始化圖像生成器
        image_generator = ImageGenerator(temp_config, pipe, request.weight_name)
        
        # 設置動作和表情
        image_generator.action_key = request.action_key or "standing"
        image_generator.expression_key = request.expression_key or "smiling"
        
        # 創建輸出目錄
        output_dir = os.path.join('outputs', f"{request.weight_name}", str(request.steps))
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成圖像
        start_time = time.time()
        image_path = image_generator.generate_single_image_api(request.steps, output_dir)
        generation_time = time.time() - start_time
        
        # 返回結果
        return {
            "image_path": image_path,
            "generation_time": generation_time,
            "parameters": {
                "weight_name": request.weight_name,
                "steps": request.steps,
                "action_key": request.action_key,
                "expression_key": request.expression_key,
                "guidance_scale": request.guidance_scale,
                "strength": request.strength,
                "noise_level": request.noise_level,
                "height": request.height,
                "width": request.width
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 獲取生成的圖像端點
@app.get("/image/{image_path:path}")
async def get_image(image_path: str):
    full_path = os.path.join('outputs', image_path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="圖像不存在")
    return FileResponse(full_path)

print("--- 步驟 14：API 端點定義完成 ---", flush=True)
print("--- 應用程式初始化完畢，準備由 uvicorn 啟動。如果在這之後沒有看到 uvicorn 的日誌，表示問題出在 Render 的 Start Command。---", flush=True)

# 主函數 (這段在 Render 環境下通常不會被執行，因為 Render 直接用 uvicorn 命令啟動)
if __name__ == "__main__":
    import uvicorn
    
    host = "0.0.0.0"
    port = int(os.getenv("PORT", 10000))
    
    print(f"🍄 蘑菇角色生成 API 啟動中 (本地模式)...")
    print(f"📡 監聽地址: {host}:{port}")
    
    uvicorn.run(
        "src.render.api:app",
        host=host,
        port=port,
        reload=True, # 本地開發開啟 reload
        access_log=True,
        log_level="info"
    )