"""
圖像生成 API
提供 RESTful API 來生成蘑菇角色圖像
"""
import os
import sys
import time
import uuid
import base64
from io import BytesIO
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

# 導入我們的模塊
from src.core.config_manager import Config
from src.core.model_manager import ModelManager
from src.core.image_generator import ImageGenerator

# 創建 FastAPI 應用
app = FastAPI(
    title="蘑菇角色生成 API",
    description="用於生成蘑菇角色圖像的 API",
    version="1.0.0"
)

# 添加 CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源，生產環境中應該限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局變量
config = Config()
model_manager = ModelManager(config)
loaded_models = {}  # 緩存已加載的模型

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

# 主函數
if __name__ == "__main__":
    import uvicorn
    
    # 從環境變數讀取配置（適配 Render）
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print(f"🍄 蘑菇角色生成 API 啟動中...")
    print(f"📡 監聽地址: {host}:{port}")
    print(f"🔄 重載模式: {reload}")
    
    uvicorn.run("src.api:app", host=host, port=port, reload=reload)