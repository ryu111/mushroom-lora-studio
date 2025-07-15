"""
圖像生成腳本
用於生成基於 Stable Diffusion 的圖像
"""
import os
import torch

# Mac M1 MPS 記憶體優化：在導入其他模組前設定
if torch.backends.mps.is_available():
    print("🍎 檢測到 Apple Silicon MPS，設定記憶體優化...")
    # 清除可能衝突的環境變數
    if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in os.environ:
        del os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]
    # 設定為 0.0 實現按需分配記憶體（解決 SDXL 記憶體衝突）
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print(f"✅ MPS 記憶體策略設定為按需分配 (WATERMARK_RATIO=0.0)")

from src.core.config_manager import Config
from src.core.model_manager import ModelManager
from src.core.image_generator import ImageGenerator

# 主函數
def main():
    """主函數"""
    # 加載配置
    config = Config()
    
    # 初始化模型管理器
    model_manager = ModelManager(config)
    
    # 生成圖像
    weight_names = config.get('weight_name', [])
    if weight_names is None:
        weight_names = []
    
    for weight_name in weight_names:
        # 加載模型
        pipe = model_manager.load_model(weight_name)
        
        # 初始化圖像生成器
        image_generator = ImageGenerator(config, pipe, weight_name)
        
        # 生成圖像
        image_generator.generate_images()

if __name__ == "__main__":
    main()