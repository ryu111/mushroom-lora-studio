"""
圖像生成腳本
用於生成基於 Stable Diffusion 的圖像
"""
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