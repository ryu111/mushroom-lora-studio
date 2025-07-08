"""
模型管理模塊
用於加載和管理模型
"""
import importlib
from src.models.stable_diffusion_v1_5 import StableDiffusionV15Model

class ModelManager:
    """模型管理類，用於加載和管理模型"""
    def __init__(self, config):
        self.config = config
        self.model_name = config.get('model', 'StableDiffusionV15Model')
    
    def load_model(self, weight_name):
        """加載模型"""
        # 動態導入模型類
        model_instance = self._create_model_instance()
        
        # 加載模型管道
        pipe = model_instance.load_pipeline()
        
        # 加載 LoRA 權重
        if weight_name:
            pipe = model_instance.load_lora_weights(pipe, weight_name)
        
        return pipe
    
    def _create_model_instance(self):
        """創建模型實例"""
        # 根據模型名稱動態導入模型類
        try:
            # 將駝峰命名轉換為下劃線命名，並移除 "Model" 後綴
            model_name = self.model_name
            if model_name.endswith("Model"):
                model_name = model_name[:-5]  # 移除 "Model" 後綴
            
            module_file = ''.join(['_' + c.lower() if c.isupper() else c.lower() for c in model_name]).lstrip('_')
            
            # 嘗試從 src.models 包中導入模型類
            module_name = f"src.models.{module_file}"
            module = importlib.import_module(module_name)
            
            # 獲取模型類名（假設類名是模型名稱）
            class_name = self.model_name
            model_class = getattr(module, class_name)
            
            # 實例化模型
            return model_class()
        except (ImportError, AttributeError) as e:
            # 如果找不到模型類，嘗試從 models 包中導入
            try:
                module_name = f"assets.models.{module_file}"
                module = importlib.import_module(module_name)
                
                # 獲取模型類名（假設類名是模型名稱）
                class_name = self.model_name
                model_class = getattr(module, class_name)
                
                # 實例化模型
                return model_class()
            except (ImportError, AttributeError) as e:
                print(f"⚠️ 無法載入模型 {self.model_name}: {e}")
                # 使用預設模型
                print(f"⚠️ 使用預設模型 StableDiffusionV15Model")
                return StableDiffusionV15Model()