import torch
import os
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    基礎模型類，所有模型類都應該繼承自這個類
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = self._get_device()
        
    def _get_device(self):
        """獲取可用的設備"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    @abstractmethod
    def load_pipeline(self):
        """加載模型管道，子類必須實現這個方法"""
        pass
    
    def optimize_pipeline(self, pipe):
        """優化模型管道，啟用記憶體優化等"""
        # 移除安全檢查器
        pipe.safety_checker = None
        
        # 將模型移至指定設備
        pipe = pipe.to(self.device)
        
        # 啟用記憶體優化
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
            
        return pipe
    
    def load_lora_weights(self, pipe, weight_name):
        """加載 LoRA 權重"""
        if weight_name:
            try:
                pipe.load_lora_weights("assets/weights", weight_name=weight_name)
                print(f"✅ 已加載 LoRA 權重：{weight_name}")
            except Exception as e:
                print(f"⚠️ 加載 LoRA 權重失敗: {e}")
        return pipe