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
        """獲取可用的設備 (針對 Render 部署優化)"""
        # 檢查環境變數，強制使用 CPU (適用於 Render 等雲端平台)
        import os
        if os.getenv('FORCE_CPU', 'false').lower() == 'true':
            print("🔧 環境變數 FORCE_CPU=true，強制使用 CPU")
            return "cpu"
        
        if torch.cuda.is_available():
            print("🚀 檢測到 CUDA，使用 GPU 加速")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("🍎 檢測到 MPS (Mac M1)，使用 MPS 加速")
            return "mps"
        else:
            print("💻 使用 CPU 運算")
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
        
        # 🚀 強化記憶體優化
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        
        # 嘗試啟用 CPU 卸載 (需要 accelerate 套件)
        try:
            if hasattr(pipe, "enable_model_cpu_offload") and self.device in ["mps", "cuda"]:
                pipe.enable_model_cpu_offload()
                print(f"✅ 已啟用 {self.device.upper()} CPU 卸載")
            elif hasattr(pipe, "enable_sequential_cpu_offload") and self.device in ["cuda", "mps"]:
                pipe.enable_sequential_cpu_offload()
                print(f"✅ 已啟用 {self.device.upper()} 順序 CPU 卸載")
        except Exception as e:
            print(f"⚠️ CPU 卸載不可用 (需要安裝 accelerate): {e}")
            print("💡 建議執行: pip install accelerate")
        
        # 記憶體高效注意力：Mac M1 不支援 xformers，跳過
        if self.device != "mps" and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("✅ 已啟用 xformers 記憶體高效注意力")
            except Exception as e:
                print(f"⚠️ xformers 不可用: {e}")
        elif self.device == "mps":
            print("ℹ️ Mac M1 不支援 xformers，使用原生 MPS 優化")
        
        # 啟用 VAE 切片 (減少 VAE 記憶體使用)
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        
        # 啟用 VAE 平鋪 (處理大圖像時節省記憶體)
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
            
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