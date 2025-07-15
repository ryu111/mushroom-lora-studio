from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
import torch
from src.models.base_model import BaseModel

class StableDiffusionXLModel(BaseModel):
    """
    Stable Diffusion XL 模型
    """
    def __init__(self):
        super().__init__("stable_diffusion_xl")
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    
    def load_pipeline(self):
        """載入模型管道 (Mac M1 MPS 記憶體優化版)"""
        print(f"🔄 加載 {self.model_name} 模型...")
        
        # Mac M1 MPS 記憶體優化設定
        if self.device == "mps":
            print("🍎 Mac M1 MPS 記憶體優化模式")
            # 使用 float32 減少記憶體使用
            torch_dtype = torch.float32
            variant = None
        else:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            variant = "fp16" if self.device == "cuda" else None
        
        # 載入模型
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            variant=variant,
            # Mac M1 專用：低記憶體模式
            low_cpu_mem_usage=True if self.device == "mps" else False
        )
        
        # 優化管道
        pipe = self.optimize_pipeline(pipe)
        
        return pipe