from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
import torch
from src.models.base_model import BaseModel

class StableDiffusionV15Model(BaseModel):
    """
    Stable Diffusion v1.5 模型
    """
    def __init__(self):
        super().__init__("stable_diffusion_v1_5")
        self.model_id = "runwayml/stable-diffusion-v1-5"
    
    def load_pipeline(self):
        """載入模型管道"""
        print(f"🔄 加載 {self.model_name} 模型...")
        
        # 載入模型
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # 優化管道
        pipe = self.optimize_pipeline(pipe)
        
        return pipe