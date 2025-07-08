from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
import torch
import os
from src.models.base_model import BaseModel

class AnimefullFinalPrunedFp16Model(BaseModel):
    """
    AnimefullFinalPrunedFp16 模型
    基於 Stable Diffusion v1.5，並加載 animefull-final-pruned-fp16.safetensors 權重
    """
    def __init__(self):
        super().__init__("animefull_final_pruned_fp16")
        self.base_model_id = "runwayml/stable-diffusion-v1-5"
        self.weights_file = self._get_weights_file_path()
    
    def _get_weights_file_path(self):
        """獲取權重文件路徑"""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base_dir, "assets", "models", "animefull-final-pruned-fp16.safetensors")
    
    def load_pipeline(self):
        """載入模型管道"""
        print(f"🔄 加載基礎模型 {self.base_model_id}...")
        
        # 載入基礎模型
        pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # 優化管道
        pipe = self.optimize_pipeline(pipe)
        
        # 嘗試加載自定義權重
        pipe = self._merge_custom_weights(pipe)
        
        return pipe
    
    def _merge_custom_weights(self, pipe):
        """合併自定義權重到模型中"""
        if not os.path.exists(self.weights_file):
            print(f"⚠️ 找不到權重文件: {self.weights_file}")
            print(f"繼續使用基礎模型 {self.base_model_id}")
            return pipe
        
        try:
            print(f"🔄 合併 animefull-final-pruned-fp16 權重到基礎模型...")
            
            # 使用 safetensors 庫加載權重
            try:
                from safetensors import safe_open
            except ImportError:
                print("⚠️ 未安裝 safetensors 庫，無法加載自定義權重")
                return pipe
            
            # 加載所有權重
            all_weights = self._load_all_weights(self.weights_file)
            if not all_weights:
                return pipe
            
            # 將權重應用到模型的不同組件
            pipe = self._apply_weights_to_components(pipe, all_weights)
            
            print(f"✅ 成功合併 animefull-final-pruned-fp16 權重")
            
        except Exception as e:
            print(f"⚠️ 合併權重失敗: {e}")
            print(f"繼續使用基礎模型 {self.base_model_id}")
        
        return pipe
    
    def _load_all_weights(self, weights_file):
        """加載所有權重"""
        all_weights = {}
        
        try:
            # 確保 safetensors 庫已導入
            from safetensors import safe_open
            
            with safe_open(weights_file, framework="pt") as f:
                keys = f.keys()
                
                # 加載所有權重
                for k in keys:
                    try:
                        all_weights[k] = f.get_tensor(k)
                    except Exception as e:
                        print(f"⚠️ 無法加載權重 {k}: {e}")
            
            return all_weights
        except Exception as e:
            print(f"⚠️ 加載權重文件失敗: {e}")
            return {}
    
    def _apply_weights_to_components(self, pipe, all_weights):
        """將權重應用到模型的不同組件"""
        if not all_weights:
            return pipe
        
        keys = list(all_weights.keys())
        
        # 應用 UNet 權重
        self._apply_unet_weights(pipe, all_weights, keys)
        
        # 應用 VAE 權重
        self._apply_vae_weights(pipe, all_weights, keys)
        
        # 應用 Text Encoder 權重
        self._apply_text_encoder_weights(pipe, all_weights, keys)
        
        # 應用 Scheduler 權重
        self._apply_scheduler_weights(pipe, all_weights, keys)
        
        # 應用其他權重
        self._apply_other_weights(pipe, all_weights, keys)
        
        return pipe
    
    def _apply_unet_weights(self, pipe, all_weights, keys):
        """應用 UNet 權重"""
        unet_keys = [k for k in keys if k.startswith("unet") or k.startswith("model.diffusion_model")]
        if unet_keys:
            unet_state_dict = {k: all_weights[k] for k in unet_keys}
            pipe.unet.load_state_dict(unet_state_dict, strict=False)
    
    def _apply_vae_weights(self, pipe, all_weights, keys):
        """應用 VAE 權重"""
        vae_keys = [k for k in keys if k.startswith("vae") or k.startswith("first_stage_model")]
        if vae_keys:
            vae_state_dict = {k: all_weights[k] for k in vae_keys}
            pipe.vae.load_state_dict(vae_state_dict, strict=False)
    
    def _apply_text_encoder_weights(self, pipe, all_weights, keys):
        """應用 Text Encoder 權重"""
        text_encoder_keys = [k for k in keys if k.startswith("text_encoder") or k.startswith("cond_stage_model")]
        if text_encoder_keys:
            text_encoder_state_dict = {k: all_weights[k] for k in text_encoder_keys}
            pipe.text_encoder.load_state_dict(text_encoder_state_dict, strict=False)
    
    def _apply_scheduler_weights(self, pipe, all_weights, keys):
        """應用 Scheduler 權重"""
        scheduler_keys = [
            'alphas_cumprod', 'alphas_cumprod_prev', 'betas',
            'log_one_minus_alphas_cumprod', 'posterior_log_variance_clipped',
            'posterior_mean_coef1', 'posterior_mean_coef2', 'posterior_variance',
            'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
            'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod'
        ]
        
        found_scheduler_keys = [k for k in keys if k in scheduler_keys]
        if found_scheduler_keys and hasattr(pipe, 'scheduler'):
            try:
                for k in found_scheduler_keys:
                    if hasattr(pipe.scheduler, k):
                        setattr(pipe.scheduler, k, all_weights[k].to(self.device))
                
                # 重新計算時間步長
                if hasattr(pipe.scheduler, 'set_timesteps') and hasattr(pipe.scheduler, 'num_inference_steps'):
                    if pipe.scheduler.num_inference_steps is not None:
                        pipe.scheduler.set_timesteps(pipe.scheduler.num_inference_steps)
            except Exception as e:
                print(f"⚠️ 更新 scheduler 參數失敗: {e}")
    
    def _apply_other_weights(self, pipe, all_weights, keys):
        """應用其他權重"""
        # 已處理的鍵
        processed_keys = []
        processed_keys.extend([k for k in keys if k.startswith("unet") or k.startswith("model.diffusion_model")])
        processed_keys.extend([k for k in keys if k.startswith("vae") or k.startswith("first_stage_model")])
        processed_keys.extend([k for k in keys if k.startswith("text_encoder") or k.startswith("cond_stage_model")])
        processed_keys.extend([k for k in keys if k in [
            'alphas_cumprod', 'alphas_cumprod_prev', 'betas',
            'log_one_minus_alphas_cumprod', 'posterior_log_variance_clipped',
            'posterior_mean_coef1', 'posterior_mean_coef2', 'posterior_variance',
            'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
            'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod'
        ]])
        
        # 未處理的鍵
        other_keys = [k for k in keys if k not in processed_keys]
        
        # 嘗試將這些鍵應用到模型的其他組件
        if other_keys:
            for component_name in dir(pipe):
                # 跳過私有屬性和方法
                if component_name.startswith('_'):
                    continue
                    
                try:
                    component = getattr(pipe, component_name)
                    
                    # 檢查是否為可載入權重的組件
                    if hasattr(component, 'load_state_dict'):
                        try:
                            # 創建一個只包含與該組件相關的鍵的字典
                            component_dict = {}
                            for k in other_keys:
                                if k.startswith(component_name):
                                    component_dict[k] = all_weights[k]
                            
                            if component_dict:
                                component.load_state_dict(component_dict, strict=False)
                        except Exception:
                            pass
                except Exception:
                    pass