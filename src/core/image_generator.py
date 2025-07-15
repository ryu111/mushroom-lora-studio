"""
圖像生成模塊
用於生成和處理圖像
(最終偵錯修正版)
"""
import sys
import os
import torch
# from rembg import remove  <--- [最終修正] 從頂部移除！
from PIL import Image
import time
import numpy as np
from io import BytesIO
import gc

print("--- [ImageGenerator] 模塊開始被導入... ---", flush=True)

class ImageGenerator:
    # 類別變數：共享 rembg 模型，避免重複載入
    _rembg_session = None
    
    # __init__ 和其他方法保持不變...
    def __init__(self, config, pipe, weight_name):
        print("--- [ImageGenerator] __init__ 開始執行... ---", flush=True)
        self.config = config
        self.pipe = pipe
        self.weight_name = weight_name
        self.prompt = config.get('prompt_template', '')
        self.negative_prompt = config.get('negative_prompt', '')
        self.action_key = "standing"
        self.expression_key = "smiling"
        image_size = config.get('image_size', {})
        self.height = image_size.get('height', 512)
        self.width = image_size.get('width', 512)
        parameters = config.get('parameters', {})
        self.guidance_scale = parameters.get('guidance_scale', 7.5)
        self.strength = parameters.get('strength', 0.75)
        self.noise_level = parameters.get('noise_level', 0.0)
        self.original_image, self.original_image_name = self._load_original_image()
        print("--- [ImageGenerator] __init__ 完成。 ---", flush=True)
    
    # ... 省略其他沒有變動的方法，以保持簡潔 ...
    # 您可以只修改 _process_image 方法，或者直接用這整段覆蓋
    
    def _load_original_image(self):
        original_image_config = self.config.get('original_image', {})
        if isinstance(original_image_config, dict):
            original_image_path = original_image_config.get('path')
            if original_image_path and os.path.exists(original_image_path):
                return Image.open(original_image_path), os.path.splitext(os.path.basename(original_image_path))[0]
        return None, None

    def _prepare_prompt_for_api(self):
        from src.utils.prompts import get_default_action_dict, get_default_expression_dict
        if self.original_image is None:
            action_dict, expression_dict = get_default_action_dict(), get_default_expression_dict()
            action, expression = action_dict.get(self.action_key, "standing"), expression_dict.get(self.expression_key, "smiling")
            base_prompt = "a cartoon mushroom character with a light blue mushroom cap with white dots, eyes and mouth on its body, yellow feet"
            self.prompt = f"{base_prompt}, {action}, {expression}, consistent proportions, symmetrical features"

    def _generate_image_result(self, steps, generator):
        # 除錯：印出 init_image 是否有被使用
        print(f"[DEBUG] init_image 傳入: {self.original_image is not None}")
        print(f"[DEBUG] prompt: {self.prompt}")
        print(f"[DEBUG] negative_prompt: {self.negative_prompt}")
        return self.pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=steps,
            height=self.height,
            width=self.width,
            guidance_scale=self.guidance_scale,
            init_image=self.original_image,
            strength=self.strength,
            num_images_per_prompt=1,
            generator=generator
        )

    def _extract_image_from_result(self, result):
        image = result.images[0] if hasattr(result, "images") else result[0]
        if image is None: raise ValueError("生成的圖像為 None")
        return image

    def _process_image(self, image):
        """🚀 記憶體優化版本的圖像去背處理"""
        print("--- [ImageGenerator] 正在處理圖像去背... ---", flush=True)
        
        # 使用共享的 rembg session，避免重複載入模型
        if ImageGenerator._rembg_session is None:
            from rembg import new_session
            print("--- [ImageGenerator] 初始化 rembg session... ---", flush=True)
            ImageGenerator._rembg_session = new_session('u2net')  # 使用較小的模型
        
        from rembg import remove
        
        # 使用 BytesIO 減少記憶體複製
        with BytesIO() as input_buffer:
            # 將圖像保存到記憶體緩衝區
            image.save(input_buffer, format="PNG")
            input_buffer.seek(0)
            
            # 使用 session 進行去背
            result = remove(input_buffer.getvalue(), session=ImageGenerator._rembg_session)
            
            # 處理返回結果
            if isinstance(result, bytes):
                processed_image = Image.open(BytesIO(result))
            else:
                # 如果返回的是 PIL Image 或其他格式
                processed_image = result
        
        print("--- [ImageGenerator] 圖像去背完成。 ---", flush=True)
        return processed_image

    def _close_images(self, *images):
        """強化的圖像記憶體清理 (支援 CUDA/MPS/CPU)"""
        for img in images:
            if isinstance(img, Image.Image):
                img.close()
        
        # 強制垃圾回收
        gc.collect()
        
        # 清理 GPU 快取 (如果可用)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Mac M1 MPS 記憶體清理
            torch.mps.empty_cache()
        # CPU 環境不需要特殊的快取清理
        
    def generate_single_image_api(self, steps, output_dir):
        """🚀 記憶體優化版本的圖像生成"""
        image = None
        output_img = None
        result = None
        
        try:
            # 生成前清理記憶體和檢查記憶體使用
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
                print(f"🔍 生成前 CUDA 記憶體使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                print(f"🔍 生成前 MPS 記憶體使用: {torch.mps.current_allocated_memory()/1024**3:.2f}GB")
            else:
                print("🔍 生成前記憶體清理完成 (CPU 模式)")
            
            self._prepare_prompt_for_api()
            random_seed = int(time.time())
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(random_seed)
            
            # 生成圖像
            result = self._generate_image_result(steps, generator)
            image = self._extract_image_from_result(result)
            
            # 立即清理生成結果
            del result
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
                print(f"🔍 圖像生成後 CUDA 記憶體使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                print(f"🔍 圖像生成後 MPS 記憶體使用: {torch.mps.current_allocated_memory()/1024**3:.2f}GB")
            else:
                print("🔍 圖像生成後記憶體清理完成 (CPU 模式)")
            
            # 處理圖像 (去背)
            output_img = self._process_image(image)
            
            # 立即清理原始圖像
            if image:
                image.close()
                del image
                image = None
            
            timestamp = int(time.time())
            filename = f"{self.weight_name}_{self.action_key}_{self.expression_key}_{timestamp}_transparent.png"
            full_path = os.path.join(output_dir, filename)
            
            # 修正 output_img 可能為 ndarray 或 bytes 的情況
            if isinstance(output_img, np.ndarray):
                output_img = Image.fromarray(output_img)
            elif isinstance(output_img, bytes):
                output_img = Image.open(BytesIO(output_img))
            
            # 儲存圖像
            output_img.save(full_path)
            print(f"✅ API 已生成圖像：{full_path}", flush=True)
            
            # 清理輸出圖像
            if output_img:
                output_img.close()
                del output_img
                output_img = None
            
            # 最終記憶體清理
            gc.collect()
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
                print(f"🔍 完成後 CUDA 記憶體使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                print(f"🔍 完成後 MPS 記憶體使用: {torch.mps.current_allocated_memory()/1024**3:.2f}GB")
            else:
                print("🔍 完成後記憶體清理完成 (CPU 模式)")
            
            return os.path.join(f"{self.weight_name}", str(steps), filename)
            
        except Exception as e:
            print(f"!!!!!!!! ⚠️ API 生成圖片時出錯: {e} !!!!!!!!", file=sys.stderr, flush=True)
            # 錯誤時也要清理記憶體
            self._close_images(image, output_img)
            if result:
                del result
            gc.collect()
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            raise e

    def generate_images(self):
        """批次生成圖像 (主程式使用)"""
        print("--- [ImageGenerator] 開始批次生成圖像... ---", flush=True)
        
        # 獲取推理配置
        inference_configs = self.config.get('inference_config', [{'steps': 50, 'num_images': 1}])
        
        for config in inference_configs:
            steps = config.get('steps', 50)
            num_images = config.get('num_images', 1)
            
            print(f"--- [ImageGenerator] 生成 {num_images} 張圖像，步數: {steps} ---", flush=True)
            
            for i in range(num_images):
                try:
                    # 準備輸出目錄 (與 API 保持一致)
                    import os
                    output_dir = f"outputs/{self.weight_name}/{steps}"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 生成圖像
                    result_path = self.generate_single_image_api(steps, output_dir)
                    print(f"✅ 已生成第 {i+1}/{num_images} 張圖像")
                    
                except Exception as e:
                    print(f"⚠️ 生成第 {i+1} 張圖像時出錯: {e}")
                    continue
        
        print("--- [ImageGenerator] 批次生成完成。 ---", flush=True)

# 我們甚至可以在 class 定義之後也加上 print，確保整個檔案都執行完畢
print("--- [ImageGenerator] 模塊已成功被定義，所有頂層代碼執行完畢。 ---", flush=True)