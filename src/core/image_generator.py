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
        return self.pipe(prompt=self.prompt, negative_prompt=self.negative_prompt, num_inference_steps=steps, height=self.height, width=self.width, guidance_scale=self.guidance_scale, noise_level=self.noise_level, init_image=self.original_image, strength=self.strength, num_images_per_prompt=1, generator=generator)

    def _extract_image_from_result(self, result):
        image = result.images[0] if hasattr(result, "images") else result[0]
        if image is None: raise ValueError("生成的圖像為 None")
        return image

    def _process_image(self, image):
        """[最終修正] 處理圖像，去背。在這裡才導入 rembg。"""
        # [延遲加載] 在這裡才導入 rembg，避免啟動時加載模型
        from rembg import remove
        print("--- [ImageGenerator] 正在處理圖像去背... ---", flush=True)

        with BytesIO() as image_buffer:
            image.save(image_buffer, format="PNG")
            image_buffer.seek(0)
            processed_image = remove(Image.open(image_buffer))
        
        print("--- [ImageGenerator] 圖像去背完成。 ---", flush=True)
        return processed_image

    def _close_images(self, *images):
        for img in images:
            if isinstance(img, Image.Image):
                img.close()
        gc.collect()
        
    def generate_single_image_api(self, steps, output_dir):
        try:
            self._prepare_prompt_for_api()
            random_seed = int(time.time())
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(random_seed)
            result = self._generate_image_result(steps, generator)
            image = self._extract_image_from_result(result)
            output_img = self._process_image(image)
            
            timestamp = int(time.time())
            filename = f"{self.weight_name}_{self.action_key}_{self.expression_key}_{timestamp}_transparent.png"
            full_path = os.path.join(output_dir, filename)
            
            # 修正 output_img 可能為 ndarray 或 bytes 的情況
            if isinstance(output_img, np.ndarray):
                output_img = Image.fromarray(output_img)
            elif isinstance(output_img, bytes):
                output_img = Image.open(BytesIO(output_img))
            output_img.save(full_path)
            print(f"✅ API 已生成圖像：{full_path}", flush=True)
            
            self._close_images(image, output_img)
            return os.path.join(f"{self.weight_name}", str(steps), filename)
        except Exception as e:
            print(f"!!!!!!!! ⚠️ API 生成圖片時出錯: {e} !!!!!!!!", file=sys.stderr, flush=True)
            raise e

# 我們甚至可以在 class 定義之後也加上 print，確保整個檔案都執行完畢
print("--- [ImageGenerator] 模塊已成功被定義，所有頂層代碼執行完畢。 ---", flush=True)