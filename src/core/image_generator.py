"""
圖像生成模塊
用於生成和處理圖像
(重構偵錯版)
"""
import sys
import os
import torch
from rembg import remove
from PIL import Image
import time
import numpy as np
from io import BytesIO
import gc

# [重構] 延遲加載：我們不再在模塊頂部導入 prompts，而是在需要它的函式內部導入。

print("--- [ImageGenerator] 模塊開始被導入... ---", flush=True)

class ImageGenerator:
    """圖像生成類，用於生成和處理圖像"""
    def __init__(self, config, pipe, weight_name):
        print("--- [ImageGenerator] __init__ 開始執行... ---", flush=True)
        self.config = config
        self.pipe = pipe
        self.weight_name = weight_name
        self.prompt = config.get('prompt_template', '')
        self.negative_prompt = config.get('negative_prompt', '')
        self.action_key = "standing"  # 默認動作，可由 API 請求覆蓋
        self.expression_key = "smiling"  # 默認表情，可由 API 請求覆蓋
        
        # 獲取圖像尺寸
        image_size = config.get('image_size', {})
        self.height = image_size.get('height', 512)
        self.width = image_size.get('width', 512)
        
        # 獲取參數
        parameters = config.get('parameters', {})
        self.guidance_scale = parameters.get('guidance_scale', 7.5)
        self.strength = parameters.get('strength', 0.75)
        self.noise_level = parameters.get('noise_level', 0.0)
        
        # [已修復] 移除所有對 sys.argv 的依賴，讓 __init__ 保持通用性
        
        print("--- [ImageGenerator] 正在加載原始圖像 (如果有的話)... ---", flush=True)
        self.original_image, self.original_image_name = self._load_original_image()
        print("--- [ImageGenerator] __init__ 完成。 ---", flush=True)
    
    def _load_original_image(self):
        """讀取原始圖像 (此方法保持不變)"""
        original_image_config = self.config.get('original_image', {})
        if isinstance(original_image_config, dict):
            original_image_path = original_image_config.get('path')
            if original_image_path and os.path.exists(original_image_path):
                return Image.open(original_image_path), os.path.splitext(os.path.basename(original_image_path))[0]
        return None, None

    # ========================================================================
    # == 新增/重構的方法：專為 API 設計，不依賴 sys.argv ==
    # ========================================================================

    def _prepare_prompt_for_api(self):
        """[新增] 專為 API 模式準備提示詞。"""
        print("--- [ImageGenerator] 正在為 API 準備提示詞... ---", flush=True)
        # [延遲加載] 在這裡才導入 prompts 相關函式
        from src.utils.prompts import get_default_action_dict, get_default_expression_dict

        if self.original_image is None:
            action_dict = get_default_action_dict()
            expression_dict = get_default_expression_dict()
            action = action_dict.get(self.action_key, "standing")
            expression = expression_dict.get(self.expression_key, "smiling")
            base_prompt = "a cartoon mushroom character with a light blue mushroom cap with white dots, eyes and mouth on its body, yellow feet"
            final_prompt = f"{base_prompt}, {action}, {expression}, consistent proportions, symmetrical features"
            self.prompt = final_prompt
        else:
            # 如果有原圖，通常 prompt 會由 API 請求直接提供，這裡保持不變
            print("🧷 已提供原圖，將使用請求中提供的 prompt。")
    
    def generate_single_image_api(self, steps, output_dir):
        """[重構] 為 API 生成單張圖片並返回圖像路徑"""
        try:
            # 每次 API 呼叫都重新準備提示詞，以反映請求的參數
            self._prepare_prompt_for_api()
            
            random_seed = int(time.time())
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(random_seed)
            result = self._generate_image_result(steps, generator)
            image = self._extract_image_from_result(result)
            output_img = self._process_image(image)
            
            timestamp = int(time.time())
            filename = f"{self.weight_name}_{self.action_key}_{self.expression_key}_{timestamp}_transparent.png"
            full_path = os.path.join(output_dir, filename)
            
            output_img.save(full_path)
            print(f"✅ API 已生成圖像：{full_path}", flush=True)
            
            self._close_images(image, output_img)
            return os.path.join(f"{self.weight_name}", str(steps), filename)
        except Exception as e:
            print(f"!!!!!!!! ⚠️ API 生成圖片時出錯: {e} !!!!!!!!", file=sys.stderr, flush=True)
            raise e

    # ========================================================================
    # == 保留的舊方法：用於命令行執行，與 API 無關 ==
    # ========================================================================
    
    def generate_images(self):
        """[保留] 生成圖像 (用於命令行模式)"""
        action_key, expression_key = self._select_action_expression()
        print(f"🔄 使用提示詞：{self.prompt}")
        print(f"🔄 使用負面提示詞：{self.negative_prompt}")
        for step_config in self.config.get('inference_config', [{'steps': 50, 'num_images': 1}]):
            steps, num_images = step_config.get('steps', 50), step_config.get('num_images', 1)
            output_dir = os.path.join('outputs', f"{self.weight_name}", str(steps))
            os.makedirs(output_dir, exist_ok=True)
            self._save_parameters(steps, output_dir)
            for i in range(num_images):
                print(f"生成圖片 {i + 1}/{num_images}...")
                self._generate_single_image(steps, output_dir, action_key, expression_key)
    
    def _select_action_expression(self):
        """[保留] 選擇動作和表情並整合到提示詞中 (用於命令行模式)"""
        # [延遲加載] 在這裡才導入 prompts 相關函式
        from src.utils.prompts import get_random_action_dict, get_default_action_dict, get_random_expression_dict, get_default_expression_dict
        
        mode = sys.argv[1] if len(sys.argv) > 1 else "-d"
        if mode == "-t":
            test_type = sys.argv[2] if len(sys.argv) > 2 else "basic"
            self.prompt = self._get_test_prompt(test_type)
            return test_type, "test"
        if self.original_image is None:
            action_dict = get_random_action_dict() if mode == "-r" else get_default_action_dict()
            expression_dict = get_random_expression_dict() if mode == "-r" else get_default_expression_dict()
            action_key, action = next(iter(action_dict.items()))
            expression_key, expression = next(iter(expression_dict.items()))
            base_prompt = "a cartoon mushroom character with a light blue mushroom cap with white dots, eyes and mouth on its body, yellow feet"
            self.prompt = f"{base_prompt}, {action}, {expression}, consistent proportions, symmetrical features"
            return action_key, expression_key
        return "none", "none"

    # ========================================================================
    # == 通用輔助方法 (Helper Methods) ==
    # ========================================================================

    def _generate_single_image(self, steps, output_dir, action_key, expression_key):
        """[保留] 生成單張圖片的底層邏輯"""
        try:
            random_seed = int(time.time())
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(random_seed)
            result = self._generate_image_result(steps, generator)
            image = self._extract_image_from_result(result)
            output_img = self._process_image(image)
            self._save_image(output_img, output_dir, action_key, expression_key)
            self._close_images(image, output_img)
        except Exception as e:
            print(f"⚠️ 生成圖片時出錯: {e}")

    def _generate_image_result(self, steps, generator):
        """[通用] 生成圖像結果"""
        print(f"--- [ImageGenerator] 最終提示詞: {self.prompt} ---", flush=True)
        return self.pipe(prompt=self.prompt, negative_prompt=self.negative_prompt, num_inference_steps=steps, height=self.height, width=self.width, guidance_scale=self.guidance_scale, noise_level=self.noise_level, init_image=self.original_image, strength=self.strength, num_images_per_prompt=1, generator=generator)

    def _extract_image_from_result(self, result):
        """[通用] 從結果中提取圖像"""
        image = result.images[0] if hasattr(result, "images") else result[0]
        if image is None: raise ValueError("生成的圖像為 None")
        return image

    def _process_image(self, image):
        """[通用] 處理圖像，去背"""
        with BytesIO() as image_buffer:
            image.save(image_buffer, format="PNG")
            image_buffer.seek(0)
            return remove(Image.open(image_buffer))

    def _save_image(self, output_img, output_dir, action_key, expression_key):
        """[通用] 保存圖像"""
        timestamp = int(time.time())
        filename = f"{self.weight_name}_{self.original_image_name or action_key}_{expression_key}_{timestamp}_transparent.png"
        full_path = os.path.join(output_dir, filename)
        output_img.save(full_path)
        print(f"✅ 已保存圖像：{full_path}")
        
    def _close_images(self, *images):
        """[通用] 關閉圖像釋放資源"""
        for img in images:
            if isinstance(img, Image.Image):
                img.close()
        gc.collect()

    # 其他保留的輔助方法...
    def _get_test_prompt(self, test_type):
        """[保留] 獲取測試提示詞"""
        test_prompts = {"basic": "...", "side": "...", "back": "...", "action": "...", "expression": "..."}
        return test_prompts.get(test_type, test_prompts["basic"])

    def _save_parameters(self, steps, output_dir):
        """[保留] 保存參數信息到文件"""
        params_filename = os.path.join(output_dir, "parameters.txt")
        with open(params_filename, 'w') as params_file:
            params_file.write(f"...") # 內容省略

print("--- [ImageGenerator] 模塊已成功被定義。 ---", flush=True)