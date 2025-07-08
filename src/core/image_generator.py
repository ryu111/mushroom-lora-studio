"""
圖像生成模塊
用於生成和處理圖像
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
from src.utils.prompts import get_random_action_dict, get_default_action_dict, get_random_expression_dict, get_default_expression_dict

class ImageGenerator:
    """圖像生成類，用於生成和處理圖像"""
    def __init__(self, config, pipe, weight_name):
        self.config = config
        self.pipe = pipe
        self.weight_name = weight_name
        self.prompt = config.get('prompt_template', '')
        self.negative_prompt = config.get('negative_prompt', '')
        self.action_key = "standing"  # 默認動作
        self.expression_key = "smiling"  # 默認表情
        
        # 獲取圖像尺寸
        image_size = config.get('image_size', {})
        self.height = image_size.get('height', 512) if isinstance(image_size, dict) else config.get('height', 512)
        self.width = image_size.get('width', 512) if isinstance(image_size, dict) else config.get('width', 512)
        
        # 獲取參數
        parameters = config.get('parameters', {})
        self.guidance_scale = parameters.get('guidance_scale', 7.5) if isinstance(parameters, dict) else config.get('guidance_scale', 7.5)
        self.strength = parameters.get('strength', 0.75) if isinstance(parameters, dict) else config.get('strength', 0.75)
        self.noise_level = parameters.get('noise_level', 0.0) if isinstance(parameters, dict) else config.get('noise_level', 0.0)
        
        # 獲取命令行參數
        self.mode = sys.argv[1] if len(sys.argv) > 1 else "-d"
        
        # 如果是測試模式，設置測試類型
        if self.mode == "-t":
            self.test_type = sys.argv[2] if len(sys.argv) > 2 else "basic"
            print(f"🧪 測試模式：{self.test_type}")
            
        self.original_image, self.original_image_name = self._load_original_image()
    
    def _load_original_image(self):
        """讀取原始圖像"""
        # 嘗試從新的配置結構中獲取路徑
        original_image = self.config.get('original_image', {})
        if isinstance(original_image, dict):
            original_image_path = original_image.get('path')
        else:
            # 嘗試從舊的配置結構中獲取路徑
            original_image_path = self.config.get('original_image_path')
        if original_image_path and os.path.exists(original_image_path):
            original_image = Image.open(original_image_path)
            original_image_name = os.path.splitext(os.path.basename(original_image_path))[0]
            return original_image, original_image_name
        return None, None
    
    def generate_images(self):
        """生成圖像"""
        # 獲取動作和表情
        action_key, expression_key = self._select_action_expression()
        
        # 輸出提示詞和負面提示詞（只輸出一次）
        print(f"🔄 使用提示詞：{self.prompt}")
        print(f"🔄 使用負面提示詞：{self.negative_prompt}")
        print(f"🔄 使用引導尺度：{self.guidance_scale}")
        
        # 生成圖像
        for step_config in self.config.get('inference_config', [{'steps': 50, 'num_images': 1}]):
            steps = step_config.get('steps', 50)
            num_images = step_config.get('num_images', 1)
            
            # 創建輸出目錄
            output_dir = os.path.join('outputs', f"{self.weight_name}", str(steps))
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存參數信息
            self._save_parameters(steps, output_dir)
            
            # 生成指定數量的圖片
            for image_index in range(num_images):
                print(f"生成圖片 {image_index + 1}/{num_images}，使用模型 {self.weight_name}，進行 {steps} 步推理...")
                self._generate_single_image(steps, output_dir, action_key, expression_key)
    
    def _select_action_expression(self):
        """選擇動作和表情並整合到提示詞中"""
        # 檢查是否為測試模式
        if self.mode == "-t":
            test_type = sys.argv[2] if len(sys.argv) > 2 else "basic"
            self.prompt = self._get_test_prompt(test_type)
            print(f"🧪 使用測試提示詞：{test_type}")
            return test_type, "test"
            
        if self.original_image is None:
            if self.mode == "-r":
                action_dict = get_random_action_dict()
                action_key, action = next(iter(action_dict.items()))
                expression_dict = get_random_expression_dict()
                expression_key, expression = next(iter(expression_dict.items()))
                print(f"🔄 隨機動作：{action_key}，表情：{expression_key}")
            else:
                action_dict = get_default_action_dict()
                action_key, action = next(iter(action_dict.items()))
                expression_dict = get_default_expression_dict()
                expression_key, expression = next(iter(expression_dict.items()))
                print(f"🔄 預設動作：{action_key}，表情：{expression_key}")
            
            # 將動作和表情整合到提示詞中
            base_prompt = "a cartoon mushroom character with a light blue mushroom cap with white dots, eyes and mouth on its body, yellow feet"
            self.prompt = f"{base_prompt}, {action}, {expression}, consistent proportions, symmetrical features"
            print(f"📝 生成提示詞：{self.prompt}")
            
            return action_key, expression_key
        else:
            print("🧷 已提供原圖，省略動作與表情附加至 prompt。")
            return "none", "none"
            
    def _get_test_prompt(self, test_type):
        """獲取測試提示詞"""
        test_prompts = {
            "basic": "a cartoon mushroom character with a light blue mushroom cap with white dots, eyes and mouth on its body, yellow feet, front view, consistent proportions, symmetrical features",
            "side": "a cartoon mushroom character with a light blue mushroom cap with white dots, eyes and mouth on its body, yellow feet, side view, facing left, consistent proportions, symmetrical features",
            "back": "a cartoon mushroom character with a light blue mushroom cap with white dots, yellow feet, back view, consistent proportions, symmetrical features",
            "action": "a cartoon mushroom character with a light blue mushroom cap with white dots, eyes and mouth on its body, yellow feet, jumping happily, consistent proportions, symmetrical features",
            "expression": "a cartoon mushroom character with a light blue mushroom cap with white dots, eyes and mouth on its body, yellow feet, smiling widely, consistent proportions, symmetrical features"
        }
        return test_prompts.get(test_type, test_prompts["basic"])
    
    def _save_parameters(self, steps, output_dir):
        """保存參數信息到文件"""
        params_filename = os.path.join(output_dir, "parameters.txt")
        with open(params_filename, 'w') as params_file:
            params_file.write(f"weight_name: {self.weight_name}\n")
            params_file.write(f"num_inference_steps: {steps}\n")
            params_file.write(f"prompt_template: {self.prompt}\n")
            params_file.write(f"negative_prompt: {self.negative_prompt}\n")
            params_file.write(f"guidance_scale: {self.guidance_scale}\n")
            params_file.write(f"strength: {self.strength}\n")
            params_file.write(f"model: {self.config.get('model')}\n")
    
    def _generate_single_image(self, steps, output_dir, action_key, expression_key):
        """生成單張圖片"""
        try:
            # 設置隨機種子
            random_seed = int(time.time())
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(random_seed)
            
            # 生成圖像
            result = self._generate_image_result(steps, generator)
            
            # 處理圖像
            image = self._extract_image_from_result(result)
            output_img = self._process_image(image)
            
            # 保存圖像
            self._save_image(output_img, output_dir, action_key, expression_key)
            
            # 釋放資源
            self._close_images(image, output_img)
            
        except Exception as e:
            print(f"⚠️ 生成圖片時出錯: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_image_result(self, steps, generator):
        """生成圖像結果"""
        # 確保提示詞包含關鍵特徵描述
        base_prompt = "a cartoon mushroom character with a light blue mushroom cap with white dots, eyes and mouth on its body, yellow feet"
        
        # 如果提示詞中沒有包含基本特徵描述，且不是測試模式，則添加
        if base_prompt not in self.prompt and self.mode != "-t":
            enhanced_prompt = f"{base_prompt}, {self.prompt}"
        else:
            enhanced_prompt = self.prompt
        
        # 確保提示詞中包含一致性和對稱性關鍵詞
        if "consistent proportions" not in enhanced_prompt:
            enhanced_prompt += ", consistent proportions"
        if "symmetrical features" not in enhanced_prompt:
            enhanced_prompt += ", symmetrical features"
        
        return self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=steps,
            height=self.height,
            width=self.width,
            guidance_scale=self.guidance_scale,
            noise_level=self.noise_level,
            init_image=self.original_image,
            strength=self.strength,
            num_images_per_prompt=1,
            generator=generator,
        )
    
    def _extract_image_from_result(self, result):
        """從結果中提取圖像"""
        if hasattr(result, "images"):
            image = result.images[0]
        else:
            image = result[0]
        
        if image is None:
            raise ValueError("生成的圖像為 None")
        
        return image
    
    def _process_image(self, image):
        """處理圖像，去背並返回處理後的圖像"""
        # 確保 image 是 PIL Image 類型
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        
        # 將圖像保存到緩衝區並重新打開
        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        input_img = Image.open(image_buffer)
        
        # 使用 rembg 去背
        return remove(input_img)
    
    def _save_image(self, output_img, output_dir, action_key, expression_key):
        """保存圖像到指定目錄"""
        if isinstance(output_img, Image.Image):
            timestamp = int(time.time())
            if self.original_image_name:
                filename = os.path.join(output_dir, f"{self.weight_name}_{self.original_image_name}_{timestamp}_transparent.png")
            else:
                filename = os.path.join(output_dir, f"{self.weight_name}_{action_key}_{expression_key}_{timestamp}_transparent.png")
            
            output_img.save(filename)
            print(f"✅ 已去背圖：{filename}")
    
    def _close_images(self, image, output_img):
        """關閉圖像以釋放資源"""
        if isinstance(image, Image.Image):
            image.close()
        if isinstance(output_img, Image.Image):
            output_img.close()
        gc.collect()
        
    def generate_single_image_api(self, steps, output_dir):
        """為 API 生成單張圖片並返回圖像路徑"""
        try:
            # 設置隨機種子
            random_seed = int(time.time())
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(random_seed)
            
            # 生成圖像
            result = self._generate_image_result(steps, generator)
            
            # 處理圖像
            image = self._extract_image_from_result(result)
            output_img = self._process_image(image)
            
            # 保存圖像
            timestamp = int(time.time())
            if self.original_image_name:
                filename = f"{self.weight_name}_{self.original_image_name}_{timestamp}_transparent.png"
            else:
                filename = f"{self.weight_name}_{self.action_key}_{self.expression_key}_{timestamp}_transparent.png"
            
            full_path = os.path.join(output_dir, filename)
            if isinstance(output_img, Image.Image):
                output_img.save(full_path)
            else:
                # 如果不是 PIL Image，嘗試轉換
                Image.fromarray(np.array(output_img)).save(full_path)
            print(f"✅ API 已生成圖像：{full_path}")
            
            # 釋放資源
            self._close_images(image, output_img)
            
            # 返回相對路徑
            return os.path.join(f"{self.weight_name}", str(steps), filename)
            
        except Exception as e:
            print(f"⚠️ API 生成圖片時出錯: {e}")
            raise e