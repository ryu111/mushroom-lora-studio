"""
API 測試腳本
用於測試圖像生成 API
"""
import requests
import json
import os
import sys
import time
from PIL import Image

# API 端點
API_URL = "http://localhost:10000"

def test_health():
    """測試健康檢查端點"""
    response = requests.get(f"{API_URL}/health")
    print(f"健康檢查響應: {response.status_code}")
    print(response.json())
    return response.status_code == 200

def test_models():
    """測試獲取可用模型端點"""
    response = requests.get(f"{API_URL}/models")
    print(f"獲取模型響應: {response.status_code}")
    print(response.json())
    return response.status_code == 200

def test_actions():
    """測試獲取可用動作端點"""
    response = requests.get(f"{API_URL}/actions")
    print(f"獲取動作響應: {response.status_code}")
    print(response.json())
    return response.status_code == 200

def test_expressions():
    """測試獲取可用表情端點"""
    response = requests.get(f"{API_URL}/expressions")
    print(f"獲取表情響應: {response.status_code}")
    print(response.json())
    return response.status_code == 200

def test_generate_image():
    """測試生成圖像端點"""
    # 請求數據
    data = {
        "weight_name": "mushroom-16.safetensors",
        "steps": 50,
        "action_key": "standing",
        "expression_key": "smiling",
        "guidance_scale": 7.0,
        "strength": 0.25,
        "height": 512,
        "width": 512
    }
    
    # 發送請求
    print(f"發送生成圖像請求: {json.dumps(data, indent=2)}")
    start_time = time.time()
    response = requests.post(f"{API_URL}/generate", json=data)
    elapsed_time = time.time() - start_time
    
    print(f"生成圖像響應: {response.status_code}, 耗時: {elapsed_time:.2f}秒")
    
    if response.status_code == 200:
        result = response.json()
        print(f"生成結果: {json.dumps(result, indent=2)}")
        
        # 獲取生成的圖像
        image_path = result["image_path"]
        image_url = f"{API_URL}/image/{image_path}"
        print(f"圖像 URL: {image_url}")
        
        # 下載圖像
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            # 保存圖像
            local_path = f"test_api_result_{int(time.time())}.png"
            with open(local_path, "wb") as f:
                f.write(image_response.content)
            print(f"圖像已保存到: {local_path}")
            
            # 顯示圖像
            try:
                image = Image.open(local_path)
                image.show()
            except Exception as e:
                print(f"無法顯示圖像: {e}")
        else:
            print(f"獲取圖像失敗: {image_response.status_code}")
    else:
        print(f"生成圖像失敗: {response.text}")
    
    return response.status_code == 200

def test_generate_with_original_image():
    """測試使用原始圖像生成圖像"""
    # 請求數據
    data = {
        "weight_name": "mushroom-16.safetensors",
        "steps": 50,
        "original_image_path": "assets/reference_images/dgu_01.png",
        "guidance_scale": 7.0,
        "strength": 0.25,
        "height": 512,
        "width": 512
    }
    
    # 發送請求
    print(f"發送使用原始圖像生成請求: {json.dumps(data, indent=2)}")
    start_time = time.time()
    response = requests.post(f"{API_URL}/generate", json=data)
    elapsed_time = time.time() - start_time
    
    print(f"生成圖像響應: {response.status_code}, 耗時: {elapsed_time:.2f}秒")
    
    if response.status_code == 200:
        result = response.json()
        print(f"生成結果: {json.dumps(result, indent=2)}")
        
        # 獲取生成的圖像
        image_path = result["image_path"]
        image_url = f"{API_URL}/image/{image_path}"
        print(f"圖像 URL: {image_url}")
        
        # 下載圖像
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            # 保存圖像
            local_path = f"test_api_original_{int(time.time())}.png"
            with open(local_path, "wb") as f:
                f.write(image_response.content)
            print(f"圖像已保存到: {local_path}")
            
            # 顯示圖像
            try:
                image = Image.open(local_path)
                image.show()
            except Exception as e:
                print(f"無法顯示圖像: {e}")
        else:
            print(f"獲取圖像失敗: {image_response.status_code}")
    else:
        print(f"生成圖像失敗: {response.text}")
    
    return response.status_code == 200

def main():
    """主函數"""
    print("開始測試 API...")
    
    # 測試健康檢查
    if not test_health():
        print("健康檢查失敗，API 可能未運行")
        return
    
    # 測試獲取可用模型
    test_models()
    
    # 測試獲取可用動作
    test_actions()
    
    # 測試獲取可用表情
    test_expressions()
    
    # 測試生成圖像
    test_generate_image()
    
    # 測試使用原始圖像生成圖像
    test_generate_with_original_image()
    
    print("API 測試完成")

if __name__ == "__main__":
    main()