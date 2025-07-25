# """
# 配置管理模塊
# 用於加載和管理配置
# """
# import yaml

# class Config:
#     """配置類，用於加載和管理配置"""
#     def __init__(self, config_path='src/config/config.yaml'):
#         self.config = self._load_config(config_path)
    
#     def _load_config(self, config_path):
#         """讀取配置文件並返回配置字典"""
#         try:
#             with open(config_path, 'r') as config_file:
#                 return yaml.safe_load(config_file)
#         except Exception as e:
#             print(f"⚠️ 加載配置文件失敗: {e}")
#             return {}
    
#     def get(self, key, default=None):
#         """獲取配置項"""
#         return self.config.get(key, default)
    
#     def __getitem__(self, key):
#         """通過索引獲取配置項"""
#         return self.config[key]

"""
配置管理模塊
用於加載和管理配置
"""
import os
import sys
import yaml

class Config:
    """配置類，用於加載和管理配置"""
    def __init__(self, config_path=None):
        # 如果沒有提供路徑，自動計算正確的路徑
        if config_path is None:
            # 獲取當前檔案 (config_manager.py) 所在的目錄
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 構造出指向 'src/config/config.yaml' 的絕對路徑
            config_path = os.path.join(current_dir, '..', 'config', 'config.yaml')
            # 規格化路徑 (例如處理 '..')
            config_path = os.path.normpath(config_path)

        # 在載入前打印路徑，便於除錯
        print(f"--- [Config] 準備從路徑加載設定檔: {config_path} ---", flush=True)
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path):
        """
        讀取配置文件並返回配置字典。
        如果失敗，則打印錯誤並終止程式。
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as config_file:
                config_data = yaml.safe_load(config_file)
                if config_data is None:
                    print(f"!!!!!!!! 致命錯誤：設定檔 {config_path} 是空的或格式不正確 !!!!!!!!", file=sys.stderr, flush=True)
                    sys.exit(1) # 終止程式
                print(f"--- [Config] 設定檔 {config_path} 加載成功 ---", flush=True)
                return config_data
        except FileNotFoundError:
            # 這是最可能在 Render 上發生的錯誤
            print(f"!!!!!!!! 致命錯誤：找不到設定檔！路徑: {config_path} !!!!!!!!", file=sys.stderr, flush=True)
            print("--- [Config] 請檢查檔案是否已推送到 Git 儲存庫，且路徑是否正確。 ---", file=sys.stderr, flush=True)
            sys.exit(1) # 終止程式
        except Exception as e:
            # 捕捉其他所有可能的錯誤，例如 YAML 語法錯誤
            print(f"!!!!!!!! 致命錯誤：加載設定檔時發生未知錯誤: {e} !!!!!!!!", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.exit(1) # 終止程式
    
    def get(self, key, default=None):
        """獲取配置項，支援模型預設配置"""
        if not self.config:
            return default
            
        # 如果是請求 parameters 或 image_size，先檢查是否有模型預設配置
        if key in ['parameters', 'image_size']:
            model_name = self.config.get('model')
            if model_name and 'models' in self.config:
                model_config = self.config['models'].get(model_name, {})
                default_key = f'default_{key}'
                if default_key in model_config:
                    # 如果配置檔案中有手動設定，則合併配置
                    manual_config = self.config.get(key, {})
                    default_config = model_config[default_key]
                    if manual_config:
                        # 手動配置優先，但缺失的項目用預設值補充
                        merged_config = default_config.copy()
                        merged_config.update(manual_config)
                        return merged_config
                    else:
                        return default_config
        
        # 如果是請求 weight_name，根據模型類型選擇對應的 LoRA 權重
        if key == 'weight_name':
            model_name = self.config.get('model')
            if model_name and 'models' in self.config and 'lora_weights' in self.config:
                model_config = self.config['models'].get(model_name, {})
                lora_type = model_config.get('compatible_lora_type')
                if lora_type and lora_type in self.config['lora_weights']:
                    return self.config['lora_weights'][lora_type]
        
        return self.config.get(key, default)
    
    def __getitem__(self, key):
        """通過索引獲取配置項"""
        if self.config:
            result = self.get(key)
            if result is not None:
                return result
            return self.config[key]
        raise KeyError(f"配置字典未初始化，無法獲取鍵 '{key}'")