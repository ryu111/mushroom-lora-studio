"""
配置管理模塊
用於加載和管理配置
"""
import yaml

class Config:
    """配置類，用於加載和管理配置"""
    def __init__(self, config_path='src/config/config.yaml'):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path):
        """讀取配置文件並返回配置字典"""
        try:
            with open(config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except Exception as e:
            print(f"⚠️ 加載配置文件失敗: {e}")
            return {}
    
    def get(self, key, default=None):
        """獲取配置項"""
        return self.config.get(key, default)
    
    def __getitem__(self, key):
        """通過索引獲取配置項"""
        return self.config[key]