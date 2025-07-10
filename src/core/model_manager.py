"""
模型管理模塊
用於加載和管理模型
(偵錯修正版)
"""
import sys
import importlib

print("--- [ModelManager] 模塊開始被導入... ---", flush=True)

class ModelManager:
    """模型管理類，用於加載和管理模型"""
    def __init__(self, config):
        print("--- [ModelManager] __init__ 開始執行... ---", flush=True)
        self.config = config
        # 從設定檔獲取模型名稱，如果沒有則使用預設值
        self.model_name = self.config.get('model', 'StableDiffusionV15Model')
        print(f"--- [ModelManager] __init__ 完成。將要使用的模型名稱為: {self.model_name} ---", flush=True)
    
    def load_model(self, weight_name):
        """加載模型"""
        print(f"--- [ModelManager] load_model 開始執行，準備加載模型: {self.model_name} ---", flush=True)
        # 動態創建模型實例
        model_instance = self._create_model_instance()
        
        # 加載模型管道 (這一步可能會非常耗時和耗資源)
        print(f"--- [ModelManager] 準備調用 {self.model_name}.load_pipeline()... ---", flush=True)
        pipe = model_instance.load_pipeline()
        print("--- [ModelManager] pipeline 加載成功。---", flush=True)
        
        # 加載 LoRA 權重
        if weight_name:
            print(f"--- [ModelManager] 準備為 pipeline 加載 LoRA 權重: {weight_name} ---", flush=True)
            pipe = model_instance.load_lora_weights(pipe, weight_name)
            print("--- [ModelManager] LoRA 權重加載成功。---", flush=True)
        
        return pipe
    
    def _create_model_instance(self):
        """
        創建模型實例。
        這是動態導入的核心，錯誤處理被強化。
        """
        print(f"--- [ModelManager] _create_model_instance 開始，目標模型: {self.model_name} ---", flush=True)
        
        # 將駝峰命名轉換為下劃線命名 (例如 StableDiffusionV15 -> stable_diffusion_v15)
        model_name_for_file = self.model_name
        if model_name_for_file.endswith("Model"):
            model_name_for_file = model_name_for_file[:-5]
        module_file = ''.join(['_' + c.lower() if c.isupper() else c.lower() for c in model_name_for_file]).lstrip('_')
        
        try:
            # 主要嘗試路徑: src.models
            module_name = f"src.models.{module_file}"
            print(f"--- [ModelManager] 嘗試從模塊導入: {module_name} ---", flush=True)
            module = importlib.import_module(module_name)
            model_class = getattr(module, self.model_name)
            print(f"--- [ModelManager] 成功從 {module_name} 找到類別: {self.model_name} ---", flush=True)
            return model_class()
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            print(f"--- [ModelManager] 從 {module_name} 導入失敗: {e}。嘗試備用路徑... ---", flush=True)
            # 可以在此添加其他備用路徑，但為了除錯，我們先專注於主要路徑
            # 在這裡，如果主要路徑失敗，我們應該直接拋出錯誤，而不是靜默地使用預設值
            
            # --- 延遲加載 (Lazy Loading) 的應用 ---
            # 只有在所有動態導入都失敗時，才嘗試導入預設的 StableDiffusionV15Model 作為最後手段
            print("--- [ModelManager] 所有動態導入嘗試均失敗，準備導入預設的 StableDiffusionV15Model... ---", flush=True)
            try:
                # 在這裡才進行導入，而不是在檔案頂部
                from src.models.stable_diffusion_v1_5 import StableDiffusionV15Model
                print("--- [ModelManager] 預設模型 StableDiffusionV15Model 導入成功，將使用它。 ---", flush=True)
                return StableDiffusionV15Model()
            except Exception as final_e:
                print(f"!!!!!!!! 致命錯誤：連預設的 StableDiffusionV15Model 都無法導入！錯誤: {final_e} !!!!!!!!", file=sys.stderr, flush=True)
                # 拋出一個更清晰的異常，讓 FastAPI 可以捕捉並返回 500 錯誤
                raise ImportError(f"無法載入指定的模型 {self.model_name}，也無法載入預設的 StableDiffusionV15Model。請檢查模型檔案和路徑。")

print("--- [ModelManager] 模塊已成功被定義。---", flush=True)