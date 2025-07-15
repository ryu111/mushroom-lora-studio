# Stable Diffusion XL Base 1.0 下載連結

## 🔗 直接下載連結

### 主要模型檔案：

1. **UNet** (最大檔案 ~10.3GB)：
   ```
   https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.safetensors
   ```

2. **Text Encoder 1** (~492MB)：
   ```
   https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.safetensors
   ```

3. **Text Encoder 2** (~2.78GB)：
   ```
   https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.safetensors
   ```

4. **VAE** (~335MB)：
   ```
   https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae/diffusion_pytorch_model.safetensors
   ```

### 配置檔案：

5. **模型配置**：
   ```
   https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/model_index.json
   ```

6. **UNet 配置**：
   ```
   https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/config.json
   ```

7. **Text Encoder 配置**：
   ```
   https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/config.json
   https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/config.json
   ```

8. **VAE 配置**：
   ```
   https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae/config.json
   ```

### Tokenizer 檔案：

9. **Tokenizer**：
   ```
   https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer/tokenizer.json
   https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer_2/tokenizer.json
   ```

## 📁 本地目錄結構

下載後請放置在以下目錄結構：
```
~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/
├── snapshots/
│   └── [hash]/
│       ├── model_index.json
│       ├── unet/
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       ├── text_encoder/
│       │   ├── config.json
│       │   └── model.safetensors
│       ├── text_encoder_2/
│       │   ├── config.json
│       │   └── model.safetensors
│       ├── vae/
│       │   ├── config.json
│       │   └── diffusion_pytorch_model.safetensors
│       └── tokenizer/
│           └── tokenizer.json
```

## ⚠️ Mac M1 記憶體限制警告

SD XL 在 Mac M1 上可能會遇到 MPS 記憶體溢出問題：
- SD XL 需要約 15-16GB 記憶體
- Mac M1 Pro 的 MPS 限制約 18GB
- 建議使用較小的圖像尺寸 (512x512 而非 1024x1024)
- 或考慮使用 CPU 模式 (設定 FORCE_CPU=true)

## 💡 快速下載指令

```bash
# 創建目錄
mkdir -p ~/.cache/huggingface/models/stable-diffusion-xl-base-1.0

# 使用 wget 下載 (Linux/Mac)
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.safetensors

# 使用 curl 下載 (Mac)
curl -L -o diffusion_pytorch_model.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.safetensors