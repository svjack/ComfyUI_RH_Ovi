# ComfyUI Ovi 节点

适用于 ComfyUI 的自定义节点，集成 Ovi 模型实现文本或图像输入的同步视频+音频生成。

## ✨ 功能特性

* 🎬 **联合视频+音频生成**: 同时生成同步的视频和音频内容
* 📝 **文本到视频+音频**: 从文本提示创建带语音和音效的视频
* 🖼️ **图像到视频+音频**: 从图像和文本输入生成视频
* ⏱️ **5秒视频**: 24 FPS，720×720 区域，支持多种宽高比（9:16、16:9、1:1 等）
* ⚙️ **显存优化**: FP8 精度 + CPU 卸载，支持 24GB 显存 GPU
* 🚀 **灵活控制**: 高级参数控制，精细调整生成质量

## 🔧 节点列表

### 核心节点
* **RunningHub Ovi Model Loader**: 加载和初始化 Ovi 引擎，支持优化选项
* **RunningHub Ovi Text to Video**: 从文本提示生成视频+音频
* **RunningHub Ovi Image to Video**: 从图像和文本输入生成视频+音频

## 🚀 快速安装

### 步骤 1: 安装节点

```bash
# 进入 ComfyUI custom_nodes 目录
cd ComfyUI/custom_nodes/

# 克隆仓库
git clone https://github.com/HM-RunningHub/ComfyUI_RH_Ovi.git

cd ComfyUI_RH_Ovi

# 安装 PyTorch（如果尚未安装）
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# 安装依赖
pip install -r requirements.txt

# 安装 Flash Attention
pip install flash_attn --no-build-isolation
```

### 步骤 2: 下载所需模型

```bash
# 下载模型（默认下载到 ComfyUI/models/Ovi）
python download_weights.py

# 下载 fp8 量化模型（用于 24GB 显存模式）
cd ../../models/Ovi
wget -O "model_fp8_e4m3fn.safetensors" \
  "https://huggingface.co/rkfg/Ovi-fp8_quantized/resolve/main/model_fp8_e4m3fn.safetensors"
cd ../../custom_nodes/ComfyUI_RH_Ovi

# 最终模型结构应该是：
# ComfyUI/models/Ovi/
# ├── MMAudio/
# │   └── ext_weights/
# │       ├── best_netG.pt
# │       └── v1-16.pth
# ├── Ovi/
# │   ├── model.safetensors
# │   └── model_fp8_e4m3fn.safetensors
# └── Wan2.2-TI2V-5B/
#     ├── google/umt5-xxl/
#     ├── models_t5_umt5-xxl-enc-bf16.pth
#     └── Wan2.2_VAE.pth

# 重启 ComfyUI
```

## 📖 使用方法

### 基础工作流

```
[RunningHub Ovi Model Loader] → [RunningHub Ovi Text to Video] → [保存/预览视频]
```

### 提示词格式

Ovi 使用特殊标签来控制语音和音频：

* **语音**: `<S>你的语音内容<E>` - 文本将被转换为语音
* **音频描述**: `<AUDCAP>音频描述<ENDAUDCAP>` - 描述视频中的音频/音效

**提示词示例**:
```
<S>你好世界！<E> <AUDCAP>轻柔的钢琴音乐<ENDAUDCAP>
```

### 生成类型

#### 文本到视频+音频
1. 将 `RunningHub Ovi Model Loader` 连接到 `RunningHub Ovi Text to Video`
2. 输入带有语音和音频标签的文本提示词
3. 设置视频尺寸、种子和生成参数
4. 生成同步的视频+音频

#### 图像到视频+音频
1. 使用 ComfyUI 的 `Load Image` 节点加载图像
2. 将图像和 `ovi_engine` 连接到 `RunningHub Ovi Image to Video`
3. 输入带有语音和音频标签的文本提示词
4. 基于图像生成视频+音频

### 示例提示词

* **文本到视频**: 参见 [example_prompts/gpt_examples_t2v.csv](example_prompts/gpt_examples_t2v.csv)
* **图像到视频**: 参见 [example_prompts/gpt_examples_i2v.csv](example_prompts/gpt_examples_i2v.csv)

## 🛠️ 技术要求

* **GPU**: 24GB+ 显存（使用 CPU 卸载 + FP8 优化）
  * 32GB+ 显存（不使用优化）
* **内存**: 推荐 32GB+
* **存储**: 所有模型约 30GB
  * Ovi 模型: ~12GB
  * MMAudio: ~2GB
  * Wan2.2-TI2V-5B: ~13GB
  * FP8 量化模型: ~6GB
* **CUDA**: 需要 CUDA 支持以获得最佳性能

## ⚠️ 重要说明

* **模型路径**: 模型必须放置在 `ComfyUI/models/Ovi/` 目录下
* **默认配置**: Model Loader 默认使用 CPU 卸载 + FP8，适配 24GB 显存
  * 如有 32GB+ 显存，可禁用两项优化（质量更好，速度更快）
* **FP8 模型**: 24GB 显存模式必需（质量略有下降）
* 首次使用前必须下载所有模型文件

## 📄 许可证

本项目基于原始 [Ovi](https://github.com/character-ai/Ovi) 项目。

## 🔗 参考链接

* [Ovi 项目主页](https://aaxwaz.github.io/Ovi/)
* [Ovi 论文](https://arxiv.org/abs/2510.01284)
* [Ovi HuggingFace](https://huggingface.co/chetwinlow1/Ovi)
* [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## ⭐ 引用

如果您觉得本项目有用，请考虑引用原始 Ovi 论文：

```bibtex
@misc{low2025ovitwinbackbonecrossmodal,
      title={Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation}, 
      author={Chetwin Low and Weimin Wang and Calder Katyal},
      year={2025},
      eprint={2510.01284},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2510.01284}, 
}
```
