"""
ComfyUI Custom Nodes for Ovi Video+Audio Generation
Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation

Installation:
1. Clone this repository to ComfyUI/custom_nodes/
2. Install dependencies: pip install -r requirements.txt
3. Download model weights: python download_weights.py (auto-downloads to ComfyUI/models/Ovi)
4. Download fp8 model to ComfyUI/models/Ovi/Ovi/model_fp8_e4m3fn.safetensors
5. Restart ComfyUI

Usage:
- Use "Ovi Model Loader" to load the engine
  * Default path: ComfyUI/models/Ovi (auto-detected)
  * Default mode: CPU offload + fp8 for 24GB VRAM
- Use "Ovi Text to Video+Audio" for text-to-video generation
- Use "Ovi Image to Video+Audio" for image-to-video generation

Default Configuration:
- Model path: ComfyUI/models/Ovi (follows ComfyUI standard)
- CPU offload: enabled (reduces VRAM usage)
- fp8 quantization: enabled (24GB VRAM compatible)
- If you have >80GB VRAM, disable both for faster generation
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
