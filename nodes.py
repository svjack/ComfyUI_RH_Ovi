"""
ComfyUI Custom Nodes for Ovi Video+Audio Generation
Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation
"""

import os
import sys
import torch
import numpy as np
import tempfile
import folder_paths
from PIL import Image
import comfy.utils

# Add current directory to Python path to allow importing ovi module
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ovi.ovi_fusion_engine import OviFusionEngine, DEFAULT_CONFIG
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import clean_text
from omegaconf import OmegaConf

# Import VideoFromFile for VIDEO type conversion
try:
    from comfy_api.input_impl import VideoFromFile
    VIDEO_SUPPORT = True
except ImportError:
    print("[RH_Ovi WARNING] VideoFromFile not available, using fallback method")
    VIDEO_SUPPORT = False


# Global engine instance to avoid reloading
_global_ovi_engine = None
_global_engine_config = None


class OviModelLoader:
    """
    Load Ovi Fusion Engine for video+audio generation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cpu_offload": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled"
                }),
                "fp8": ("BOOLEAN", {
                    "default": True,
                    "label_on": "enabled",
                    "label_off": "disabled"
                }),
            }
        }
    
    RETURN_TYPES = ("OVI_ENGINE",)
    RETURN_NAMES = ("ovi_engine",)
    FUNCTION = "load_model"
    CATEGORY = "Runninghub/Ovi"
    
    def load_model(self, cpu_offload, fp8):
        global _global_ovi_engine, _global_engine_config
        
        # Auto-detect ComfyUI models directory
        try:
            comfyui_models_dir = folder_paths.models_dir
            ckpt_dir = os.path.join(comfyui_models_dir, "Ovi")
        except:
            # Fallback: use relative path from custom_nodes
            ckpt_dir = "../../models/Ovi"
        
        print(f"Using model directory: {ckpt_dir}")
        
        # Create config
        config = OmegaConf.create({
            "ckpt_dir": ckpt_dir,
            "cpu_offload": cpu_offload,
            "fp8": fp8,
            "mode": "t2v"  # Will be overridden by generation nodes
        })
        
        # Check if we need to reload
        config_key = (ckpt_dir, cpu_offload, fp8)
        if _global_ovi_engine is None or _global_engine_config != config_key:
            print(f"Loading Ovi Fusion Engine... (cpu_offload={cpu_offload}, fp8={fp8})")
            _global_ovi_engine = OviFusionEngine(
                config=config,
                device=0,
                target_dtype=torch.bfloat16
            )
            _global_engine_config = config_key
            print("Ovi Fusion Engine loaded!")
        else:
            print("Using cached Ovi Fusion Engine")
        
        return (_global_ovi_engine,)


class OviTextToVideo:
    """
    Generate video+audio from text prompt using Ovi
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ovi_engine": ("OVI_ENGINE",),
                "text_prompt": ("STRING", {
                    "default": "<S>Hello world!<E> <AUDCAP>Background music playing<ENDAUDCAP>",
                    "multiline": True
                }),
                "video_height": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 1280,
                    "step": 32
                }),
                "video_width": ("INT", {
                    "default": 992,
                    "min": 128,
                    "max": 1280,
                    "step": 32
                }),
                "seed": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "sample_steps": ("INT", {
                    "default": 50,
                    "min": 20,
                    "max": 100,
                    "step": 1
                }),
                "solver_name": (["unipc", "euler", "dpm++"], {
                    "default": "unipc"
                }),
                "shift": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5
                }),
                "video_guidance_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5
                }),
                "audio_guidance_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5
                }),
                "slg_layer": ("INT", {
                    "default": 11,
                    "min": -1,
                    "max": 30,
                    "step": 1
                }),
            },
            "optional": {
                "video_negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "audio_negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "Runninghub/Ovi"
    
    def generate(self, ovi_engine, text_prompt, video_height, video_width, 
                 seed, sample_steps, solver_name, shift, 
                 video_guidance_scale, audio_guidance_scale, slg_layer,
                 video_negative_prompt="", audio_negative_prompt=""):
        
        try:
            # Create progress bar
            pbar = comfy.utils.ProgressBar(sample_steps)
            
            def progress_callback():
                pbar.update(1)
            
            generated_video, generated_audio, _ = ovi_engine.generate(
                text_prompt=text_prompt,
                image_path=None,
                video_frame_height_width=[video_height, video_width],
                seed=seed,
                solver_name=solver_name,
                sample_steps=sample_steps,
                shift=shift,
                video_guidance_scale=video_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                slg_layer=slg_layer,
                video_negative_prompt=video_negative_prompt,
                audio_negative_prompt=audio_negative_prompt,
                progress_callback=progress_callback,
            )
            
            # Save video to temp file
            tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path = tmpfile.name
            save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
            
            # Convert mp4 file to VIDEO object
            if VIDEO_SUPPORT:
                video_object = VideoFromFile(output_path)
            else:
                # Fallback: return file path as video object
                video_object = output_path
            
            print(f"[RH_Ovi INFO] Video saved to: {output_path}")
            
            return (video_object,)
            
        except Exception as e:
            print(f"[RH_Ovi ERROR] Error during video generation: {e}")
            import traceback
            traceback.print_exc()
            raise e


class OviImageToVideo:
    """
    Generate video+audio from image and text prompt using Ovi
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ovi_engine": ("OVI_ENGINE",),
                "image": ("IMAGE",),
                "text_prompt": ("STRING", {
                    "default": "<S>Hello world!<E> <AUDCAP>Background music playing<ENDAUDCAP>",
                    "multiline": True
                }),
                "seed": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "sample_steps": ("INT", {
                    "default": 50,
                    "min": 20,
                    "max": 100,
                    "step": 1
                }),
                "solver_name": (["unipc", "euler", "dpm++"], {
                    "default": "unipc"
                }),
                "shift": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5
                }),
                "video_guidance_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5
                }),
                "audio_guidance_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5
                }),
                "slg_layer": ("INT", {
                    "default": 11,
                    "min": -1,
                    "max": 30,
                    "step": 1
                }),
            },
            "optional": {
                "video_negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "audio_negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "Runninghub/Ovi"
    
    def generate(self, ovi_engine, image, text_prompt, seed, 
                 sample_steps, solver_name, shift,
                 video_guidance_scale, audio_guidance_scale, slg_layer,
                 video_negative_prompt="", audio_negative_prompt=""):
        
        try:
            # Convert ComfyUI image format (B, H, W, C) to PIL Image
            # ComfyUI images are in [0, 1] float format
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            
            # Save to temp file
            tmpfile_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            pil_image.save(tmpfile_img.name)
            image_path = tmpfile_img.name
            
            # Create progress bar
            pbar = comfy.utils.ProgressBar(sample_steps)
            
            def progress_callback():
                pbar.update(1)
            
            generated_video, generated_audio, _ = ovi_engine.generate(
                text_prompt=text_prompt,
                image_path=image_path,
                video_frame_height_width=None,  # Will be inferred from image
                seed=seed,
                solver_name=solver_name,
                sample_steps=sample_steps,
                shift=shift,
                video_guidance_scale=video_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                slg_layer=slg_layer,
                video_negative_prompt=video_negative_prompt,
                audio_negative_prompt=audio_negative_prompt,
                progress_callback=progress_callback,
            )
            
            # Clean up temp image
            os.unlink(image_path)
            
            # Save video to temp file
            tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            output_path = tmpfile.name
            save_video(output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
            
            # Convert mp4 file to VIDEO object
            if VIDEO_SUPPORT:
                video_object = VideoFromFile(output_path)
            else:
                # Fallback: return file path as video object
                video_object = output_path
            
            print(f"[RH_Ovi INFO] Video saved to: {output_path}")
            
            return (video_object,)
            
        except Exception as e:
            print(f"[RH_Ovi ERROR] Error during video generation: {e}")
            import traceback
            traceback.print_exc()
            raise e


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "RunningHub Ovi Model Loader": OviModelLoader,
    "RunningHub Ovi Text to Video": OviTextToVideo,
    "RunningHub Ovi Image to Video": OviImageToVideo,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub Ovi Model Loader": "RunningHub Ovi Model Loader",
    "RunningHub Ovi Text to Video": "RunningHub Ovi Text to Video",
    "RunningHub Ovi Image to Video": "RunningHub Ovi Image to Video",
}
