import os
import argparse
import logging
import time
from huggingface_hub import snapshot_download

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

def timed_download(repo_id: str, local_dir: str, allow_patterns: list):
    """Download files from HF repo and log time + destination."""
    logging.info(f"Starting download from {repo_id} into {local_dir}")
    start_time = time.time()

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    elapsed = time.time() - start_time
    logging.info(
        f"✅ Finished downloading {repo_id} "
        f"in {elapsed:.2f} seconds. Files saved at: {local_dir}"
    )

def main(output_dir: str):
    # Wan2.2
    wan_dir = os.path.join(output_dir, "Wan2.2-TI2V-5B")
    timed_download(
        repo_id="Wan-AI/Wan2.2-TI2V-5B",
        local_dir=wan_dir,
        allow_patterns=[
            "google/*",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.2_VAE.pth"
        ]
    )

    # MMAudio
    mm_audio_dir = os.path.join(output_dir, "MMAudio")
    timed_download(
        repo_id="hkchengrex/MMAudio",
        local_dir=mm_audio_dir,
        allow_patterns=[
            "ext_weights/best_netG.pt",
            "ext_weights/v1-16.pth"
        ]
    )

    ovi_dir = os.path.join(output_dir, "Ovi")
    timed_download(
        repo_id="chetwinlow1/Ovi",
        local_dir=ovi_dir,
        allow_patterns=[
            "model.safetensors"
        ]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face")
    
    # Detect ComfyUI models directory (ComfyUI/models/Ovi)
    default_dir = None
    try:
        # Try using ComfyUI's folder_paths API
        import folder_paths
        comfyui_models_dir = folder_paths.models_dir
        default_dir = os.path.join(comfyui_models_dir, "Ovi")
    except:
        # Fallback: use relative path from custom_nodes/ComfyUI_RH_Ovi
        default_dir = "../../models/Ovi"
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_dir,
        help=f"Base directory to save downloaded models (default: ComfyUI/models/Ovi)"
    )
    args = parser.parse_args()
    
    # Create directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Using output directory: {os.path.abspath(args.output_dir)}")
    
    main(args.output_dir)