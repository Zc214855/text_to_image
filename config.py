import os
from dotenv import load_dotenv

load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
BASE_URL = "https://api.siliconflow.cn/v1"

# LLM model for splitting story into scenes and generating prompts
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")

# Image generation model
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")

# Default image size
IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1024x1024")

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# SiliconFlow models
SF_MODELS = {
    "Kwai-Kolors/Kolors": {
        "image_sizes": ["1024x1024", "960x1280", "768x1024", "720x1440", "720x1280"],
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "supports_negative_prompt": True,
    },
    "Tongyi-MAI/Z-Image": {
        "image_sizes": ["1024x1024", "1280x720", "720x1280", "1024x768", "768x1024", "864x1152", "1152x864"],
        "num_inference_steps": 20,
        "guidance_scale": 5.0,
        "supports_negative_prompt": True,
    },
    "Tongyi-MAI/Z-Image-Turbo": {
        "image_sizes": ["1024x1024", "1280x720", "720x1280", "1024x768", "768x1024", "864x1152", "1152x864"],
        "num_inference_steps": 10,
        "guidance_scale": 5.0,
        "supports_negative_prompt": True,
    },
    "baidu/ERNIE-Image-Turbo": {
        "image_sizes": ["1024x1024", "1280x720", "720x1280", "1024x768", "768x1024"],
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "supports_negative_prompt": True,
    },
    "Qwen/Qwen-Image": {
        "image_sizes": ["1328x1328", "1664x928", "928x1664", "1472x1140", "1140x1472", "1584x1056", "1056x1584"],
        "num_inference_steps": 20,
        "guidance_scale": 5.0,
        "supports_negative_prompt": True,
    },
}

# HuggingFace free models
HF_MODELS = {
    "black-forest-labs/FLUX.1-schnell": {
        "image_sizes": ["1024x1024", "768x1344", "864x1152", "1344x768", "1152x864"],
        "num_inference_steps": 4,
        "guidance_scale": None,
        "supports_negative_prompt": False,
    },
    "stabilityai/stable-diffusion-3-medium-diffusers": {
        "image_sizes": ["1024x1024", "768x1024", "1024x768"],
        "num_inference_steps": 25,
        "guidance_scale": 7.0,
        "supports_negative_prompt": True,
    },
}

ALL_MODELS = {**HF_MODELS, **SF_MODELS}


def get_model_config():
    """Get model-specific config, falling back to FLUX.1-schnell defaults."""
    return ALL_MODELS.get(IMAGE_MODEL, HF_MODELS["black-forest-labs/FLUX.1-schnell"])


def validate_config():
    from image_generator import _is_hf_model

    if _is_hf_model(IMAGE_MODEL):
        if not os.getenv("HF_TOKEN"):
            raise SystemExit(
                "Error: HF_TOKEN not set.\n"
                "Get one at https://huggingface.co/settings/tokens\n"
                "Then set it in .env file."
            )
    else:
        if not SILICONFLOW_API_KEY:
            raise SystemExit(
                "Error: SILICONFLOW_API_KEY not set.\n"
                "Get your key at https://cloud.siliconflow.cn/account/ak\n"
                "Then set it in .env file or environment variable."
            )
