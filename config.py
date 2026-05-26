import os
from dotenv import load_dotenv

load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
BASE_URL = "https://api.siliconflow.cn/v1"

# LLM model for splitting story into scenes and generating prompts
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")

# Image generation model (default: free)
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "Kwai-Kolors/Kolors")

# Default image size
IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1024x1024")

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# SiliconFlow available models
MODEL_DEFAULTS = {
    "Kwai-Kolors/Kolors": {
        "image_sizes": ["1024x1024", "960x1280", "768x1024", "720x1440", "720x1280"],
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
    },
    "Tongyi-MAI/Z-Image": {
        "image_sizes": ["1024x1024", "1280x720", "720x1280", "1024x768", "768x1024", "864x1152", "1152x864"],
        "num_inference_steps": 20,
        "guidance_scale": 5.0,
    },
    "Tongyi-MAI/Z-Image-Turbo": {
        "image_sizes": ["1024x1024", "1280x720", "720x1280", "1024x768", "768x1024", "864x1152", "1152x864"],
        "num_inference_steps": 10,
        "guidance_scale": 5.0,
    },
    "baidu/ERNIE-Image-Turbo": {
        "image_sizes": ["1024x1024", "1280x720", "720x1280", "1024x768", "768x1024"],
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
    },
    "Qwen/Qwen-Image": {
        "image_sizes": ["1328x1328", "1664x928", "928x1664", "1472x1140", "1140x1472", "1584x1056", "1056x1584"],
        "num_inference_steps": 20,
        "guidance_scale": 5.0,
    },
}


def get_model_config():
    return MODEL_DEFAULTS.get(IMAGE_MODEL, MODEL_DEFAULTS["Kwai-Kolors/Kolors"])


def validate_config():
    if not SILICONFLOW_API_KEY:
        raise SystemExit(
            "Error: SILICONFLOW_API_KEY not set.\n"
            "Get your key at https://cloud.siliconflow.cn/account/ak\n"
            "Then set it in .env file or environment variable."
        )
