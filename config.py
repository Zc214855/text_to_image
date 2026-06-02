import os
from dotenv import load_dotenv

load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = "https://api.siliconflow.cn/v1"

# LLM: 直接走 SiliconFlow
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B")

# 图片生成模型
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "Kwai-Kolors/Kolors")
IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1024x1024")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# SiliconFlow 图片模型
SF_MODELS = {
    "Kwai-Kolors/Kolors": {
        "provider": "siliconflow",
        "image_sizes": ["1024x1024", "960x1280", "768x1024", "720x1440", "720x1280"],
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "price": "免费",
    },
    "Tongyi-MAI/Z-Image-Turbo": {
        "provider": "siliconflow",
        "image_sizes": ["1024x1024", "1280x720", "720x1280", "1024x768", "768x1024", "864x1152", "1152x864"],
        "num_inference_steps": 10,
        "guidance_scale": 5.0,
        "price": "¥0.10/张",
    },
    "Tongyi-MAI/Z-Image": {
        "provider": "siliconflow",
        "image_sizes": ["1024x1024", "1280x720", "720x1280", "1024x768", "768x1024", "864x1152", "1152x864"],
        "num_inference_steps": 20,
        "guidance_scale": 5.0,
        "price": "¥0.30/张",
    },
    "baidu/ERNIE-Image-Turbo": {
        "provider": "siliconflow",
        "image_sizes": ["1024x1024", "1280x720", "720x1280", "1024x768", "768x1024"],
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "price": "¥0.11/张",
    },
    "Qwen/Qwen-Image": {
        "provider": "siliconflow",
        "image_sizes": ["1328x1328", "1664x928", "928x1664", "1472x1140", "1140x1472", "1584x1056", "1056x1584"],
        "num_inference_steps": 20,
        "guidance_scale": 5.0,
        "price": "¥0.30/张",
    },
}

# 阿里云百炼(DashScope) 图片模型
DS_MODELS = {
    "wanx2.1-t2i-turbo": {
        "provider": "dashscope",
        "image_sizes": ["1024*1024", "720*1280", "1280*720", "768*1024", "1024*768"],
        "price": "新用户免费送额度",
    },
    "wanx2.1-t2i-plus": {
        "provider": "dashscope",
        "image_sizes": ["1024*1024", "720*1280", "1280*720", "768*1024", "1024*768"],
        "price": "新用户免费送额度",
    },
    "qwen-image-plus": {
        "provider": "dashscope",
        "image_sizes": ["1664*928", "928*1664", "1472*1104", "1104*1472", "1328*1328"],
        "price": "新用户免费送额度",
    },
}

ALL_MODELS = {**SF_MODELS, **DS_MODELS}


def get_provider(model: str) -> str:
    cfg = ALL_MODELS.get(model, {})
    return cfg.get("provider", "siliconflow")


def get_model_config():
    return ALL_MODELS.get(IMAGE_MODEL, SF_MODELS["Kwai-Kolors/Kolors"])


def validate_config():
    provider = get_provider(IMAGE_MODEL)
    if provider == "dashscope" and not DASHSCOPE_API_KEY:
        raise SystemExit(
            "Error: DASHSCOPE_API_KEY not set.\n"
            "Get one at https://bailian.console.aliyun.com/\n"
            "Then set it in .env file."
        )
    if provider == "siliconflow" and not SILICONFLOW_API_KEY:
        raise SystemExit(
            "Error: SILICONFLOW_API_KEY not set.\n"
            "Get your key at https://cloud.siliconflow.cn/account/ak\n"
            "Then set it in .env file."
        )
