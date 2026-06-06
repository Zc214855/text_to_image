import os
from dotenv import load_dotenv

load_dotenv()

# 部分 httpx 版本无法解析 Windows 注入的裸 IPv6 NO_PROXY 条目。
for proxy_variable in ("NO_PROXY", "no_proxy"):
    proxy_value = os.getenv(proxy_variable)
    if proxy_value:
        proxy_hosts = [
            host.strip()
            for host in proxy_value.split(",")
            if host.strip() not in {"::1", "::1/128", "[::1]"}
        ]
        os.environ[proxy_variable] = ",".join(proxy_hosts)

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = "https://api.siliconflow.cn/v1"
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_BASE_URL = os.getenv(
    "ZHIPU_BASE_URL", "https://api.lkeap.cloud.tencent.com/plan/v3"
)
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-5.1")
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
ARK_BASE_URL = os.getenv(
    "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
)
ARK_LLM_MODEL = os.getenv("ARK_LLM_MODEL", "doubao-seed-2-0-lite-260215")
ARK_IMAGE_MODEL = os.getenv("ARK_IMAGE_MODEL", "doubao-seedream-5-0-260128")
FREELLMAPI_BASE_URL = os.getenv(
    "FREELLMAPI_BASE_URL", "http://localhost:3001/v1"
)
FREELLMAPI_API_KEY = os.getenv(
    "FREELLMAPI_API_KEY", "freellmapi-placeholder"
)
FREELLMAPI_MODEL = os.getenv("FREELLMAPI_MODEL", "auto")

# ---------- LLM 配置 ----------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "siliconflow").strip().lower()
_LLM_DEFAULTS = {
    "siliconflow": (BASE_URL, SILICONFLOW_API_KEY, "Qwen/Qwen3-8B"),
    "freellmapi": (
        FREELLMAPI_BASE_URL,
        FREELLMAPI_API_KEY,
        FREELLMAPI_MODEL,
    ),
    "zhipu": (ZHIPU_BASE_URL, ZHIPU_API_KEY, ZHIPU_MODEL),
    "volcengine": (ARK_BASE_URL, ARK_API_KEY, ARK_LLM_MODEL),
}
LLM_PROVIDERS = {
    "zhipu": {
        "label": "智谱 GLM-5.1（当前默认）",
        "summary": "通过腾讯云 LKEAP Token Plan 调用，负责故事理解、角色设定和分镜提示词。",
    },
    "siliconflow": {
        "label": "Qwen3-8B · 硅基流动",
        "summary": "使用硅基流动上的 Qwen3-8B，成本较低，但复杂结构化分镜能力弱于 GLM-5.1。",
    },
    "freellmapi": {
        "label": "FreeLLMAPI（自动路由）",
        "summary": "调用本机 localhost:3001 代理，由代理自动选择已配置的免费文本模型。",
    },
    "volcengine": {
        "label": "豆包 Seed 2.0 Lite · 火山方舟",
        "summary": "使用火山方舟文本模型分析故事；与 Seedream 图片模型是两个不同模型。",
    },
}
_DEFAULT_LLM_BASE_URL, _DEFAULT_LLM_API_KEY, _DEFAULT_LLM_MODEL = (
    _LLM_DEFAULTS.get(LLM_PROVIDER, _LLM_DEFAULTS["siliconflow"])
)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", _DEFAULT_LLM_BASE_URL).rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", _DEFAULT_LLM_API_KEY)
LLM_MODEL = os.getenv("LLM_MODEL", _DEFAULT_LLM_MODEL)


def set_llm_provider(provider: str):
    """运行时切换分镜 LLM；仅影响当前进程，不修改 .env。"""
    global LLM_PROVIDER, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

    if provider not in _LLM_DEFAULTS:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    LLM_PROVIDER = provider
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL = _LLM_DEFAULTS[provider]


def get_llm_provider_config(provider: str):
    """返回供应商的界面说明及实际连接配置。"""
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    base_url, api_key, model = _LLM_DEFAULTS[provider]
    return {
        **LLM_PROVIDERS[provider],
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
    }

# ---------- 图片生成模型 ----------
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "Kwai-Kolors/Kolors")
IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1024x1024")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# ---------- SiliconFlow 图片模型 ----------
SF_MODELS = {
    "Kwai-Kolors/Kolors": {
        "label": "Kolors（免费基础款）· 硅基流动",
        "provider": "siliconflow",
        "image_sizes": ["1024x1024", "960x1280", "768x1024", "720x1440", "720x1280"],
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "price": "免费",
        "summary": "免费文生图模型，适合调试流程和低成本预览；复杂构图、文字控制和角色一致性较弱。",
    },
    "Tongyi-MAI/Z-Image-Turbo": {
        "label": "Z-Image Turbo（快速低价）· 硅基流动",
        "provider": "siliconflow",
        "image_sizes": ["1024x1024", "1280x720", "720x1280", "1024x768", "768x1024", "864x1152", "1152x864"],
        "num_inference_steps": 10,
        "guidance_scale": 5.0,
        "price": "¥0.10/张",
        "summary": "Z-Image 快速版，速度和成本优先，适合批量草图与构图试错。",
    },
    "Tongyi-MAI/Z-Image": {
        "label": "Z-Image（质量版）· 硅基流动",
        "provider": "siliconflow",
        "image_sizes": ["1024x1024", "1280x720", "720x1280", "1024x768", "768x1024", "864x1152", "1152x864"],
        "num_inference_steps": 20,
        "guidance_scale": 5.0,
        "price": "¥0.30/张",
        "summary": "Z-Image 质量版，比 Turbo 使用更多推理步数，适合追求细节的单张插图。",
    },
    "baidu/ERNIE-Image-Turbo": {
        "label": "ERNIE Image Turbo（中文理解）· 硅基流动",
        "provider": "siliconflow",
        "image_sizes": ["1024x1024", "1280x720", "720x1280", "1024x768", "768x1024"],
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "price": "¥0.11/张",
        "summary": "百度文心图像模型的快速版本，偏中文语义理解和常规文生图。",
    },
    "Qwen/Qwen-Image": {
        "label": "Qwen Image（复杂提示词）· 硅基流动",
        "provider": "siliconflow",
        "image_sizes": ["1328x1328", "1664x928", "928x1664", "1472x1140", "1140x1472", "1584x1056", "1056x1584"],
        "num_inference_steps": 20,
        "guidance_scale": 5.0,
        "price": "¥0.30/张",
        "summary": "千问图像模型，适合复杂提示词、中文语义和高分辨率横竖构图。",
    },
}

# ---------- 阿里云百炼(DashScope) 图片模型 ----------
DS_MODELS = {
    "wanx2.1-t2i-turbo": {
        "label": "万相 2.1 Turbo（旧版快速）· 阿里云百炼",
        "provider": "dashscope",
        "image_sizes": ["1024*1024", "720*1280", "1280*720", "768*1024", "1024*768"],
        "price": "按百炼控制台计费",
        "summary": "仍可调用的旧版万相模型，生成速度优先；百炼当前更推荐万相 2.7 系列。",
    },
    "wanx2.1-t2i-plus": {
        "label": "万相 2.1 Plus（旧版质量）· 阿里云百炼",
        "provider": "dashscope",
        "image_sizes": ["1024*1024", "720*1280", "1280*720", "768*1024", "1024*768"],
        "price": "按百炼控制台计费",
        "summary": "仍可调用的旧版万相质量模型，细节高于 2.1 Turbo；百炼当前更推荐万相 2.7 系列。",
    },
    "qwen-image-plus": {
        "label": "Qwen Image Plus（高分辨率）· 阿里云百炼",
        "provider": "dashscope",
        "image_sizes": ["1664*928", "928*1664", "1472*1104", "1104*1472", "1328*1328"],
        "price": "按百炼控制台计费",
        "summary": "百炼版千问图像单图模型，适合复杂语义和高分辨率；新项目可评估 Qwen Image 2.0。",
    },
}

# ---------- 火山方舟 Seedream 图片模型 ----------
ARK_MODELS = {
    "doubao-seedream-5-0-260128": {
        "label": "Seedream 5.0 Lite（当前推荐）· 火山方舟",
        "provider": "volcengine",
        "image_sizes": [
            "2048x2048",
            "2304x1728",
            "1728x2304",
            "2848x1600",
            "1600x2848",
            "2496x1664",
            "1664x2496",
        ],
        "price": "约 ¥0.22/张，以方舟控制台为准",
        "summary": "当前默认模型。提示词理解、细节和多主体构图较强；官方还支持参考图与组图，本工具当前使用文生单图。",
    },
    "doubao-seedream-4-0-250828": {
        "label": "Seedream 4.0（1K 预览）· 火山方舟",
        "provider": "volcengine",
        "image_sizes": [
            "1024x1024",
            "1152x864",
            "864x1152",
            "1312x736",
            "736x1312",
            "1248x832",
            "832x1248",
        ],
        "price": "约 ¥0.20/张，以方舟控制台为准",
        "summary": "支持 1K 输出，适合低分辨率预览和构图确认；质量与提示词理解弱于 Seedream 5.0 Lite。",
    },
}

ALL_MODELS = {**SF_MODELS, **DS_MODELS, **ARK_MODELS}

PROVIDER_LABELS = {
    "siliconflow": "硅基流动",
    "dashscope": "阿里云百炼",
    "volcengine": "火山方舟",
}


def get_provider(model: str) -> str:
    cfg = ALL_MODELS.get(model, {})
    return cfg.get("provider", "siliconflow")


def get_model_config(model: str | None = None):
    """返回指定模型配置，避免界面切换模型后读取旧配置。"""
    selected_model = model or IMAGE_MODEL
    return ALL_MODELS.get(selected_model, SF_MODELS["Kwai-Kolors/Kolors"])


def get_llm_client_config():
    """返回 OpenAI 兼容 LLM 调用所需配置。"""
    if LLM_PROVIDER not in _LLM_DEFAULTS:
        supported = ", ".join(_LLM_DEFAULTS)
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Supported: {supported}"
        )
    if not LLM_API_KEY:
        raise ValueError(f"LLM provider '{LLM_PROVIDER}' API key is not set")
    if not LLM_MODEL:
        raise ValueError(f"LLM provider '{LLM_PROVIDER}' model is not set")
    return LLM_BASE_URL, LLM_API_KEY, LLM_MODEL


def validate_config(model: str | None = None):
    selected_model = model or IMAGE_MODEL
    provider = get_provider(selected_model)
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
    if provider == "volcengine" and not ARK_API_KEY:
        raise SystemExit(
            "Error: ARK_API_KEY not set.\n"
            "Get one at https://console.volcengine.com/ark/region:ark+cn-beijing/apikey\n"
            "Then set it in .env file."
        )
