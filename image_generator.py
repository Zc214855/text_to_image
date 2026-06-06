import os
import time
import httpx
from config import (
    SILICONFLOW_API_KEY,
    DASHSCOPE_API_KEY,
    ARK_API_KEY,
    ARK_BASE_URL,
    BASE_URL,
    OUTPUT_DIR,
    get_provider,
    get_model_config,
)

SF_API_URL = f"{BASE_URL}/images/generations"
DS_ASYNC_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
DS_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks"
ARK_IMAGE_URL = f"{ARK_BASE_URL}/images/generations"
NEGATIVE_PROMPT = (
    "text, typography, letters, words, captions, subtitles, signs, labels, "
    "logo, watermark, speech bubble, malformed hands, extra fingers, duplicate characters"
)


class ImageDownloadError(RuntimeError):
    """图片已生成，但下载阶段失败；禁止重新提交付费生图请求。"""


def _get_model():
    import config
    return config.IMAGE_MODEL


def _get_size():
    import config
    return config.IMAGE_SIZE


def _sf_generate(prompt: str, model: str, size: str, seed: int | None = None) -> str:
    """SiliconFlow image generation, returns image URL."""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    model_config = get_model_config(model)
    payload = {
        "model": model,
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "image_size": size,
        "num_inference_steps": model_config["num_inference_steps"],
    }
    if model_config.get("guidance_scale") is not None:
        payload["guidance_scale"] = model_config["guidance_scale"]
    if seed is not None:
        payload["seed"] = seed

    resp = httpx.post(SF_API_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    images = data.get("images", [])
    if not images or not images[0].get("url"):
        raise ValueError(f"No image URL in response: {data}")
    return images[0]["url"]


def _ds_generate(prompt: str, model: str, size: str, seed: int | None = None) -> str:
    """DashScope async image generation, returns image URL."""
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    # DashScope uses * as size separator, convert x to *
    ds_size = size.replace("x", "*")
    payload: dict = {
        "model": model,
        "input": {"prompt": prompt},
        "parameters": {
            "size": ds_size,
            "n": 1,
            "negative_prompt": NEGATIVE_PROMPT,
        },
    }
    if seed is not None:
        payload["parameters"]["seed"] = seed

    # Step 1: submit task
    resp = httpx.post(DS_ASYNC_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    task_id = data.get("output", {}).get("task_id")
    if not task_id:
        raise ValueError(f"DashScope submit failed: {data}")

    # Step 2: poll for result
    poll_headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
    for _ in range(60):  # max 5 min
        time.sleep(5)
        resp = httpx.get(f"{DS_TASK_URL}/{task_id}", headers=poll_headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("output", {}).get("task_status", "")
        if status == "SUCCEEDED":
            results = data.get("output", {}).get("results", [])
            if results and results[0].get("url"):
                return results[0]["url"]
            raise ValueError(f"DashScope succeeded but no URL: {data}")
        if status == "FAILED":
            raise ValueError(f"DashScope task failed: {data}")
        # PENDING / RUNNING -> continue polling

    raise TimeoutError("DashScope task timed out")


def _ark_generate(prompt: str, model: str, size: str) -> str:
    """火山方舟 Seedream 图片生成，返回限时图片 URL。"""
    headers = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "response_format": "url",
        "watermark": False,
        "sequential_image_generation": "disabled",
        "optimize_prompt_options": {"mode": "standard"},
    }
    if model in {
        "doubao-seedream-5-0-260128",
        "doubao-seedream-4-0-250828",
    }:
        payload["output_format"] = "png"
    resp = httpx.post(ARK_IMAGE_URL, json=payload, headers=headers, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    images = data.get("data", [])
    if not images or not images[0].get("url"):
        raise ValueError(f"No image URL in Volcengine response: {data}")
    return images[0]["url"]


def generate_image(
    prompt: str,
    seed: int | None = None,
    model: str | None = None,
    size: str | None = None,
) -> str:
    """Generate image, returns image URL."""
    model = model or _get_model()
    size = size or _get_size()
    provider = get_provider(model)
    if provider == "dashscope":
        return _ds_generate(prompt, model, size, seed)
    if provider == "volcengine":
        return _ark_generate(prompt, model, size)
    return _sf_generate(prompt, model, size, seed)


def download_image(url: str, save_path: str) -> str:
    """Download image from URL and save to local path."""
    resp = httpx.get(url, timeout=60)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "").lower()
    if not content_type.startswith("image/"):
        raise ValueError(
            f"Downloaded content is not an image: {content_type or 'unknown'}"
        )
    if not resp.content:
        raise ValueError("Downloaded image is empty")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(resp.content)
    return save_path


def generate_and_save(
    prompt: str,
    filename: str,
    seed: int | None = None,
    output_dir: str | None = None,
    model: str | None = None,
    size: str | None = None,
    download_retries: int = 2,
    sleep_fn=time.sleep,
) -> str:
    """Generate an image and save it locally. Returns the local file path."""
    image_url = generate_image(prompt, seed, model=model, size=size)
    save_path = os.path.join(output_dir or OUTPUT_DIR, filename)
    for attempt in range(download_retries + 1):
        try:
            return download_image(image_url, save_path)
        except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as error:
            if attempt == download_retries:
                raise ImageDownloadError(
                    f"图片已生成但下载失败：{error}（尝试 {attempt + 1}/{download_retries + 1} 次）"
                ) from error
            sleep_fn(2 ** attempt)
