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
    "logo, watermark, speech bubble, malformed hands, extra fingers, duplicate characters, "
    "scary, horror, gore, violent, nsfw, photorealistic, photographic, "
    "blurry, low quality, distorted faces, deformed"
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
    if resp.status_code >= 400:
        try:
            err_msg = resp.json().get("message", "") or resp.text[:300]
            raise ValueError(f"硅基流动错误（{resp.status_code}）：{err_msg}")
        except (ValueError, KeyError):
            resp.raise_for_status()
    data = resp.json()

    images = data.get("images", [])
    if not images or not images[0].get("url"):
        raise ValueError("硅基流动未返回图片地址")
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
    if resp.status_code >= 400:
        try:
            err_msg = resp.json().get("message", "") or resp.text[:300]
            raise ValueError(f"阿里云百炼错误（{resp.status_code}）：{err_msg}")
        except (ValueError, KeyError):
            resp.raise_for_status()
    data = resp.json()

    task_id = data.get("output", {}).get("task_id")
    if not task_id:
        # DashScope 可能返回 HTTP 200 但 body 里含错误码
        err_code = data.get("code", "")
        err_msg = data.get("message", "") or str(data)[:300]
        prefix = f"阿里云百炼错误（{err_code}）：" if err_code else "阿里云百炼提交失败："
        raise ValueError(f"{prefix}{err_msg}")

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
            raise ValueError(f"阿里云百炼生成成功但未返回图片地址")
        if status == "FAILED":
            err_msg = data.get("output", {}).get("message", "") or str(data)[:300]
            raise ValueError(f"阿里云百炼生成失败：{err_msg}")
        # PENDING / RUNNING -> continue polling

    raise TimeoutError("阿里云百炼生成超时（等待超过 5 分钟）")


def _ark_generate(
    prompt: str, model: str, size: str, seed: int | None = None
) -> str:
    """火山方舟 Seedream 图片生成，返回限时图片 URL。"""
    headers = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "size": size,
        "response_format": "url",
        "watermark": False,
        "sequential_image_generation": "disabled",
        "optimize_prompt_options": {"mode": "standard"},
    }
    if seed is not None:
        payload["seed"] = seed
    # output_format 由 config 统一管理，避免在代码里硬编码模型名
    output_format = get_model_config(model).get("output_format")
    if output_format:
        payload["output_format"] = output_format
    resp = httpx.post(ARK_IMAGE_URL, json=payload, headers=headers, timeout=180)
    if resp.status_code >= 400:
        # 解析火山方舟的错误信息，展示给用户而非只显示 HTTP 状态码
        try:
            err_data = resp.json()
            err_msg = err_data.get("error", {}).get("message", "") or resp.text[:300]
            raise ValueError(f"火山方舟错误（{resp.status_code}）：{err_msg}")
        except (ValueError, KeyError):
            resp.raise_for_status()
    data = resp.json()
    images = data.get("data", [])
    if not images or not images[0].get("url"):
        raise ValueError("火山方舟未返回图片地址")
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
        return _ark_generate(prompt, model, size, seed)
    return _sf_generate(prompt, model, size, seed)


# content-type → 扩展名映射；优先用真实响应类型，保证文件内容与扩展名一致
_CONTENT_TYPE_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def detect_image_extension(content_type: str, content: bytes) -> str:
    """根据 content-type 或字节魔数推断图片扩展名，保证文件内容与扩展名一致。"""
    ct = (content_type or "").lower().split(";")[0].strip()
    if ct in _CONTENT_TYPE_EXT:
        return _CONTENT_TYPE_EXT[ct]
    # 部分 CDN 返回 octet-stream，按字节魔数兜底
    if content[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if content[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def download_image(url: str, save_path: str) -> str:
    """Download image from URL and save to local path."""
    resp = httpx.get(url, timeout=60)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    if not resp.content:
        raise ValueError("Downloaded image is empty")

    # 拒绝非图片内容：content-type 明确不是图片、且字节也不含图片魔数
    ct_lower = (content_type or "").lower().split(";")[0].strip()
    is_image_ct = ct_lower.startswith("image/") or ct_lower in (
        "application/octet-stream",
        "binary/octet-stream",
    )
    has_magic = (
        resp.content[:8] == b"\x89PNG\r\n\x1a\n"
        or resp.content[:3] == b"\xff\xd8\xff"
        or (resp.content[:4] == b"RIFF" and resp.content[8:12] == b"WEBP")
    )
    if not is_image_ct and not has_magic:
        raise ValueError(
            f"Downloaded content is not an image: {content_type or 'unknown'}"
        )

    # 写入真实扩展名，保证文件内容与扩展名一致
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    ext = detect_image_extension(content_type, resp.content)
    base = os.path.splitext(save_path)[0]
    final_path = base + ext
    with open(final_path, "wb") as f:
        f.write(resp.content)
    return final_path


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
