import os
import time
import httpx
from config import (
    SILICONFLOW_API_KEY,
    DASHSCOPE_API_KEY,
    BASE_URL,
    get_provider,
    get_model_config,
)

SF_API_URL = f"{BASE_URL}/images/generations"
DS_ASYNC_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
DS_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks"

MAX_RETRIES = 5
RETRY_DELAYS = [5, 10, 20, 40, 60]


def _sf_generate(prompt: str, model: str, size: str, seed: int | None = None) -> str:
    """SiliconFlow image generation with retry."""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    model_config = get_model_config()
    payload = {
        "model": model,
        "prompt": prompt,
        "image_size": size,
        "num_inference_steps": model_config["num_inference_steps"],
    }
    if model_config.get("guidance_scale") is not None:
        payload["guidance_scale"] = model_config["guidance_scale"]
    if seed is not None:
        payload["seed"] = seed

    for attempt in range(MAX_RETRIES):
        resp = httpx.post(SF_API_URL, json=payload, headers=headers, timeout=120)

        if resp.status_code == 429:
            wait = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else 60
            time.sleep(wait)
            continue

        if resp.status_code == 403:
            data = resp.json()
            code = data.get("code", "")
            msg = data.get("message", "")
            if "insufficient" in msg.lower() or code == 30001:
                raise RuntimeError(
                    "SiliconFlow balance insufficient! "
                    "Top up at https://cloud.siliconflow.cn/account/balance"
                )
            raise RuntimeError(f"SiliconFlow 403: {msg}")

        resp.raise_for_status()
        data = resp.json()
        images = data.get("images", [])
        if not images or not images[0].get("url"):
            raise ValueError(f"No image URL in response: {data}")
        return images[0]["url"]

    raise RuntimeError(f"Rate limited after {MAX_RETRIES} retries")


def _ds_generate(prompt: str, model: str, size: str, seed: int | None = None) -> str:
    """DashScope async image generation."""
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    ds_size = size.replace("x", "*")
    payload: dict = {
        "model": model,
        "input": {"prompt": prompt},
        "parameters": {
            "size": ds_size,
            "n": 1,
        },
    }
    if seed is not None:
        payload["parameters"]["seed"] = seed

    resp = httpx.post(DS_ASYNC_URL, json=payload, headers=headers, timeout=60)
    if resp.status_code == 403:
        raise RuntimeError("DashScope 403 - check API key at https://bailian.console.aliyun.com/")
    resp.raise_for_status()
    data = resp.json()

    task_id = data.get("output", {}).get("task_id")
    if not task_id:
        raise ValueError(f"DashScope submit failed: {data}")

    poll_headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
    for _ in range(60):
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

    raise TimeoutError("DashScope task timed out")


def generate_image(prompt: str, model: str, size: str, seed: int | None = None) -> str:
    """Generate image, returns image URL."""
    if get_provider(model) == "dashscope":
        return _ds_generate(prompt, model, size, seed)
    return _sf_generate(prompt, model, size, seed)


def download_image(url: str, save_path: str) -> str:
    """Download image from URL and save to local path."""
    resp = httpx.get(url, timeout=60)
    resp.raise_for_status()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(resp.content)
    return save_path


def generate_and_save(prompt: str, save_path: str, model: str, size: str, seed: int | None = None) -> str:
    """Generate an image and save it to the given full path. Returns the local file path."""
    image_url = generate_image(prompt, model, size, seed)
    download_image(image_url, save_path)
    return save_path
