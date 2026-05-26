import os
import time
import httpx
from config import (
    SILICONFLOW_API_KEY,
    DASHSCOPE_API_KEY,
    BASE_URL,
    IMAGE_MODEL,
    IMAGE_SIZE,
    OUTPUT_DIR,
    get_provider,
    get_model_config,
)

SF_API_URL = f"{BASE_URL}/images/generations"
DS_ASYNC_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
DS_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks"


def _sf_generate(prompt: str, seed: int | None = None) -> str:
    """SiliconFlow image generation, returns image URL."""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    model_config = get_model_config()
    payload = {
        "model": IMAGE_MODEL,
        "prompt": prompt,
        "image_size": IMAGE_SIZE,
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


def _ds_generate(prompt: str, seed: int | None = None) -> str:
    """DashScope async image generation, returns image URL."""
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    payload: dict = {
        "model": IMAGE_MODEL,
        "input": {"prompt": prompt},
        "parameters": {
            "size": IMAGE_SIZE,
            "n": 1,
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


def generate_image(prompt: str, seed: int | None = None) -> str:
    """Generate image, returns image URL."""
    if get_provider(IMAGE_MODEL) == "dashscope":
        return _ds_generate(prompt, seed)
    return _sf_generate(prompt, seed)


def download_image(url: str, save_path: str) -> str:
    """Download image from URL and save to local path."""
    resp = httpx.get(url, timeout=60)
    resp.raise_for_status()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(resp.content)
    return save_path


def generate_and_save(prompt: str, filename: str, seed: int | None = None) -> str:
    """Generate an image and save it locally. Returns the local file path."""
    image_url = generate_image(prompt, seed)
    save_path = os.path.join(OUTPUT_DIR, filename)
    download_image(image_url, save_path)
    return save_path
