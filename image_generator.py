import os
import httpx
from config import SILICONFLOW_API_KEY, BASE_URL, IMAGE_MODEL, IMAGE_SIZE, OUTPUT_DIR, get_model_config

API_URL = f"{BASE_URL}/images/generations"


def generate_image(prompt: str, seed: int | None = None) -> str:
    """Call SiliconFlow image generation API, return the image URL."""
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

    resp = httpx.post(API_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    images = data.get("images", [])
    if not images or not images[0].get("url"):
        raise ValueError(f"No image URL in response: {data}")

    return images[0]["url"]


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
