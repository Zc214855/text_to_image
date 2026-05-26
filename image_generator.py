import os
import httpx
from config import SILICONFLOW_API_KEY, BASE_URL, IMAGE_MODEL, IMAGE_SIZE, OUTPUT_DIR, get_model_config

SF_API_URL = f"{BASE_URL}/images/generations"

HF_API_URL = "https://router.huggingface.co/hf-inference/models/{model}"


def _is_hf_model(model: str) -> bool:
    """Models served by HuggingFace hf-inference (free)."""
    hf_models = {
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-3-medium-diffusers",
    }
    return model in hf_models


def generate_image_sf(prompt: str, seed: int | None = None) -> str:
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
    if model_config["guidance_scale"] is not None:
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


def generate_image_hf(prompt: str, seed: int | None = None) -> str:
    """HuggingFace Inference API, returns local path (API returns raw bytes)."""
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("HF_TOKEN not set. Get one at https://huggingface.co/settings/tokens")

    url = HF_API_URL.format(model=IMAGE_MODEL)
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload: dict = {"inputs": prompt}
    if seed is not None:
        payload["parameters"] = {"seed": seed}

    resp = httpx.post(url, json=payload, headers=headers, timeout=120)
    if resp.status_code == 503:
        raise ValueError("Model is loading, please wait ~30s and retry")
    resp.raise_for_status()

    # HF returns raw image bytes directly
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import tempfile
    tmp = os.path.join(OUTPUT_DIR, f"_hf_tmp_{seed or 0}.png")
    with open(tmp, "wb") as f:
        f.write(resp.content)
    return tmp


def generate_image(prompt: str, seed: int | None = None) -> str:
    """Generate image via SiliconFlow or HuggingFace. Returns URL or local path."""
    if _is_hf_model(IMAGE_MODEL):
        return generate_image_hf(prompt, seed)
    return generate_image_sf(prompt, seed)


def download_image(url: str, save_path: str) -> str:
    """Download image from URL and save to local path."""
    if os.path.isfile(url):
        # Already a local file (HF mode), just rename
        if url != save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            os.replace(url, save_path)
        return save_path

    resp = httpx.get(url, timeout=60)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(resp.content)
    return save_path


def generate_and_save(prompt: str, filename: str, seed: int | None = None) -> str:
    """Generate an image and save it locally. Returns the local file path."""
    result = generate_image(prompt, seed)
    save_path = os.path.join(OUTPUT_DIR, filename)
    download_image(result, save_path)
    return save_path
