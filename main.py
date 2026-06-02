import gradio as gr
import json
import os
import time

from config import (
    SILICONFLOW_API_KEY,
    DASHSCOPE_API_KEY,
    IMAGE_MODEL,
    IMAGE_SIZE,
    LLM_MODEL,
    OUTPUT_DIR,
    ALL_MODELS,
    get_provider,
    get_model_config,
    validate_config,
)
from story_parser import parse_story
from image_generator import generate_and_save

# Build display labels for model dropdown: modelName | provider | free/paid | effect
def _model_label(name, cfg):
    provider = cfg.get("provider", "siliconflow")
    price = cfg.get("price", "")
    if provider == "dashscope":
        who = "Ali"
        if "qwen" in name:
            effect = "Best"
        else:
            effect = "Good"
    else:
        who = "SiliconFlow"
        if "Kolors" in name:
            effect = "OK"
        elif "Turbo" in name:
            effect = "Good"
        else:
            effect = "Good"
    free_tag = "FREE" if "免费" in price or "free" in price.lower() else price
    return f"{name}  |  {who}  |  {free_tag}  |  {effect}"

MODEL_CHOICES = [_model_label(k, v) for k, v in ALL_MODELS.items()]

def _name_from_label(label):
    return label.split("|")[0].strip()


def generate(story_text, image_model_label, image_size, progress=gr.Progress()):
    if not story_text.strip():
        yield "Please enter a fairy tale", None
        return

    image_model = _name_from_label(image_model_label)
    provider = get_provider(image_model)

    if provider == "dashscope" and not DASHSCOPE_API_KEY:
        yield "Error: DASHSCOPE_API_KEY required for DashScope models\nGet: https://bailian.console.aliyun.com/", None
        return
    if provider == "siliconflow" and not SILICONFLOW_API_KEY:
        yield "Error: SILICONFLOW_API_KEY required for SiliconFlow models\nGet: https://cloud.siliconflow.cn/account/ak", None
        return

    import config
    config.IMAGE_MODEL = image_model
    config.IMAGE_SIZE = image_size

    # Step 1: Parse story
    progress(0, desc="Analyzing story, splitting scenes...")
    try:
        result = parse_story(story_text)
    except Exception as e:
        yield f"Story analysis failed: {e}", None
        return

    title = result.get("title", "Untitled")
    scenes = result["scenes"]

    # Story subfolder
    import re
    safe_title = re.sub(r'[\\/:*?"<>|]', '_', title.strip())
    story_output_dir = os.path.join(OUTPUT_DIR, safe_title)
    os.makedirs(story_output_dir, exist_ok=True)
    with open(os.path.join(story_output_dir, "prompts.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    config.OUTPUT_DIR = story_output_dir

    # Build info
    info_lines = [f"Story: {title}"]
    info_lines.append(f"Scenes: {len(scenes)}")
    info_lines.append(f"Model: {image_model}")
    info_lines.append("")
    for s in scenes:
        info_lines.append(f"[{s['scene_number']}] {s['story_text']}")
        info_lines.append("")
    scene_info = "\n".join(info_lines)

    # Step 2: Generate images — directly pass full path, no relying on global OUTPUT_DIR
    cfg = get_model_config()
    generated_images = []
    failed = []

    for i, scene in enumerate(scenes):
        num = scene["scene_number"]
        prompt = scene["prompt"]
        filename = f"scene_{num:02d}.png"
        full_path = os.path.join(story_output_dir, filename)
        progress((i + 1) / len(scenes), desc=f"Generating scene {num}/{len(scenes)}...")

        try:
            path = generate_and_save(
                prompt, full_path,
                model=image_model, size=image_size
            )
            generated_images.append(path)
        except RuntimeError as e:
            failed.append((num, str(e)))
            break
        except Exception as e:
            err = str(e)
            if "429" in err:
                time.sleep(30)
                try:
                    path = generate_and_save(
                        prompt, full_path,
                        model=image_model, size=image_size
                    )
                    generated_images.append(path)
                    continue
                except Exception as e2:
                    failed.append((num, f"Rate limited: {e2}"))
            else:
                failed.append((num, err))

        if num < len(scenes):
            time.sleep(3)

    result_lines = [f"Done! {len(generated_images)}/{len(scenes)} images generated"]
    result_lines.append(f"Saved to: {os.path.abspath(story_output_dir)}")
    if failed:
        result_lines.append("")
        result_lines.append("Failed scenes:")
        for num, err in failed:
            result_lines.append(f"  Scene {num}: {err}")
    result_text = "\n".join(result_lines)

    yield scene_info + "\n---\n" + result_text, generated_images


def on_model_change(model_label):
    name = _name_from_label(model_label)
    cfg = ALL_MODELS.get(name, ALL_MODELS["Kwai-Kolors/Kolors"])
    sizes = cfg.get("image_sizes", ["1024x1024"])
    display_sizes = [s.replace("*", "x") for s in sizes]
    return gr.Dropdown(choices=display_sizes, value=display_sizes[0])


def build_ui():
    default_choice = _model_label(IMAGE_MODEL, ALL_MODELS[IMAGE_MODEL])

    with gr.Blocks(title="Fairy Tale Illustrator") as app:
        gr.Markdown("# Fairy Tale Illustrator")
        gr.Markdown("Enter a fairy tale, auto-split scenes and generate illustrations")

        with gr.Row():
            with gr.Column(scale=1):
                story_input = gr.Textbox(
                    label="Story",
                    placeholder="Paste your fairy tale here...",
                    lines=15,
                )
                model_dropdown = gr.Dropdown(
                    label="Image Model",
                    choices=MODEL_CHOICES,
                    value=default_choice,
                )
                size_dropdown = gr.Dropdown(
                    label="Size",
                    choices=[s.replace("*", "x") for s in ALL_MODELS[IMAGE_MODEL].get("image_sizes", ["1024x1024"])],
                    value=IMAGE_SIZE,
                )
                model_dropdown.change(
                    fn=on_model_change,
                    inputs=model_dropdown,
                    outputs=size_dropdown,
                )
                generate_btn = gr.Button("Generate Illustrations", variant="primary", size="lg")

            with gr.Column(scale=1):
                result_text = gr.Textbox(label="Result", lines=20, interactive=False)
                gallery = gr.Gallery(
                    label="Illustrations",
                    columns=2,
                    height="auto",
                    object_fit="contain",
                )

        generate_btn.click(
            fn=generate,
            inputs=[story_input, model_dropdown, size_dropdown],
            outputs=[result_text, gallery],
        )

        gr.Markdown("---")
        gr.Markdown(
            "### Model Info\n"
            "Format: Model | Provider | Price | Quality\n"
            "- **FREE** = always free\n"
            "- **New user free credits** = free until credits run out, then paid\n"
            "- **Price/pc** = always paid\n\n"
            "### Tips\n"
            "1. Use DashScope (Ali) models first - best quality, free credits for new users\n"
            "2. When credits run out, switch to Kwai-Kolors/Kolors - always free\n"
            "3. If you get 403 error on SiliconFlow, top up balance at https://cloud.siliconflow.cn/account/balance"
        )

    return app


def main():
    try:
        validate_config()
    except SystemExit as e:
        print(str(e))

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
