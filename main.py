import gradio as gr
import json
import os
import time

from config import (
    SILICONFLOW_API_KEY,
    IMAGE_MODEL,
    IMAGE_SIZE,
    LLM_MODEL,
    OUTPUT_DIR,
    ALL_MODELS,
    HF_MODELS,
    SF_MODELS,
    get_model_config,
    validate_config,
)
from story_parser import parse_story
from image_generator import generate_and_save, _is_hf_model


def generate(story_text, image_model, image_size, progress=gr.Progress()):
    if not story_text.strip():
        yield "请输入童话故事文本", None
        return

    # Check API credentials
    if _is_hf_model(image_model):
        if not os.getenv("HF_TOKEN"):
            yield "错误：未配置 HF_TOKEN，请在 .env 文件中设置\n获取地址：https://huggingface.co/settings/tokens", None
            return
    else:
        if not SILICONFLOW_API_KEY:
            yield "错误：未配置 SILICONFLOW_API_KEY，请在 .env 文件中设置", None
            return

    # Override model settings
    import config
    config.IMAGE_MODEL = image_model
    config.IMAGE_SIZE = image_size

    model_config = get_model_config()
    if image_size not in model_config["image_sizes"]:
        yield f"错误：{image_model} 不支持尺寸 {image_size}，请选择: {model_config['image_sizes']}", None
        return

    # Step 1: Parse story
    progress(0, desc="正在分析故事、生成场景描述...")
    try:
        result = parse_story(story_text)
    except Exception as e:
        yield f"故事分析失败：{e}", None
        return

    title = result.get("title", "未命名")
    scenes = result["scenes"]

    # Build scene info text
    provider = "HuggingFace (免费)" if _is_hf_model(image_model) else "SiliconFlow"
    info_lines = [f"标题：{title}"]
    info_lines.append(f"场景数量：{len(scenes)}")
    info_lines.append(f"图像模型：{image_model} [{provider}]")
    info_lines.append("")
    for s in scenes:
        info_lines.append(f"【场景 {s['scene_number']}】{s['story_text']}")
        info_lines.append(f"  Prompt: {s['prompt']}")
        info_lines.append("")

    scene_info = "\n".join(info_lines)

    # Save prompts
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "prompts.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Step 2: Generate images one by one
    generated_images = []
    failed = []

    for i, scene in enumerate(scenes):
        num = scene["scene_number"]
        prompt = scene["prompt"]
        progress((i + 1) / len(scenes), desc=f"正在生成场景 {num}/{len(scenes)}...")

        try:
            path = generate_and_save(prompt, f"scene_{num:02d}.png")
            generated_images.append(path)
        except Exception as e:
            failed.append((num, str(e)))

        if num < len(scenes):
            time.sleep(1)

    # Final result
    result_lines = [f"生成完成！成功 {len(generated_images)}/{len(scenes)} 张"]
    result_lines.append(f"保存目录：{os.path.abspath(OUTPUT_DIR)}")
    if failed:
        result_lines.append("")
        result_lines.append("失败场景：")
        for num, err in failed:
            result_lines.append(f"  场景 {num}: {err}")
    result_text = "\n".join(result_lines)

    yield scene_info + "\n---\n" + result_text, generated_images


def on_model_change(model_name):
    sizes = ALL_MODELS.get(model_name, ALL_MODELS["black-forest-labs/FLUX.1-schnell"])["image_sizes"]
    return gr.Dropdown(choices=sizes, value=sizes[0])


def build_ui():
    with gr.Blocks(title="童话插图生成器") as app:
        gr.Markdown("# 童话插图生成器")
        gr.Markdown("输入童话故事文本，自动拆分场景并生成对应插图")

        with gr.Row():
            # Left: input
            with gr.Column(scale=1):
                story_input = gr.Textbox(
                    label="童话故事",
                    placeholder="在此输入童话故事文本...",
                    lines=15,
                )
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        label="图像模型",
                        choices=list(ALL_MODELS.keys()),
                        value=IMAGE_MODEL,
                    )
                    size_dropdown = gr.Dropdown(
                        label="图片尺寸",
                        choices=ALL_MODELS[IMAGE_MODEL]["image_sizes"],
                        value=IMAGE_SIZE,
                    )
                model_dropdown.change(
                    fn=on_model_change,
                    inputs=model_dropdown,
                    outputs=size_dropdown,
                )
                generate_btn = gr.Button("生成插图", variant="primary", size="lg")

            # Right: output
            with gr.Column(scale=1):
                result_text = gr.Textbox(label="生成结果", lines=20, interactive=False)
                gallery = gr.Gallery(
                    label="生成图片",
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
            "### 模型说明\n"
            "| 模型 | 渠道 | 费用 | 效果 |\n"
            "|------|------|------|------|\n"
            "| FLUX.1-schnell | HuggingFace | **免费** | 最好 |\n"
            "| SD3-medium | HuggingFace | **免费** | 中等 |\n"
            "| Kwai-Kolors/Kolors | SiliconFlow | **免费** | 一般 |\n"
            "| Z-Image-Turbo | SiliconFlow | ¥0.10/张 | 较好 |\n\n"
            "使用 HuggingFace 模型需配置 `HF_TOKEN`，获取：https://huggingface.co/settings/tokens"
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
