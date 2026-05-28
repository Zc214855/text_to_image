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
    LLM_PROVIDER,
    LLM_BASE_URL,
    OUTPUT_DIR,
    ALL_MODELS,
    get_provider,
    get_model_config,
    validate_config,
)
from story_parser import parse_story
from image_generator import generate_and_save


def generate(story_text, image_model, image_size, progress=gr.Progress()):
    if not story_text.strip():
        yield "请输入童话故事文本", None
        return

    # Check credentials for image generation
    provider = get_provider(image_model)
    if provider == "dashscope" and not DASHSCOPE_API_KEY:
        yield "错误：使用阿里云百炼模型需配置 DASHSCOPE_API_KEY\n获取：https://bailian.console.aliyun.com/", None
        return
    if provider == "siliconflow" and not SILICONFLOW_API_KEY:
        yield "错误：使用硅基流动模型需配置 SILICONFLOW_API_KEY\n获取：https://cloud.siliconflow.cn/account/ak", None
        return

    # Override model settings
    import config
    config.IMAGE_MODEL = image_model
    config.IMAGE_SIZE = image_size

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
    provider_label = "阿里云百炼" if provider == "dashscope" else "硅基流动"
    llm_label = "FreeLLMAPI(免费)" if LLM_PROVIDER == "freellmapi" else "SiliconFlow"
    info_lines = [f"标题：{title}"]
    info_lines.append(f"场景数量：{len(scenes)}")
    info_lines.append(f"LLM：{LLM_MODEL} [{llm_label}]")
    info_lines.append(f"图像模型：{image_model} [{provider_label}]")
    info_lines.append("")
    for s in scenes:
        info_lines.append(f"【场景 {s['scene_number']}】{s['story_text']}")
        info_lines.append(f"  Prompt: {s['prompt'][:100]}...")
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
    cfg = ALL_MODELS.get(model_name, ALL_MODELS["Kwai-Kolors/Kolors"])
    sizes = cfg.get("image_sizes", ["1024x1024"])
    display_sizes = [s.replace("*", "x") for s in sizes]
    return gr.Dropdown(choices=display_sizes, value=display_sizes[0])


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
                        choices=[s.replace("*", "x") for s in ALL_MODELS[IMAGE_MODEL].get("image_sizes", ["1024x1024"])],
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
            "| **wanx2.1-t2i-turbo** | 阿里云百炼 | 新用户免费送额度 | 较好 |\n"
            "| **wanx2.1-t2i-plus** | 阿里云百炼 | 新用户免费送额度 | 较好 |\n"
            "| **qwen-image-plus** | 阿里云百炼 | 新用户免费送额度 | 最好 |\n"
            "| Kwai-Kolors/Kolors | 硅基流动 | 免费 | 一般 |\n"
            "| Z-Image-Turbo | 硅基流动 | ¥0.10/张 | 较好 |\n"
            "| Z-Image | 硅基流动 | ¥0.30/张 | 较好 |\n"
            "| Qwen-Image | 硅基流动 | ¥0.30/张 | 较好 |\n"
            "| ERNIE-Image-Turbo | 硅基流动 | ¥0.11/张 | 中等 |\n\n"
            "### 省钱技巧\n"
            "- LLM调用可通过 [FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) 完全免费，配置方式：\n"
            "  1. `git clone` 并启动 FreeLLMAPI（`npm install && npm run dev`）\n"
            "  2. 在 `.env` 中设置:\\n```\\nLLM_PROVIDER=freellmapi\\n"
            "LLM_BASE_URL=http://localhost:3001/v1\\nLLM_API_KEY=freellmapi-你的key\\n"
            "LLM_MODEL=auto\\n```\\n\n"
            "- 图片先用阿里云百炼免费额度，用完切硅基流动 Kolors（持久免费）"
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
