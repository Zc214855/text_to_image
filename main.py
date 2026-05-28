import gradio as gr
import json
import os
import time
import httpx

from config import (
    SILICONFLOW_API_KEY,
    DASHSCOPE_API_KEY,
    IMAGE_MODEL,
    IMAGE_SIZE,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_BASE_URL,
    LLM_API_KEY,
    OUTPUT_DIR,
    ALL_MODELS,
    get_provider,
    get_model_config,
    validate_config,
)
from story_parser import parse_story
from image_generator import generate_and_save


def check_llm_status():
    """检测 LLM 服务是否可用，返回状态文本"""
    if LLM_PROVIDER == "freellmapi":
        try:
            resp = httpx.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                },
                timeout=10,
            )
            data = resp.json()
            if "choices" in data:
                routed = resp.headers.get("X-Routed-Via", "unknown")
                return f"FreeLLMAPI 连接正常 (路由: {routed})"
            else:
                msg = data.get("error", {}).get("message", str(data))
                return f"FreeLLMAPI 连接失败: {msg}"
        except Exception as e:
            return f"FreeLLMAPI 连接失败: {e}"
    else:
        if SILICONFLOW_API_KEY:
            return "SiliconFlow 直连 - 已配置 API Key"
        return "SiliconFlow 直连 - 未配置 API Key"


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
    llm_status = check_llm_status()
    progress(0, desc="正在分析故事、生成场景描述...")
    try:
        result = parse_story(story_text)
    except Exception as e:
        yield f"故事分析失败（{llm_status}）\n\n错误：{e}", None
        return

    title = result.get("title", "未命名")
    scenes = result["scenes"]

    # Build scene info text
    provider_label = "阿里云百炼" if provider == "dashscope" else "硅基流动"
    if LLM_PROVIDER == "freellmapi":
        llm_label = "FreeLLMAPI(免费)"
    else:
        llm_label = "SiliconFlow"
    info_lines = [f"标题：{title}"]
    info_lines.append(f"场景数量：{len(scenes)}")
    info_lines.append(f"LLM：{LLM_MODEL} [{llm_label}]")
    info_lines.append(f"LLM状态：{llm_status}")
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
    llm_status = check_llm_status()

    with gr.Blocks(title="童话插图生成器") as app:
        gr.Markdown("# 童话插图生成器")
        gr.Markdown("输入童话故事文本，自动拆分场景并生成对应插图")

        # LLM 状态栏
        llm_status_display = gr.Textbox(
            label="LLM 状态",
            value=llm_status,
            interactive=False,
        )

        # 刷新按钮
        refresh_btn = gr.Button("刷新 LLM 状态", size="sm")

        def refresh_llm_status():
            return check_llm_status()

        refresh_btn.click(fn=refresh_llm_status, outputs=llm_status_display)

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
            "### 界面说明\n"
            "- **LLM 状态栏**：顶部显示当前 LLM 连接状态，点击「刷新」可重新检测\n"
            "- **绿色=正常**：FreeLLMAPI 连接正常 或 SiliconFlow Key 已配置\n"
            "- **红色=异常**：连接失败或未配置，需要修复后才能拆分故事\n\n"
            "### 模型说明\n"
            "| 模型 | 渠道 | 费用 | 效果 |\n"
            "|------|------|------|------|\n"
            "| **wanx2.1-t2i-turbo** | 阿里云百炼 | 新用户免费送额度 | 较好 |\n"
            "| **wanx2.1-t2i-plus** | 阿里云百炼 | 新用户免费送额度 | 较好 |\n"
            "| **qwen-image-plus** | 阿里云百炼 | 新用户免费送额度 | 最好 |\n"
            "| Kwai-Kolors/Kolors | 硅基流动 | 免费 | 一般 |\n"
            "| Z-Image-Turbo | 硅基流动 | ¥0.10/张 | 较好 |\n\n"
            "### FreeLLMAPI 配置（让 LLM 调用完全免费）\n"
            "1. 去以下平台注册免费 API Key（只需邮箱）：\n"
            "   - Google AI Studio: https://aistudio.google.com/apikey → 创建 Gemini Key\n"
            "   - Groq: https://console.groq.com/keys → 创建 Key\n"
            "2. 启动 FreeLLMAPI：`cd f:/MyTool/freellmapi && npm run dev`\n"
            "3. 打开管理面板 http://localhost:3001 → Keys 页面 → 添加你的免费 Key\n"
            "4. 在本项目 `.env` 中设置：\n"
            "```\n"
            "LLM_PROVIDER=freellmapi\n"
            "LLM_BASE_URL=http://localhost:3001/v1\n"
            "LLM_API_KEY=freellmapi-你的key\n"
            "LLM_MODEL=auto\n"
            "```\n"
            "5. 回到这里点「刷新 LLM 状态」，确认显示连接正常"
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
