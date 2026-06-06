import json
import os
import time
import httpx

import config
import gradio as gr

from config import (
    SILICONFLOW_API_KEY,
    DASHSCOPE_API_KEY,
    ARK_API_KEY,
    IMAGE_MODEL,
    IMAGE_SIZE,
    OUTPUT_DIR,
    ALL_MODELS,
    LLM_PROVIDERS,
    PROVIDER_LABELS,
    get_llm_provider_config,
    get_provider,
    validate_config,
)
from story_parser import parse_story
from image_generator import generate_and_save
from output_manager import create_story_output_dir


def check_llm_status():
    """检测 LLM 服务是否可用，返回状态文本"""
    if config.LLM_PROVIDER == "freellmapi":
        try:
            resp = httpx.post(
                f"{config.LLM_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
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
    provider_keys = {
        "siliconflow": SILICONFLOW_API_KEY,
        "zhipu": config.LLM_API_KEY,
        "volcengine": config.LLM_API_KEY,
    }
    if config.LLM_PROVIDER not in provider_keys:
        return f"未知 LLM 供应商：{config.LLM_PROVIDER}"
    status = "已配置 API Key" if provider_keys[config.LLM_PROVIDER] else "未配置 API Key"
    label = LLM_PROVIDERS[config.LLM_PROVIDER]["label"]
    return f"{label} - {status}"


def generate(story_text, llm_provider, image_model, image_size, progress=gr.Progress()):
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
    if provider == "volcengine" and not ARK_API_KEY:
        yield "错误：使用火山方舟模型需配置 ARK_API_KEY", None
        return

    # 切换当前任务使用的分镜 LLM 和图片模型。
    config.set_llm_provider(llm_provider)
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
    story_output_dir = create_story_output_dir(title)

    # Build scene info text
    provider_labels = {
        "dashscope": "阿里云百炼",
        "siliconflow": "硅基流动",
        "volcengine": "火山方舟",
    }
    provider_label = provider_labels.get(provider, provider)
    llm_label = LLM_PROVIDERS[config.LLM_PROVIDER]["label"]
    info_lines = [f"标题：{title}"]
    info_lines.append(f"场景数量：{len(scenes)}")
    info_lines.append(f"LLM：{config.LLM_MODEL} [{llm_label}]")
    info_lines.append(f"LLM状态：{llm_status}")
    info_lines.append(f"图像模型：{image_model} [{provider_label}]")
    info_lines.append("")
    for s in scenes:
        info_lines.append(f"【场景 {s['scene_number']}】{s['story_text']}")
        info_lines.append(f"  Prompt: {s['prompt'][:100]}...")
        info_lines.append("")

    scene_info = "\n".join(info_lines)

    # 提示词和图片统一保存到以故事名命名的独立目录。
    with open(
        os.path.join(story_output_dir, "prompts.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Step 2: Generate images one by one
    generated_images = []
    failed = []

    for i, scene in enumerate(scenes):
        num = scene["scene_number"]
        prompt = scene["prompt"]
        progress((i + 1) / len(scenes), desc=f"正在生成场景 {num}/{len(scenes)}...")

        try:
            path = generate_and_save(
                prompt,
                f"scene_{num:02d}.png",
                output_dir=story_output_dir,
            )
            generated_images.append(path)
        except Exception as e:
            failed.append((num, str(e)))

        if num < len(scenes):
            time.sleep(1)

    # Final result
    result_lines = [f"生成完成！成功 {len(generated_images)}/{len(scenes)} 张"]
    result_lines.append(f"保存目录：{os.path.abspath(story_output_dir)}")
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
    return (
        gr.Dropdown(choices=display_sizes, value=display_sizes[0]),
        build_model_detail(model_name),
    )


def on_llm_change(provider):
    config.set_llm_provider(provider)
    return check_llm_status(), build_llm_detail(provider)


def build_llm_detail(provider):
    cfg = get_llm_provider_config(provider)
    return (
        f"**当前分镜 LLM：{cfg['label']}**  \n"
        f"模型：`{cfg['model']}`  \n"
        f"用途：{cfg['summary']}  \n"
        "说明：它只负责分析故事和编写提示词，不直接生成图片。"
    )


def build_model_detail(model_name):
    """生成当前图像模型说明，内容与模型配置保持同步。"""
    cfg = ALL_MODELS.get(model_name, ALL_MODELS["Kwai-Kolors/Kolors"])
    provider = PROVIDER_LABELS.get(cfg["provider"], cfg["provider"])
    sizes = "、".join(size.replace("*", "×").replace("x", "×") for size in cfg["image_sizes"])
    return (
        f"**当前图像模型：{cfg['label']}**  \n"
        f"模型 ID：`{model_name}`  \n"
        f"渠道：{provider}　费用：{cfg['price']}  \n"
        f"用途：{cfg['summary']}  \n"
        f"可选尺寸：{sizes}"
    )


def build_model_table():
    """生成全部已接入图像模型表，避免界面文案遗漏新增模型。"""
    rows = [
        "| 界面名称 | 模型 ID | 渠道 | 费用 | 定位 |",
        "|---|---|---|---|---|",
    ]
    for model_id, cfg in ALL_MODELS.items():
        provider = PROVIDER_LABELS.get(cfg["provider"], cfg["provider"])
        rows.append(
            f"| {cfg['label']} | `{model_id}` | {provider} | "
            f"{cfg['price']} | {cfg['summary']} |"
        )
    return "\n".join(rows)


def build_usage_guide():
    llm_label = {
        "freellmapi": "FreeLLMAPI 自动路由",
        "siliconflow": "硅基流动",
        "zhipu": "智谱 GLM，经腾讯云 LKEAP Token Plan",
        "volcengine": "火山方舟",
    }.get(config.LLM_PROVIDER, config.LLM_PROVIDER)
    return (
        "### 当前实际使用链路\n"
        f"1. **故事理解与分镜**：`{config.LLM_MODEL}`（{llm_label}）。"
        "它负责读取故事、固定角色外观、拆分镜头并编写图像提示词，不直接画图。\n"
        f"2. **插图生成**：`{IMAGE_MODEL}`（"
        f"{PROVIDER_LABELS.get(get_provider(IMAGE_MODEL), get_provider(IMAGE_MODEL))}）。"
        "它接收每个镜头的提示词并生成最终图片。\n"
        "3. **模型下拉框只切换插图模型**，不会改变上面的故事分析 LLM。\n\n"
        "### 图像模型说明\n"
        f"{build_model_table()}\n\n"
        "### 选择建议\n"
        "- **正式生成童话插图**：Seedream 5.0 Lite，当前默认和推荐选项。\n"
        "- **免费测试完整流程**：Kolors。\n"
        "- **低成本批量草图**：Z-Image Turbo 或万相 2.1 Turbo。\n"
        "- **复杂提示词与高分辨率构图**：Qwen Image、Qwen Image Plus。\n"
        "- **更稳定的跨镜头角色一致性**：仅切换文生图模型不够，后续需要接入角色参考图或组图生成。\n\n"
        "### 配置说明\n"
        "- `LLM_PROVIDER` / `LLM_MODEL`：控制故事分析和分镜模型。\n"
        "- 页面上的“分镜 LLM”可以临时切换当前进程使用的文本模型；重启后恢复 `.env` 默认值。\n"
        "- `IMAGE_MODEL` / `IMAGE_SIZE`：控制启动时默认图像模型和尺寸。\n"
        "- 页面切换图像模型后，尺寸列表会自动更新。\n"
        "- 费用和模型可用性会由平台调整，实际以对应平台控制台为准。"
    )


def build_ui():
    llm_status = check_llm_status()

    with gr.Blocks(title="童话插图生成器") as app:
        gr.Markdown("# 童话插图生成器")
        gr.Markdown("输入童话故事文本，自动拆分场景并生成对应插图")

        with gr.Row():
            llm_dropdown = gr.Dropdown(
                label="分镜 LLM（分析故事，不负责画图）",
                choices=[
                    (cfg["label"], provider)
                    for provider, cfg in LLM_PROVIDERS.items()
                ],
                value=config.LLM_PROVIDER,
            )
            llm_status_display = gr.Textbox(
                label="LLM 状态",
                value=llm_status,
                interactive=False,
            )
        llm_detail = gr.Markdown(build_llm_detail(config.LLM_PROVIDER))
        llm_dropdown.change(
            fn=on_llm_change,
            inputs=llm_dropdown,
            outputs=[llm_status_display, llm_detail],
        )

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
                        choices=[
                            (cfg["label"], model_id)
                            for model_id, cfg in ALL_MODELS.items()
                        ],
                        value=IMAGE_MODEL,
                    )
                    size_dropdown = gr.Dropdown(
                        label="图片尺寸",
                        choices=[s.replace("*", "x") for s in ALL_MODELS[IMAGE_MODEL].get("image_sizes", ["1024x1024"])],
                        value=IMAGE_SIZE,
                    )
                model_detail = gr.Markdown(build_model_detail(IMAGE_MODEL))
                model_dropdown.change(
                    fn=on_model_change,
                    inputs=model_dropdown,
                    outputs=[size_dropdown, model_detail],
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
            inputs=[story_input, llm_dropdown, model_dropdown, size_dropdown],
            outputs=[result_text, gallery],
        )

        gr.Markdown("---")
        gr.Markdown(build_usage_guide())

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
