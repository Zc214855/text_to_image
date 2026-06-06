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
from generation_tasks import (
    TaskAlreadyRunningError,
    collect_generated_images,
    find_latest_failed_task,
    load_prompts,
    load_task,
    save_task,
    task_execution_lock,
)

AUTO_RETRY_COUNT = 2


def validate_image_credentials(image_model):
    provider = get_provider(image_model)
    if provider == "dashscope" and not DASHSCOPE_API_KEY:
        return "错误：使用阿里云百炼模型需配置 DASHSCOPE_API_KEY"
    if provider == "siliconflow" and not SILICONFLOW_API_KEY:
        return "错误：使用硅基流动模型需配置 SILICONFLOW_API_KEY"
    if provider == "volcengine" and not ARK_API_KEY:
        return "错误：使用火山方舟模型需配置 ARK_API_KEY"
    return None


def is_retryable_image_error(error: Exception) -> bool:
    """只重试网络、限流和服务端临时故障，避免重复提交永久失败请求。"""
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status in {408, 409, 425, 429} or 500 <= status <= 599
    # POST 传输超时无法确定服务端是否已受理，自动重提可能重复计费。
    return False


def generate_scene_with_retry(
    prompt,
    filename,
    output_dir,
    image_model,
    image_size,
    max_retries=AUTO_RETRY_COUNT,
    sleep_fn=time.sleep,
):
    """生成单张图片；瞬时错误自动重试，返回图片路径和实际尝试次数。"""
    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            path = generate_and_save(
                prompt,
                filename,
                output_dir=output_dir,
                model=image_model,
                size=image_size,
            )
            return path, attempt
        except Exception as error:
            if attempt == total_attempts or not is_retryable_image_error(error):
                raise RuntimeError(
                    f"{error}（尝试 {attempt}/{total_attempts} 次）"
                ) from error
            sleep_fn(2 ** (attempt - 1))


def get_image_extension(image_model):
    return ".png"


def build_generation_summary(task, task_dir):
    failed = task.get("failed_scenes", [])
    succeeded = task.get("successful_scenes", [])
    lines = [
        f"图片生成完成：成功 {len(succeeded)}/{task['scene_count']} 张",
        f"保存目录：{os.path.abspath(task_dir)}",
    ]
    if failed:
        lines.extend(["", "失败场景（可点击“仅重试失败图片”）："])
        for item in failed:
            lines.append(f"  场景 {item['scene_number']}: {item['error']}")
    return "\n".join(lines)


def set_generation_buttons_enabled(enabled):
    """同步切换生成与重试按钮，防止长任务执行期间重复排队。"""
    return (
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
    )


def check_llm_status(provider=None):
    """检测 LLM 服务是否可用，返回状态文本"""
    provider = provider or config.LLM_PROVIDER
    llm_config = get_llm_provider_config(provider)
    if provider == "freellmapi":
        try:
            resp = httpx.post(
                f"{llm_config['base_url'].rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {llm_config['api_key']}"},
                json={
                    "model": llm_config["model"],
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
        "zhipu": llm_config["api_key"],
        "volcengine": llm_config["api_key"],
    }
    if provider not in provider_keys:
        return f"未知 LLM 供应商：{provider}"
    status = "已配置 API Key" if provider_keys[provider] else "未配置 API Key"
    label = LLM_PROVIDERS[provider]["label"]
    return f"{label} - {status}"


def generate(story_text, llm_provider, image_model, image_size, progress=gr.Progress()):
    if not story_text.strip():
        yield "请输入童话故事文本", None, None
        return

    credential_error = validate_image_credentials(image_model)
    if credential_error:
        yield credential_error, None, None
        return
    provider = get_provider(image_model)

    selected_llm = get_llm_provider_config(llm_provider)
    llm_client_config = (
        selected_llm["base_url"],
        selected_llm["api_key"],
        selected_llm["model"],
    )

    # Step 1: Parse story
    llm_status = check_llm_status(llm_provider)
    progress(0, desc="正在分析故事、生成场景描述...")
    try:
        result = parse_story(story_text, llm_client_config=llm_client_config)
    except Exception as e:
        yield f"故事分析失败（{llm_status}）\n\n错误：{e}", None, None
        return

    title = result.get("title", "未命名")
    scenes = result["scenes"]
    story_output_dir = os.path.abspath(create_story_output_dir(title))
    image_extension = get_image_extension(image_model)

    # Build scene info text
    provider_labels = {
        "dashscope": "阿里云百炼",
        "siliconflow": "硅基流动",
        "volcengine": "火山方舟",
    }
    provider_label = provider_labels.get(provider, provider)
    llm_label = LLM_PROVIDERS[llm_provider]["label"]
    info_lines = [f"标题：{title}"]
    info_lines.append(f"场景数量：{len(scenes)}")
    info_lines.append(f"LLM：{selected_llm['model']} [{llm_label}]")
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

    task = {
        "title": title,
        "image_model": image_model,
        "image_size": image_size,
        "image_extension": image_extension,
        "scene_count": len(scenes),
        "successful_scenes": [],
        "failed_scenes": [],
        "status": "running",
    }
    with task_execution_lock(story_output_dir):
        save_task(story_output_dir, task)
        for i, scene in enumerate(scenes):
            num = scene["scene_number"]
            prompt = scene["prompt"]
            progress((i + 1) / len(scenes), desc=f"正在生成场景 {num}/{len(scenes)}...")

            try:
                _, attempts = generate_scene_with_retry(
                    prompt,
                    f"scene_{num:02d}{image_extension}",
                    story_output_dir,
                    image_model,
                    image_size,
                )
                task["successful_scenes"].append(num)
                if attempts > 1:
                    task.setdefault("retried_scenes", {})[str(num)] = attempts
            except Exception as e:
                task["failed_scenes"].append(
                    {"scene_number": num, "error": str(e)}
                )
            save_task(story_output_dir, task)

            if num < len(scenes):
                time.sleep(1)

        task["status"] = "partial" if task["failed_scenes"] else "completed"
        save_task(story_output_dir, task)
    generated_images = collect_generated_images(
        story_output_dir,
        task["successful_scenes"],
        image_extension,
    )
    result_text = build_generation_summary(task, story_output_dir)

    yield scene_info + "\n---\n" + result_text, generated_images, story_output_dir


def retry_failed_images(task_dir, progress=gr.Progress()):
    """获取任务互斥锁后恢复失败图片，避免与原生成流程并行执行。"""
    task_dir = task_dir or find_latest_failed_task(OUTPUT_DIR)
    if not task_dir:
        yield "没有可重试的失败任务", None, None
        return
    try:
        with task_execution_lock(task_dir, blocking=False):
            yield from _retry_failed_images_locked(task_dir, progress)
    except TaskAlreadyRunningError as error:
        yield str(error), None, task_dir


def _retry_failed_images_locked(task_dir, progress):
    """读取已保存提示词，仅重试失败或缺失图片，不重新调用 LLM。"""
    try:
        task = load_task(task_dir)
        prompts = load_prompts(task_dir)
    except Exception as error:
        yield f"读取失败任务出错：{error}", None, task_dir
        return

    image_model = task["image_model"]
    credential_error = validate_image_credentials(image_model)
    if credential_error:
        yield credential_error, None, task_dir
        return

    image_size = task["image_size"]
    extension = task.get("image_extension", get_image_extension(image_model))
    scenes_by_number = {
        scene["scene_number"]: scene for scene in prompts["scenes"]
    }
    successful = set(task.get("successful_scenes", []))
    failed_numbers = {
        item["scene_number"] for item in task.get("failed_scenes", [])
    }
    missing_numbers = {
        number
        for number in scenes_by_number
        if not os.path.isfile(
            os.path.join(task_dir, f"scene_{number:02d}{extension}")
        )
    }
    retry_numbers = sorted(failed_numbers | missing_numbers)
    if not retry_numbers:
        task["status"] = "completed"
        task["failed_scenes"] = []
        save_task(task_dir, task)
        images = collect_generated_images(task_dir, list(scenes_by_number), extension)
        yield "任务没有失败或缺失图片", images, task_dir
        return

    successful.difference_update(retry_numbers)
    task["status"] = "retrying"
    task["failed_scenes"] = []
    save_task(task_dir, task)

    retry_failures = []
    for index, scene_number in enumerate(retry_numbers, start=1):
        scene = scenes_by_number.get(scene_number)
        if not scene:
            retry_failures.append(
                {"scene_number": scene_number, "error": "prompts.json 中缺少该场景"}
            )
            continue
        progress(
            index / len(retry_numbers),
            desc=f"仅重试图片 {index}/{len(retry_numbers)}（场景 {scene_number}）...",
        )
        try:
            _, attempts = generate_scene_with_retry(
                scene["prompt"],
                f"scene_{scene_number:02d}{extension}",
                task_dir,
                image_model,
                image_size,
            )
            successful.add(scene_number)
            task.setdefault("retried_scenes", {})[str(scene_number)] = attempts
        except Exception as error:
            retry_failures.append(
                {"scene_number": scene_number, "error": str(error)}
            )
        task["successful_scenes"] = sorted(successful)
        task["failed_scenes"] = retry_failures
        save_task(task_dir, task)

    task["status"] = "partial" if retry_failures else "completed"
    task["successful_scenes"] = sorted(successful)
    task["failed_scenes"] = retry_failures
    save_task(task_dir, task)
    images = collect_generated_images(task_dir, sorted(successful), extension)
    summary = "未调用 LLM，仅重试图片。\n" + build_generation_summary(task, task_dir)
    yield summary, images, task_dir


def on_model_change(model_name):
    cfg = ALL_MODELS.get(model_name, ALL_MODELS["Kwai-Kolors/Kolors"])
    sizes = cfg.get("image_sizes", ["1024x1024"])
    display_sizes = [s.replace("*", "x") for s in sizes]
    return (
        gr.Dropdown(choices=display_sizes, value=display_sizes[0]),
        build_model_detail(model_name),
    )


def on_llm_change(provider):
    return check_llm_status(provider), build_llm_detail(provider)


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
        "### 默认启动配置\n"
        f"1. **故事理解与分镜**：`{config.LLM_MODEL}`（{llm_label}）。"
        "它负责读取故事、固定角色外观、拆分镜头并编写图像提示词，不直接画图。\n"
        f"2. **插图生成**：`{IMAGE_MODEL}`（"
        f"{PROVIDER_LABELS.get(get_provider(IMAGE_MODEL), get_provider(IMAGE_MODEL))}）。"
        "它接收每个镜头的提示词并生成最终图片。\n"
        "3. **模型下拉框只切换插图模型**，不会改变上面的故事分析 LLM；"
        "页面当前实际选择以上方模型状态为准。\n\n"
        "### 图像模型说明\n"
        f"{build_model_table()}\n\n"
        "### 选择建议\n"
        "- **正式生成童话插图**：Seedream 5.0 Lite，当前默认和推荐选项。\n"
        "- **1K 低分辨率预览**：Seedream 4.0，先确认构图再切换 5.0 Lite 正式生成。\n"
        "- **免费测试完整流程**：Kolors。\n"
        "- **低成本批量草图**：Z-Image Turbo 或万相 2.1 Turbo。\n"
        "- **复杂提示词与高分辨率构图**：Qwen Image、Qwen Image Plus。\n"
        "- **更稳定的跨镜头角色一致性**：仅切换文生图模型不够，后续需要接入角色参考图或组图生成。\n\n"
        "### 配置说明\n"
        "- `LLM_PROVIDER` / `LLM_MODEL`：控制故事分析和分镜模型。\n"
        "- 页面上的“分镜 LLM”只切换当前任务使用的文本模型，不修改 `.env` 默认值。\n"
        "- `IMAGE_MODEL` / `IMAGE_SIZE`：控制启动时默认图像模型和尺寸。\n"
        "- 页面切换图像模型后，尺寸列表会自动更新。\n"
        "- 生图请求明确返回限流或 5xx 时自动重试 2 次；POST 传输超时不自动重提，避免重复计费。\n"
        "- 图片下载失败会对同一个结果 URL 重试，不会重新提交生图请求；最终失败后可仅重试图片，不重复调用 LLM。\n"
        "- 费用和模型可用性会由平台调整，实际以对应平台控制台为准。"
    )


def build_ui():
    llm_status = check_llm_status()

    with gr.Blocks(title="童话插图生成器") as app:
        current_task_dir = gr.State(value=None)
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
                retry_btn = gr.Button(
                    "仅重试失败图片（不调用 LLM）",
                    variant="secondary",
                    size="lg",
                )

            # Right: output
            with gr.Column(scale=1):
                result_text = gr.Textbox(label="生成结果", lines=20, interactive=False)
                gallery = gr.Gallery(
                    label="生成图片",
                    columns=2,
                    height="auto",
                    object_fit="contain",
                )

        generate_event = generate_btn.click(
            fn=lambda: set_generation_buttons_enabled(False),
            outputs=[generate_btn, retry_btn],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=generate,
            inputs=[story_input, llm_dropdown, model_dropdown, size_dropdown],
            outputs=[result_text, gallery, current_task_dir],
            concurrency_limit=1,
            concurrency_id="image-generation",
        )
        generate_event.then(
            fn=lambda: set_generation_buttons_enabled(True),
            outputs=[generate_btn, retry_btn],
            queue=False,
            show_progress="hidden",
        )

        retry_event = retry_btn.click(
            fn=lambda: set_generation_buttons_enabled(False),
            outputs=[generate_btn, retry_btn],
            queue=False,
            show_progress="hidden",
        ).then(
            fn=retry_failed_images,
            inputs=[current_task_dir],
            outputs=[result_text, gallery, current_task_dir],
            concurrency_limit=1,
            concurrency_id="image-generation",
        )
        retry_event.then(
            fn=lambda: set_generation_buttons_enabled(True),
            outputs=[generate_btn, retry_btn],
            queue=False,
            show_progress="hidden",
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
