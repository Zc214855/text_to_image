# 阶段 9：图片重试功能代码审查

> 状态：本文件记录的问题已在阶段 10 修复，详见 `10_retry_safety_fixes.md`。

## 结论

当前功能可运行，21 项单元测试、语法编译和 Gradio UI 构建均通过，但仍有以下问题。

## P1：并发任务会串用图片模型和尺寸

- `generate()` 和 `retry_failed_images()` 将任务参数写入全局 `config.IMAGE_MODEL`、`config.IMAGE_SIZE`。
- `image_generator.generate_image()` 在每张图片开始时重新读取这两个全局变量。
- “生成插图”和“仅重试失败图片”属于两个独立 Gradio 队列事件，可以同时执行。
- 已通过双线程测试复现：任务 A 原本选择 Kolors，任务 B 切换 Seedream 后，任务 A 也读取到 Seedream。

影响：

1. 图片可能使用错误模型或错误尺寸。
2. `generation.json` 记录的模型与实际调用模型可能不一致。
3. 可能产生预期外费用。

建议：

- 将 `image_model`、`image_size` 作为显式参数传递到 `generate_and_save()`，禁止生成链路读取可变全局配置。
- 为生成和重试事件配置相同的 `concurrency_id` 与单并发限制，防止同一用户重复点击产生并行任务。

## P1：生成中点击重试会重复生成并竞争写任务文件

- 页面任务状态只在完整生成结束时写入 `gr.State`。
- 生成过程中点击“仅重试失败图片”会回退查找最近的 `running` 任务。
- 尚未轮到的场景会被判断为缺失图片，并与原生成流程并行生成。
- `save_task()` 使用固定的 `generation.json.tmp` 临时文件；并发写同一任务时可能覆盖、替换失败或丢失状态。

影响：

1. 同一场景可能重复请求并重复计费。
2. `generation.json` 可能记录错误状态。
3. 其中一个流程可能因临时文件已被另一流程替换而异常。

建议：

- 生成运行期间禁用重试，或让重试拒绝 `running/retrying` 状态。
- 临时文件使用唯一文件名，并增加任务级互斥锁。

## P2：自动重试可能重复提交付费生图请求

- `httpx.TransportError` 和 `ReadTimeout` 会从完整生成流程重新开始。
- 如果服务端已接受首次 POST、但客户端读取响应超时，重试会再次提交一个新任务。
- DashScope 轮询超时后也会重新提交任务，而不是继续轮询原 `task_id`。

建议：

- 将“提交任务”“轮询任务”“下载图片”拆分并分别重试。
- 对结果不确定的 POST 超时不要自动重新提交，除非平台支持幂等键。
- 优先自动重试 429、明确 5xx、轮询 GET 和图片下载。

## P3：Seedream 4.0 的参数说明不准确

- 代码和测试声称 `output_format` 仅 Seedream 5.0 Lite 支持。
- 火山方舟官方 Seedream 4.0 示例明确支持 `output_format: "png"`。
- 当前省略该参数并使用 JPEG 通常可运行，但注释和测试名称错误。

## 验证结果

- `python -m unittest discover -s tests -v`：21 项通过。
- `python -m compileall -q .`：通过。
- `main.build_ui()`：通过。
- `.env` 当前状态：智谱 GLM-5.1、Seedream 5.0 Lite、`1728x2304`，模型和尺寸配置有效。
- 双线程模型隔离测试：失败，已稳定复现全局模型串用。
