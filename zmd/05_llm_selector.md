# 阶段 5：分镜 LLM 选择

## 参数分组

- `LLM_PROVIDER`：启动时默认使用的分镜 LLM 供应商。
- `LLM_BASE_URL`：默认分镜 LLM 的 OpenAI 兼容接口地址。
- `LLM_API_KEY`：默认分镜 LLM 的鉴权密钥。
- `LLM_MODEL`：默认分镜 LLM 的模型 ID。
- `ZHIPU_*`：智谱专用配置。
- `ARK_*`：火山方舟文本和图片模型配置。
- `IMAGE_MODEL` / `IMAGE_SIZE`：启动时默认图片模型及尺寸。

## 已完成

1. 清除 `.env` 内重复的两组 `LLM_*` 配置。
2. 保留 `LLM_PROVIDER=zhipu` 作为启动默认值。
3. 新增“分镜 LLM”下拉框。
4. 支持智谱 GLM-5.1、硅基流动 Qwen3-8B 和火山豆包文本模型。
5. 切换 LLM 后同步更新模型名称、用途和连接状态。
6. 页面切换只影响当前进程，重启后恢复 `.env` 默认值。

## 验证

- `.env` 中仅保留一组通用 `LLM_*` 配置。
- 当前默认供应商：`zhipu`。
- 当前默认模型：`glm-5.1`。
- 10 项单元测试通过。
