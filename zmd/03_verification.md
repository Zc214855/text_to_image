# 阶段 3：验证结果

## 配置验证

- `.env` 已启用 `LLM_PROVIDER=zhipu`。
- 智谱真实请求成功，返回模型 `glm-5.1`。
- `.env` 已启用 `IMAGE_MODEL=doubao-seedream-5-0-260128`。
- 火山方舟 `/models` 返回 HTTP 200，共返回 119 个模型。
- 当前账户模型列表包含 `doubao-seedream-5-0-260128`。
- 未执行付费图片生成，避免产生未经确认的费用。

## 功能验证

- `python -m unittest discover -s tests -v`：7 项测试全部通过。
- `python -m compileall -q .`：通过。
- `main.build_ui()`：通过。
- `git diff --check`：通过。

## 真实分镜验证

使用 `examples/little_red_riding_hood.txt` 调用智谱 GLM-5.1：

- 角色定义：5 个。
- 目标场景：14 个。
- 实际场景：14 个。
- 最长最终提示词：1106 字符。
- 原输出为 17 个场景，第一次优化仍产生 21 个场景；调整合并规则后稳定为 14 个完整动作镜头。

## 安全项

`.env.example` 中出现过真实智谱和火山密钥。即使当前文件已改为占位符，这两个密钥仍应在控制台轮换。
