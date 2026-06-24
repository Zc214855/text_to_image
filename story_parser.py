import json
import re
from json_repair import repair_json
from openai import OpenAI
from config import get_llm_client_config

SYSTEM_PROMPT_OUTLINE = """You are a fairy tale illustration assistant. Your job is to:
1. Read the given fairy tale text
2. Split it into coherent visual story beats. Each scene must show one complete, decisive action. Merge dialogue or narration that has no independent visual action into the nearest scene.
3. Define ALL visible named or recurring characters before writing scenes.
4. Build one consistent art direction for the entire book.
5. For each scene, return ONLY lightweight fields: story_text and characters_in_scene. Do NOT generate visual_action / shot / environment / composition / lighting / state_tracking yet — those will be filled in a second pass.

## Character Definition Rules:
- List every visible character, including characters that appear only once
- Give each character a concise FIXED English visual description: age, body type, face/hair/fur, clothing colors, and one distinctive feature
- Keep the visual description under 45 English words
- Do not include pose, action, emotion, environment, lighting, or art style in character definitions

## Scene Outline Rules:
- Cover the full story in chronological order without inventing plot events
- Keep causally connected actions together; do not create fragments such as a scene containing only "再也没有起来"
- Different locations or different actions should each become their own scene — give the reader as many distinct visuals as possible
- When a scene is primarily dialogue, plan to give characters a physical activity (crafting, walking, running, cooking, etc.) instead of two people standing face-to-face merely talking — this will be filled in the next pass
- The response must be valid JSON. Never place an unescaped ASCII double quote inside a JSON string
- In Chinese story_text, replace dialogue quotation marks with Chinese corner quotes 「」 instead of ASCII double quotes

## Style Field:
- Must include: medium, line quality, color palette, lighting logic, historical/cultural setting, age suitability, and quality phrase such as "highly detailed" or "best quality"

You MUST respond in the following JSON format only, no other text:
{
  "title": "story title in Chinese",
  "characters": [
    {
      "name": "角色中文名",
      "visual": "consistent English visual description of this character"
    }
  ],
  "style": "fixed English art direction: medium, line quality, palette, lighting logic, historical setting, age suitability, quality phrase",
  "scenes": [
    {
      "scene_number": 1,
      "story_text": "this scene's story content in Chinese",
      "characters_in_scene": ["角色名1", "角色名2"]
    }
  ]
}"""

SYSTEM_PROMPT_DETAIL = """You are a fairy tale illustration assistant filling in visual details for pre-defined scenes.

You will receive:
- The book's art style
- All character visual definitions
- A batch of scene outlines (story_text + characters_in_scene)

For each scene, you must add these English fields:
- shot: shot type and camera angle (vary deliberately: establishing shot, wide shot, medium shot, close-up, over-the-shoulder, low angle, high angle)
- visual_action: visible poses, interaction, facial expressions, and the exact story action. When a scene is primarily dialogue, characters MUST be shown performing a physical activity — never two characters standing face-to-face merely talking
- environment: location and story-relevant objects
- composition: ONLY spatial layout and subject placement (foreground, middle ground, background, focal point). Do NOT repeat environment objects or story action here
- lighting: lighting, time of day, and mood
- state_tracking: what has changed or been carried forward from the previous scene (e.g., "still wearing red hood", "now holding lantern", "snow has stopped", "dusk → night")

Rules:
- Preserve important state from the previous scene such as disguise, carried objects, weather, damage, or time of day
- Exclude typography, signs, labels, captions, letters, logos, watermarks, and speech bubbles
- The response must be valid JSON. Never place an unescaped ASCII double quote inside a JSON string
- Return the same scene_number and all original fields plus the new ones

You MUST respond in the following JSON format only, no other text:
{
  "scenes": [
    {
      "scene_number": 1,
      "story_text": "unchanged",
      "characters_in_scene": ["unchanged"],
      "shot": "English shot type and camera angle",
      "visual_action": "English visible action, poses, interaction, and expressions",
      "environment": "English location and story-relevant objects",
      "composition": "English spatial layout ONLY — no environment repetition",
      "lighting": "English lighting, time, and mood",
      "state_tracking": "English description of state changes or carried-forward elements"
    }
  ]
}"""

QUALITY_PREFIX = (
    "masterpiece, best quality, highly detailed children's book illustration"
)

NO_TEXT_RULE = (
    "clean illustration with no typography, no signs, no labels, no captions, "
    "no letters, no logos, no watermark, no speech bubbles"
)

# 每批填充的场景数量，控制单次输出长度避免截断
DETAIL_BATCH_SIZE = 8


def _extract_short_ref(full_visual: str) -> str:
    """从完整角色描述中提取简短引用：取前两个逗号分隔片段（≈年龄+外貌+服装）。"""
    parts = [p.strip() for p in full_visual.split(",") if p.strip()]
    if len(parts) <= 2:
        return full_visual
    return ", ".join(parts[:2])


def _extract_medium_tag(style: str) -> str:
    """从完整风格中提取画种+质量标记，去掉详细调色板和时代描述以缩短后续 prompt。"""
    # 取前两个逗号片段（通常是画种和笔触），保留末尾质量词
    parts = [p.strip() for p in style.split(",") if p.strip()]
    if len(parts) <= 2:
        return style
    quality_keywords = [p for p in parts if any(k in p.lower() for k in ("detail", "quality", "best", "highly"))]
    prefix = ", ".join(parts[:2])
    if quality_keywords:
        return f"{prefix}, {', '.join(quality_keywords)}"
    return prefix


def parse_llm_json(content: str) -> dict:
    """解析 LLM JSON；标准解析失败时修复常见的引号、逗号和截断问题。"""
    try:
        result = json.loads(content)
    except json.JSONDecodeError as original_error:
        try:
            repaired_content = repair_json(content)
            result = json.loads(repaired_content)
        except (json.JSONDecodeError, ValueError, TypeError) as repair_error:
            raise ValueError(
                f"Failed to parse or repair LLM response as JSON:\n{content}"
            ) from repair_error

        if not isinstance(result, dict):
            raise ValueError(
                f"Repaired LLM response is not a JSON object:\n{content}"
            ) from original_error

    if not isinstance(result, dict):
        raise ValueError(f"LLM response must be a JSON object:\n{content}")
    return result


def _call_llm(client: OpenAI, model: str, system: str, user: str) -> str:
    """调用 LLM 并返回原始文本。"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.35,
        max_tokens=16384,
    )
    return response.choices[0].message.content.strip()


def _extract_json_block(content: str) -> str:
    """从 markdown 代码块或原始文本中提取 JSON。"""
    if "```json" in content:
        return content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        return content.split("```")[1].split("```")[0].strip()
    return content


def estimate_scene_count(story_text: str) -> int:
    """基于句子数、段落结构和对话密度估算视觉场景数，鼓励更多画面。

    算法：
    1. 以句子数 * 0.65 为基础（与原逻辑一致）
    2. 至少不低于段落数（每个叙事段落至少一个画面）
    3. 对话密集时缩减场景数（合并对话节拍）
    4. 长篇故事补充额外预算
    最终夹紧到 [4, 25] 区间。
    """
    paragraphs = [p.strip() for p in re.split(r"\n+", story_text) if p.strip()]
    if not paragraphs:
        return 4

    sentences = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?])", story_text)
        if part.strip()
    ]

    # 基础数：句子 * 0.65，但不少于段落数
    base = max(len(paragraphs), round(len(sentences) * 0.65))

    # 对话密集时缩减：引号字符占比越高，合并越多（最多缩减 30%）
    dialogue_chars = sum(1 for c in story_text if c in '"「」『』""')
    dialogue_ratio = dialogue_chars / max(len(story_text), 1)
    dialogue_reduction = round(base * dialogue_ratio * 0.3)

    # 长篇故事额外预算
    char_bonus = max(0, (len(story_text) - 500) / 200)

    adjusted = base - dialogue_reduction + char_bonus
    return max(4, min(25, round(adjusted)))


def parse_story(
    story_text: str,
    llm_client_config: tuple[str, str, str] | None = None,
) -> dict:
    """Use LLM to split story into scenes and generate image prompts.

    采用两轮生成策略：
    1. 第一轮：生成标题、角色定义、风格和场景大纲（仅 story_text + 角色列表）
    2. 第二轮：分批为大纲场景填充视觉详细信息（shot/visual_action/environment 等）

    这样每个场景的输出被拆分到两次较短的 LLM 调用中，避免长输出截断。
    """
    base_url, api_key, model = llm_client_config or get_llm_client_config()
    client = OpenAI(api_key=api_key, base_url=base_url)

    # ========== 第一轮：生成大纲 ==========
    target_scenes = estimate_scene_count(story_text)

    user_message_outline = (
        f"请将以下童话故事拆分为大约 {target_scenes} 个完整视觉场景。"
        f"场景总数不得超过 {min(25, target_scenes + 3)} 个。"
        "优先合并连续对话和同一地点内的连续动作；保证叙事完整、角色一致和画面可见，"
        "不要为了凑数量拆出残句。但同时，不同地点、不同动作的画面应各自成景，"
        "让读者看到更多画面、更多视角。"
        "注意：只需输出角色定义、风格和场景大纲（story_text + characters_in_scene），"
        "不要生成 visual_action / shot / environment 等视觉细节。\n\n"
        f"---\n{story_text}"
    )

    content = _call_llm(client, model, SYSTEM_PROMPT_OUTLINE, user_message_outline)
    content = _extract_json_block(content)

    outline = parse_llm_json(content)

    if not isinstance(outline.get("scenes"), list) or not outline["scenes"]:
        raise ValueError(f"LLM 大纲响应缺少 'scenes': {content}")

    title = outline.get("title", "未命名")
    style = outline.get(
        "style",
        "hand-painted children's picture book, watercolor and gouache, "
        "gentle textured paper, cohesive muted color palette, "
        "highly detailed, child-friendly",
    )

    characters = {}
    for character in outline.get("characters", []):
        name = character.get("name")
        visual = character.get("visual")
        if name and visual:
            characters[name] = {
                "full": visual,
                "short": _extract_short_ref(visual),
            }

    # ========== 第二轮：分批填充视觉细节 ==========
    all_scenes = outline["scenes"]
    detailed_scenes = []

    total_batches = (len(all_scenes) + DETAIL_BATCH_SIZE - 1) // DETAIL_BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * DETAIL_BATCH_SIZE
        end = min(start + DETAIL_BATCH_SIZE, len(all_scenes))
        batch = all_scenes[start:end]

        # 构建上下文：角色定义 + 风格 + 前一批最后一个场景的状态（如有）
        context_parts = [
            f"## Style\n{style}",
            "\n## Characters",
        ]
        for name, info in characters.items():
            context_parts.append(f"- {name}: {info['full']}")

        # 告知上一批的最后场景状态，确保跨批一致性
        if detailed_scenes:
            last_prev = detailed_scenes[-1]
            context_parts.append(
                f"\n## Previous scene state (for continuity)\n"
                f"Scene {last_prev['scene_number']}: "
                f"state_tracking = \"{last_prev.get('state_tracking', '')}\", "
                f"lighting = \"{last_prev.get('lighting', '')}\""
            )

        context_parts.append("\n## Scenes to fill")

        batch_for_prompt = []
        for s in batch:
            batch_for_prompt.append({
                "scene_number": s["scene_number"],
                "story_text": s["story_text"],
                "characters_in_scene": s.get("characters_in_scene", []),
            })

        user_message_detail = (
            "请为以下场景填充视觉细节（shot / visual_action / environment / "
            "composition / lighting / state_tracking）。\n\n"
            + "\n".join(context_parts)
            + "\n\n```json\n"
            + json.dumps({"scenes": batch_for_prompt}, ensure_ascii=False, indent=2)
            + "\n```"
        )

        batch_content = _call_llm(
            client, model, SYSTEM_PROMPT_DETAIL, user_message_detail
        )
        batch_content = _extract_json_block(batch_content)
        batch_result = parse_llm_json(batch_content)

        if not isinstance(batch_result.get("scenes"), list):
            raise ValueError(
                f"第二轮 LLM 响应缺少 'scenes'（批次 {batch_idx + 1}）: {batch_content}"
            )

        # 过滤残缺场景
        valid = []
        for scene in batch_result["scenes"]:
            if not isinstance(scene, dict):
                continue
            if not scene.get("story_text") or not scene.get("visual_action"):
                continue
            valid.append(scene)

        if not valid:
            raise ValueError(
                f"第二轮批次 {batch_idx + 1} 所有场景均不完整，无法生成插图"
            )

        detailed_scenes.extend(valid)

    # ========== 组装最终结果 ==========
    result = {
        "title": title,
        "characters": [
            {"name": name, "visual": info["full"]}
            for name, info in characters.items()
        ],
        "style": style,
        "scenes": detailed_scenes,
    }

    # 检查是否有场景在大纲中但未在细节中生成
    outline_numbers = {s["scene_number"] for s in all_scenes}
    detail_numbers = {s["scene_number"] for s in detailed_scenes}
    missing = outline_numbers - detail_numbers
    if missing:
        result["_dropped_scenes"] = [
            f"场景 {n} 在大纲中存在但第二轮生成失败" for n in sorted(missing)
        ]

    # 追踪每个角色是否已首次完整出场
    first_appearance = {}

    for index, scene in enumerate(result["scenes"], start=1):
        character_refs = []
        for char_name in scene.get("characters_in_scene", []):
            char_info = characters.get(char_name)
            if not char_info:
                continue
            # 首次出场使用完整描述，后续场景使用简短引用，减少注意力稀释
            if char_name not in first_appearance:
                character_refs.append(char_info["full"])
                first_appearance[char_name] = index
            else:
                character_refs.append(char_info["short"])

        visual_action = scene.get("visual_action") or scene.get("prompt")

        # 首个场景保留完整风格描述方便构图；后续场景只保留画种和质量标记，缩短 prompt
        style_for_prompt = style if index == 1 else _extract_medium_tag(style)

        # 质量前缀 + 按视觉重要性排序，关键信息放前面
        prompt_parts = [
            QUALITY_PREFIX,
            style_for_prompt,
            scene.get("shot", ""),
            "; ".join(character_refs),
            visual_action,
            scene.get("environment", ""),
            scene.get("composition", ""),
            scene.get("lighting", ""),
            NO_TEXT_RULE,
        ]
        scene["scene_number"] = index
        scene["prompt"] = ". ".join(
            part.strip(" .,") for part in prompt_parts if part and part.strip()
        )

    return result
