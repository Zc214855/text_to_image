import json
import re
from json_repair import repair_json
from openai import OpenAI
from config import get_llm_client_config

SYSTEM_PROMPT_OUTLINE = """You are a fairy tale illustration assistant. Your job is to:
1. Read the given fairy tale text
2. Split it into coherent visual story beats. Each scene must show one complete, decisive action. Merge dialogue or narration that has no independent visual action into the nearest scene.
3. Define ALL visible named or recurring characters before writing scenes.
4. Define recurring visible objects and important locations as fixed visual anchors.
5. Build one consistent art direction for the entire book.
6. For each scene, return ONLY lightweight fields: story_text and characters_in_scene. Do NOT generate visual_action / shot / environment / composition / lighting / state_tracking yet — those will be filled in a second pass.

## Character Definition Rules:
- List every visible character, including characters that appear only once
- Give each character a concise FIXED English visual description: age, body type, face/hair/fur, clothing colors, and one distinctive feature
- Keep the visual description under 45 English words
- Give each character a short_visual under 16 English words: identity, clothing main color, and one distinctive feature
- Set importance to "main", "supporting", or "background"
- Do not include pose, action, emotion, environment, lighting, or art style in character definitions
- Character visual descriptions must be pure English and must not contain Chinese characters

## Visual Anchor Rules:
- recurring_objects: list story-critical recurring visible objects, magical items, carried items, vehicles, or distinctive props
- locations: list recurring or important visual locations
- Give each object/location a concise FIXED English visual description under 35 English words
- Do not include action, emotion, camera, lighting, or typography in object/location definitions
- Object and location visual descriptions must be pure English and must not contain Chinese characters

## Scene Outline Rules:
- Cover the full story in chronological order without inventing plot events
- Keep causally connected actions together; do not create fragments such as a scene containing only "再也没有起来"
- Different locations or different actions should each become their own scene — give the reader as many distinct visuals as possible
- When a scene is primarily dialogue, plan to give characters a physical activity (crafting, walking, running, cooking, etc.) instead of two people standing face-to-face merely talking — this will be filled in the next pass
- The response must be valid JSON. Never place an unescaped ASCII double quote inside a JSON string
- In Chinese story_text, replace dialogue quotation marks with Chinese corner quotes 「」 instead of ASCII double quotes

## Style Field:
- Must include: medium, line quality, color palette, lighting logic, historical/cultural setting, age suitability, and quality phrase such as "highly detailed" or "best quality"
- Must be pure English and must not contain Chinese characters

You MUST respond in the following JSON format only, no other text:
{
  "title": "story title in Chinese",
  "characters": [
    {
      "name": "角色中文名",
      "visual": "consistent English visual description of this character",
      "short_visual": "compact English visual anchor under 16 words",
      "importance": "main | supporting | background"
    }
  ],
  "recurring_objects": [
    {
      "name": "道具中文名",
      "visual": "consistent English visual description of this object"
    }
  ],
  "locations": [
    {
      "name": "地点中文名",
      "visual": "consistent English visual description of this location"
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
- Recurring object and location visual anchors
- A batch of scene outlines (story_text + characters_in_scene)

For each scene, you must add these English fields:
- location_in_scene: one exact location anchor name from the provided locations, or an empty string if none applies
- objects_in_scene: exact recurring object anchor names visibly present in the scene
- shot: shot type and camera angle (vary deliberately: establishing shot, wide shot, medium shot, close-up, over-the-shoulder, low angle, high angle)
- visual_action: visible poses, interaction, facial expressions, and the exact story action. When a scene is primarily dialogue, characters MUST be shown performing a physical activity — never two characters standing face-to-face merely talking
- environment: location and story-relevant objects
- composition: ONLY spatial layout and subject placement (foreground, middle ground, background, focal point). Do NOT repeat environment objects or story action here
- lighting: lighting, time of day, and mood
- state_tracking: what has changed or been carried forward from the previous scene (e.g., "still wearing red hood", "now holding lantern", "snow has stopped", "dusk → night")

Rules:
- Preserve important state from the previous scene such as disguise, carried objects, weather, damage, or time of day
- Exclude typography, signs, labels, captions, letters, logos, watermarks, and speech bubbles
- All new visual detail fields except location_in_scene and objects_in_scene must be pure English and must not contain Chinese characters
- location_in_scene and objects_in_scene must use exact anchor names from the provided lists
- The response must be valid JSON. Never place an unescaped ASCII double quote inside a JSON string
- Return the same scene_number and all original fields plus the new ones

You MUST respond in the following JSON format only, no other text:
{
  "scenes": [
    {
      "scene_number": 1,
      "story_text": "unchanged",
      "characters_in_scene": ["unchanged"],
      "location_in_scene": "provided location anchor name or empty string",
      "objects_in_scene": ["provided object anchor name"],
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
DETAIL_RETRY_COUNT = 2
MAX_SCENE_COUNT = 24
FALLBACK_DETAIL_WARN_RATIO = 0.5
ENGLISH_DETAIL_FIELDS = (
    "shot",
    "visual_action",
    "environment",
    "composition",
    "lighting",
    "state_tracking",
)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _scene_count_limit(story_text: str) -> int:
    """按故事长度给出场景上限；长故事允许更多插图，短故事避免被拆得过碎。"""
    story_length = len(story_text)
    if story_length < 800:
        return 10
    if story_length < 1600:
        return 16
    if story_length < 2800:
        return 20
    return MAX_SCENE_COUNT


def _extract_short_ref(full_visual: str) -> str:
    """从完整角色描述生成兜底短锚点，限制长度以减少多角色场景注意力稀释。"""
    words = full_visual.strip().split()
    if len(words) <= 16:
        return full_visual.strip()
    return " ".join(words[:16]).strip(" ,.;")


def _extract_medium_tag(style: str) -> str:
    """保留完整风格锚点，防止后续插图丢失调色板、地域和光照逻辑。"""
    return style.strip()


def _contains_cjk(value: str) -> bool:
    """判断文本是否含中文字符；图像 prompt 的英文视觉字段必须通过此检查。"""
    return bool(CJK_RE.search(value or ""))


def _strip_cjk(value: str) -> str:
    """清除会污染英文图像 prompt 的中文字符，并压缩多余空白。"""
    return re.sub(r"\s+", " ", CJK_RE.sub("", value or "")).strip()


def _build_anchor_map(items: list, field_name: str) -> dict:
    """构建视觉锚点表，并拒绝会污染英文 prompt 的中文视觉描述。"""
    anchors = {}
    if not isinstance(items, list):
        raise ValueError(f"LLM 大纲响应字段 '{field_name}' 必须是列表")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"LLM 大纲响应字段 '{field_name}' 包含非对象条目")
        name = item.get("name")
        visual = item.get("visual")
        if not name or not visual:
            raise ValueError(f"LLM 大纲响应字段 '{field_name}' 包含缺失 name/visual 的条目")
        if _contains_cjk(visual):
            raise ValueError(f"LLM 大纲响应字段 '{field_name}' 的视觉描述含中文字符：{name}")
        anchors[name] = {
            "full": visual.strip(),
            "short": _extract_short_ref(visual),
        }
    return anchors


def _build_character_map(items: list) -> dict:
    """构建角色锚点表，支持 full/short/importance 三层提示词策略。"""
    characters = {}
    if not isinstance(items, list):
        raise ValueError("LLM 大纲响应字段 'characters' 必须是列表")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("LLM 大纲响应字段 'characters' 包含非对象条目")
        name = item.get("name")
        visual = item.get("visual")
        if not name or not visual:
            raise ValueError("LLM 大纲响应字段 'characters' 包含缺失 name/visual 的条目")
        short_visual = item.get("short_visual") or _extract_short_ref(visual)
        importance = item.get("importance", "supporting")
        if importance not in {"main", "supporting", "background"}:
            importance = "supporting"
        if _contains_cjk(visual) or _contains_cjk(short_visual):
            raise ValueError(f"LLM 大纲响应字段 'characters' 的视觉描述含中文字符：{name}")
        characters[name] = {
            "full": visual.strip(),
            "short": short_visual.strip(),
            "importance": importance,
        }
    return characters


def _normalize_outline_scenes(all_scenes: list, characters: dict) -> list:
    """校验第一轮场景大纲，并移除未定义角色引用，避免单个漏定义角色中断任务。"""
    dropped = []
    seen_numbers = set()
    for scene in all_scenes:
        if not isinstance(scene, dict):
            raise ValueError("LLM 大纲响应 scenes 包含非对象条目")
        scene_number = scene.get("scene_number")
        if not isinstance(scene_number, int) or scene_number in seen_numbers:
            raise ValueError(f"LLM 大纲响应 scene_number 非法或重复：{scene_number}")
        seen_numbers.add(scene_number)
        if not scene.get("story_text"):
            raise ValueError(f"LLM 大纲响应场景 {scene_number} 缺少 story_text")
        characters_in_scene = scene.get("characters_in_scene", [])
        if not isinstance(characters_in_scene, list):
            raise ValueError(f"LLM 大纲响应场景 {scene_number} 的 characters_in_scene 必须是列表")
        undefined = [name for name in characters_in_scene if name not in characters]
        if undefined:
            scene["characters_in_scene"] = [
                name for name in characters_in_scene if name in characters
            ]
            dropped.extend(
                {"scene_number": scene_number, "name": name}
                for name in undefined
            )
    return dropped


def _validate_detail_scenes(
    batch_result: dict,
    expected_batch: list,
    objects: dict,
    locations: dict,
    batch_idx: int,
) -> list:
    """规范化第二轮视觉细节；缺失或不合法字段用大纲和安全默认值兜底。"""
    scenes = batch_result.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError(f"第二轮 LLM 响应缺少 'scenes'（批次 {batch_idx + 1}）")

    expected_by_number = {scene["scene_number"]: scene for scene in expected_batch}
    expected_numbers = list(expected_by_number)
    scenes_by_number = {}
    fallback_numbers = []

    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_number = scene.get("scene_number")
        if scene_number not in expected_by_number:
            continue
        if scene_number in scenes_by_number:
            continue
        expected = expected_by_number[scene_number]

        # story_text 和 characters_in_scene 以第一轮大纲为准，避免第二轮微调标点或角色表导致断批。
        scene["story_text"] = expected["story_text"]
        scene["characters_in_scene"] = expected.get("characters_in_scene", [])

        scene.setdefault("location_in_scene", "")
        scene.setdefault("objects_in_scene", [])
        if scene.get("location_in_scene") not in locations:
            scene["location_in_scene"] = ""

        object_names = scene.get("objects_in_scene", [])
        if not isinstance(object_names, list):
            object_names = []
        scene["objects_in_scene"] = [
            name for name in object_names if name in objects
        ]

        fallback_text = "a clear storybook illustration of this scene"
        for field in ENGLISH_DETAIL_FIELDS:
            value = scene.get(field, "")
            if _contains_cjk(value):
                scene.setdefault("_sanitized_fields", []).append(field)
                value = _strip_cjk(value)
            else:
                value = str(value or "").strip()
            scene[field] = value or fallback_text

        scenes_by_number[scene_number] = scene

    for number in expected_numbers:
        if number in scenes_by_number:
            continue
        expected = expected_by_number[number]
        fallback_numbers.append(number)
        scenes_by_number[number] = {
            "scene_number": number,
            "story_text": expected["story_text"],
            "characters_in_scene": expected.get("characters_in_scene", []),
            "location_in_scene": "",
            "objects_in_scene": [],
            "shot": "medium shot at eye level",
            "visual_action": "a clear storybook illustration of this scene",
            "environment": "storybook setting matching the scene",
            "composition": "main subject centered with readable foreground and background",
            "lighting": "soft child-friendly storybook lighting",
            "state_tracking": "continue visual continuity from the previous scene",
        }

    if fallback_numbers and len(fallback_numbers) / max(len(expected_numbers), 1) >= FALLBACK_DETAIL_WARN_RATIO:
        for number in fallback_numbers:
            scenes_by_number[number]["_fallback_detail"] = True

    return [scenes_by_number[number] for number in expected_numbers]


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
    最终按故事长度夹紧到 [4, 24] 区间，长故事保留更多插图预算。
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
    return max(4, min(_scene_count_limit(story_text), round(adjusted)))


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
    scene_limit = _scene_count_limit(story_text)

    user_message_outline = (
        f"请将以下童话故事拆分为大约 {target_scenes} 个完整视觉场景。"
        f"场景总数不得超过 {scene_limit} 个。"
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
    if _contains_cjk(style):
        raise ValueError("LLM 大纲响应 style 含中文字符")

    characters = _build_character_map(outline.get("characters", []))
    objects = _build_anchor_map(outline.get("recurring_objects", []), "recurring_objects")
    locations = _build_anchor_map(outline.get("locations", []), "locations")

    # ========== 第二轮：分批填充视觉细节 ==========
    all_scenes = outline["scenes"]
    dropped_characters = _normalize_outline_scenes(all_scenes, characters)
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
            context_parts.append(
                f"- {name}: {info['full']} | short: {info['short']} | importance: {info['importance']}"
            )
        context_parts.append("\n## Recurring Objects")
        for name, info in objects.items():
            context_parts.append(f"- {name}: {info['full']}")
        context_parts.append("\n## Locations")
        for name, info in locations.items():
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

        valid = None
        last_error = None
        for attempt in range(DETAIL_RETRY_COUNT + 1):
            try:
                retry_hint = ""
                if attempt and last_error:
                    retry_hint = (
                        "\n\nPrevious response was rejected because: "
                        f"{last_error}. Return corrected JSON only."
                    )
                batch_content = _call_llm(
                    client, model, SYSTEM_PROMPT_DETAIL, user_message_detail + retry_hint
                )
                batch_content = _extract_json_block(batch_content)
                batch_result = parse_llm_json(batch_content)
                valid = _validate_detail_scenes(
                    batch_result, batch, objects, locations, batch_idx
                )
                break
            except ValueError as error:
                last_error = error
        if valid is None:
            raise ValueError(
                f"第二轮批次 {batch_idx + 1} 生成失败，无法生成稳定插图提示词"
            ) from last_error

        detailed_scenes.extend(valid)

    result = {
        "title": title,
        "characters": [
            {
                "name": name,
                "visual": info["full"],
                "short_visual": info["short"],
                "importance": info["importance"],
            }
            for name, info in characters.items()
        ],
        "recurring_objects": [
            {"name": name, "visual": info["full"]}
            for name, info in objects.items()
        ],
        "locations": [
            {"name": name, "visual": info["full"]}
            for name, info in locations.items()
        ],
        "style": style,
        "scenes": detailed_scenes,
    }
    if dropped_characters:
        result["_dropped_characters"] = dropped_characters

    sanitized_prompts = []
    first_appearance = set()
    for index, scene in enumerate(result["scenes"], start=1):
        character_refs = []
        scene_characters = scene.get("characters_in_scene", [])
        many_characters = len(scene_characters) >= 3
        for char_name in scene.get("characters_in_scene", []):
            char_info = characters.get(char_name)
            if char_info:
                if char_info["importance"] == "main":
                    character_refs.append(char_info["full"])
                elif many_characters or char_info["importance"] == "background" or char_name in first_appearance:
                    character_refs.append(char_info["short"])
                else:
                    character_refs.append(char_info["full"])
                first_appearance.add(char_name)

        object_refs = []
        for object_name in scene.get("objects_in_scene", []):
            object_info = objects.get(object_name)
            if object_info:
                object_refs.append(object_info["full"])

        location_ref = ""
        location_name = scene.get("location_in_scene", "")
        location_info = locations.get(location_name)
        if location_info:
            location_ref = location_info["full"]

        visual_action = scene.get("visual_action") or scene.get("prompt")

        # 完整风格、角色、道具、地点锚点每张图都保留，优先保证跨场景一致性。
        style_for_prompt = _extract_medium_tag(style)

        # 质量前缀 + 按视觉重要性排序，关键信息放前面
        prompt_parts = [
            QUALITY_PREFIX,
            style_for_prompt,
            scene.get("shot", ""),
            "; ".join(character_refs),
            location_ref,
            "; ".join(object_refs),
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
        if _contains_cjk(scene["prompt"]):
            scene["prompt"] = _strip_cjk(scene["prompt"])
            sanitized_prompts.append(index)
        elif scene.get("_sanitized_fields") or scene.get("_fallback_detail"):
            sanitized_prompts.append(index)

    if sanitized_prompts:
        result["_sanitized_prompts"] = sanitized_prompts

    return result
