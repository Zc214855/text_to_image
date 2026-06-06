import json
import re
from json_repair import repair_json
from openai import OpenAI
from config import get_llm_client_config

SYSTEM_PROMPT = """You are a fairy tale illustration assistant. Your job is to:
1. Read the given fairy tale text
2. Split it into coherent visual story beats. Each scene must show one complete, decisive action. Merge dialogue or narration that has no independent visual action into the nearest scene.
3. Define ALL visible named or recurring characters before writing scenes.
4. Build one consistent art direction for the entire book.
5. For each scene, return structured visual fields. Do not repeat character definitions or the shared style inside visual_action.

## Character Definition Rules:
- List every visible character, including characters that appear only once
- Give each character a concise FIXED English visual description: age, body type, face/hair/fur, clothing colors, and one distinctive feature
- Keep the visual description under 45 English words
- Do not include pose, action, emotion, environment, lighting, or art style in character definitions

## Scene Rules:
- Cover the full story in chronological order without inventing plot events
- Keep causally connected actions together; do not create fragments such as a scene containing only "再也没有起来"
- visual_action must describe visible poses, interaction, facial expressions, and the exact story action
- environment must state location and story-relevant objects
- composition must state subject placement, foreground, middle ground, and background when useful
- Vary shot types deliberately across scenes: establishing shot, wide shot, medium shot, close-up, over-the-shoulder, low angle, or high angle
- Preserve important state from the previous scene, such as disguise, carried objects, weather, damage, or time of day
- Exclude typography, signs, labels, captions, letters, logos, watermarks, and speech bubbles
- The response must be valid JSON. Never place an unescaped ASCII double quote inside a JSON string
- In Chinese story_text, replace dialogue quotation marks with Chinese corner quotes 「」 instead of ASCII double quotes

You MUST respond in the following JSON format only, no other text:
{
  "title": "story title in Chinese",
  "characters": [
    {
      "name": "角色中文名",
      "visual": "consistent English visual description of this character"
    }
  ],
  "style": "fixed English art direction: medium, line quality, palette, lighting logic, historical setting, age suitability",
  "scenes": [
    {
      "scene_number": 1,
      "story_text": "this scene's story content in Chinese",
      "characters_in_scene": ["角色名1", "角色名2"],
      "shot": "English shot type and camera angle",
      "visual_action": "English visible action, poses, interaction, and expressions",
      "environment": "English location and story-relevant objects",
      "composition": "English spatial composition and focal point",
      "lighting": "English lighting, time, and mood"
    }
  ]
}"""

NO_TEXT_RULE = (
    "clean illustration with no typography, no signs, no labels, no captions, "
    "no letters, no logos, no watermark, no speech bubbles"
)


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


def estimate_scene_count(story_text: str) -> int:
    """估算完整动作镜头数，避免把对话和连续动作逐句拆开。"""
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?])", story_text)
        if part.strip()
    ]
    return max(4, min(18, round((len(sentences) or 1) * 0.65)))


def parse_story(story_text: str) -> dict:
    """Use LLM to split story into scenes and generate image prompts."""
    base_url, api_key, model = get_llm_client_config()
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 以完整句子和动作节拍估算，避免按字符数机械切出叙事碎片。
    target_scenes = estimate_scene_count(story_text)

    user_message = (
        f"请将以下童话故事拆分为大约 {target_scenes} 个完整视觉场景。"
        f"场景总数不得超过 {min(18, target_scenes + 2)} 个。"
        "优先合并连续对话和同一地点内的连续动作；保证叙事完整、角色一致和画面可见，"
        "不要为了凑数量拆出残句。\n\n"
        f"---\n{story_text}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.35,
        max_tokens=16384,
    )

    content = response.choices[0].message.content.strip()

    # Extract JSON from response (handle possible markdown code blocks)
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    result = parse_llm_json(content)

    if not isinstance(result.get("scenes"), list) or not result["scenes"]:
        raise ValueError(f"LLM response missing 'scenes': {content}")

    style = result.get(
        "style",
        "hand-painted children's picture book, watercolor and gouache, "
        "gentle textured paper, cohesive muted color palette, child-friendly",
    )
    characters = {}
    for character in result.get("characters", []):
        name = character.get("name")
        visual = character.get("visual")
        if name and visual:
            characters[name] = visual

    for index, scene in enumerate(result["scenes"], start=1):
        if not isinstance(scene, dict):
            raise ValueError(f"Scene {index} must be a JSON object")
        if not scene.get("story_text"):
            raise ValueError(f"Scene {index} missing story_text")

        character_visuals = []
        for char_name in scene.get("characters_in_scene", []):
            visual = characters.get(char_name)
            if not visual:
                raise ValueError(
                    f"Scene {index} references undefined character: {char_name}"
                )
            character_visuals.append(visual)

        visual_action = scene.get("visual_action") or scene.get("prompt")
        if not visual_action:
            raise ValueError(f"Scene {index} missing visual_action")

        # 固定顺序组装提示词，每项只出现一次，降低重复描述对模型注意力的干扰。
        prompt_parts = [
            style,
            scene.get("shot", ""),
            "; ".join(character_visuals),
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
