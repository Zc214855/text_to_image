import json
from openai import OpenAI
from config import SILICONFLOW_API_KEY, BASE_URL, LLM_MODEL

SYSTEM_PROMPT = """You are a fairy tale illustration assistant. Your job is to:
1. Read the given fairy tale text carefully
2. Split it into scenes based on PLOT and VISUAL content — each scene should be a meaningful visual moment that can be illustrated as a standalone picture
3. First define ALL recurring characters with consistent visual descriptions
4. For each scene, write an English image generation prompt that MUST reuse the exact same character descriptions and MUST closely match the specific story content of that scene

## How to split scenes:
- Split by PLOT BEATS, not by character count. Each scene should represent ONE coherent visual moment
- A scene could be: a character doing something, a dialogue moment, an emotional reaction, a setting change, a key event
- If a sentence contains multiple distinct actions or visual moments, split it further
- If a paragraph flows as one visual scene, keep it together even if it's longer
- Prioritize MEANINGFUL illustrations over quantity — each scene should be worth drawing
- Do NOT limit by a fixed number. Split as many scenes as the story content naturally requires. Short or long, let the story decide

## Character Definition Rules:
- List every character that appears more than once
- For each character, give a FIXED visual description: hair color, clothing, distinctive features
- These descriptions MUST be copied verbatim into every scene prompt where that character appears
- Example: if "小女孩" is defined as "a little girl with red hood, brown curly hair, blue dress, white socks", EVERY scene with her MUST include "a little girl with red hood, brown curly hair, blue dress, white socks"

## Prompt Rules:
- Each scene's prompt MUST closely reflect the EXACT content of that scene's story_text
- Describe the scene visually: character poses, actions, facial expressions, environment, lighting, mood
- Always START each prompt with the full character description(s) from the character list
- Always END each prompt with: "children's book illustration, watercolor style, soft colors, whimsical, no text, no letters, no words"
- Keep each prompt under 200 words
- Do NOT include any text, words, letters, or speech bubbles in the image description
- Focus on what can be SEEN, not narrated

You MUST respond in the following JSON format only, no other text:
{
  "title": "story title in Chinese",
  "characters": [
    {
      "name": "角色中文名",
      "visual": "consistent English visual description of this character"
    }
  ],
  "style": "a fixed style prefix that will be prepended to every prompt, describing the overall art style",
  "scenes": [
    {
      "scene_number": 1,
      "story_text": "this scene's story content in Chinese",
      "characters_in_scene": ["角色名1", "角色名2"],
      "prompt": "English image generation prompt (must include the full visual description of each character from the characters list, plus scene action and environment)"
    }
  ]
}"""


def parse_story(story_text: str) -> dict:
    """Use LLM to split story into scenes and generate image prompts."""
    client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=BASE_URL)

    user_message = (
        f"请阅读以下童话故事，按照剧情画面拆分为插图场景，"
        f"每个场景应该是一个有意义的、可以单独画出来的画面。\n\n"
        f"---\n{story_text}"
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=16384,
    )

    content = response.choices[0].message.content.strip()

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON:\n{content}") from e

    if "scenes" not in result or not result["scenes"]:
        raise ValueError(f"LLM response missing 'scenes': {content}")

    style = result.get("style", "children's book illustration, watercolor style, soft colors, whimsical")
    characters = {c["name"]: c["visual"] for c in result.get("characters", [])}

    for scene in result["scenes"]:
        prompt = scene["prompt"]
        for char_name in scene.get("characters_in_scene", []):
            if char_name in characters and characters[char_name] not in prompt:
                prompt = characters[char_name] + ", " + prompt
        if style not in prompt:
            prompt = prompt + ", " + style
        if "no text" not in prompt and "no letters" not in prompt:
            prompt = prompt + ", no text, no letters, no words"
        scene["prompt"] = prompt

    return result
