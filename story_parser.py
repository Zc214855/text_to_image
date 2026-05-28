import json
from openai import OpenAI
from config import get_llm_client_config, LLM_MODEL

SYSTEM_PROMPT = """You are a fairy tale illustration assistant. Your job is to:
1. Read the given fairy tale text
2. Split it into VERY detailed scenes — approximately ONE scene per 30-50 Chinese characters. Generate as many scenes as possible so that nearly every sentence gets its own illustration. DO NOT skip or merge story content.
3. First define ALL recurring characters with consistent visual descriptions
4. For each scene, write an English image generation prompt that MUST reuse the exact same character descriptions and MUST closely match the specific story content of that scene

## Character Definition Rules:
- List every character that appears more than once
- For each character, give a FIXED visual description: hair color, clothing, distinctive features
- These descriptions MUST be copied verbatim into every scene prompt where that character appears
- Example: if "小女孩" is defined as "a little girl with red hood, brown curly hair, blue dress, white socks", EVERY scene with her MUST include "a little girl with red hood, brown curly hair, blue dress, white socks"

## Prompt Rules:
- Each scene's prompt MUST closely reflect the EXACT content of that scene's story_text — if the text says the girl picks flowers, show her picking flowers; if the wolf knocks on the door, show the wolf knocking on the door
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
    base_url, api_key = get_llm_client_config()
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Estimate desired scene count: ~1 scene per 40 chars, more illustrations
    char_count = len(story_text)
    target_scenes = max(6, min(50, char_count // 40))

    user_message = (
        f"请将以下童话故事拆分为约 {target_scenes} 个场景"
        f"（故事约{char_count}字，平均每场景约40字，尽量每句话一个场景，不要遗漏任何内容）。\n\n"
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

    # Extract JSON from response (handle possible markdown code blocks)
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

    # Prepend the shared style prefix to every scene prompt
    style = result.get("style", "children's book illustration, watercolor style, soft colors, whimsical")
    characters = {c["name"]: c["visual"] for c in result.get("characters", [])}

    for scene in result["scenes"]:
        prompt = scene["prompt"]
        # Ensure character visuals are in the prompt — if LLM forgot, inject them
        for char_name in scene.get("characters_in_scene", []):
            if char_name in characters and characters[char_name] not in prompt:
                prompt = characters[char_name] + ", " + prompt
        # Ensure style suffix
        if style not in prompt:
            prompt = prompt + ", " + style
        # Ensure no-text suffix
        if "no text" not in prompt and "no letters" not in prompt:
            prompt = prompt + ", no text, no letters, no words"
        scene["prompt"] = prompt

    return result
