import os
import re

from config import OUTPUT_DIR

WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_story_title(title: str) -> str:
    """将故事名转换为可安全用作 Windows 文件夹名的文本。"""
    safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title.strip())
    safe_title = re.sub(r"\s+", " ", safe_title).rstrip(" .")
    safe_title = safe_title[:80].rstrip(" .") or "未命名故事"
    if safe_title.upper() in WINDOWS_RESERVED_NAMES:
        safe_title = f"{safe_title}_故事"
    return safe_title


def create_story_output_dir(title: str, output_root: str = OUTPUT_DIR) -> str:
    """创建故事输出目录；同名目录已存在时追加数字，避免覆盖历史结果。"""
    os.makedirs(output_root, exist_ok=True)
    folder_name = sanitize_story_title(title)

    for suffix in range(1, 10_000):
        candidate_name = folder_name if suffix == 1 else f"{folder_name}_{suffix}"
        candidate_path = os.path.join(output_root, candidate_name)
        try:
            os.mkdir(candidate_path)
            return candidate_path
        except FileExistsError:
            continue

    raise RuntimeError(f"无法为故事创建唯一输出目录：{folder_name}")
