import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

TASK_FILENAME = "generation.json"
PROMPTS_FILENAME = "prompts.json"
_TASK_LOCKS: dict[str, threading.Lock] = {}
_TASK_LOCKS_GUARD = threading.Lock()


class TaskAlreadyRunningError(RuntimeError):
    """同一任务已有生成或重试流程正在执行。"""


@contextmanager
def task_execution_lock(task_dir: str, blocking: bool = True):
    """为任务目录提供进程内互斥，防止生成和重试同时修改同一任务。"""
    normalized_dir = os.path.normcase(os.path.abspath(task_dir))
    with _TASK_LOCKS_GUARD:
        lock = _TASK_LOCKS.setdefault(normalized_dir, threading.Lock())

    acquired = lock.acquire(blocking=blocking)
    if not acquired:
        raise TaskAlreadyRunningError("该任务正在生成图片，请等待当前任务结束")
    try:
        yield
    finally:
        lock.release()


def save_task(task_dir: str, task: dict) -> str:
    """原子写入生成任务状态，避免进程中断留下半个 JSON 文件。"""
    task["updated_at"] = datetime.now(timezone.utc).isoformat()
    task_path = os.path.join(task_dir, TASK_FILENAME)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=task_dir,
            prefix=f"{TASK_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = file.name
            json.dump(task, file, ensure_ascii=False, indent=2)
            # 刷盘后再 replace，防止断电导致目录项已更新但文件内容缺失
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, task_path)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                # 清理失败不应掩盖原始写入异常
                pass
    return task_path


def load_task(task_dir: str) -> dict:
    task_path = os.path.join(task_dir, TASK_FILENAME)
    with open(task_path, "r", encoding="utf-8") as file:
        task = json.load(file)
    if not isinstance(task, dict):
        raise ValueError(f"任务文件格式错误：{task_path}")
    return task


def load_prompts(task_dir: str) -> dict:
    prompts_path = os.path.join(task_dir, PROMPTS_FILENAME)
    with open(prompts_path, "r", encoding="utf-8") as file:
        result = json.load(file)
    if not isinstance(result.get("scenes"), list):
        raise ValueError(f"提示词文件缺少 scenes：{prompts_path}")
    return result


def find_latest_failed_task(output_root: str) -> str | None:
    """查找最近更新且仍有失败场景的任务，支持页面刷新后继续。"""
    if not os.path.isdir(output_root):
        return None

    candidates = []
    with os.scandir(output_root) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            task_path = os.path.join(entry.path, TASK_FILENAME)
            if not os.path.isfile(task_path):
                continue
            try:
                task = load_task(entry.path)
            except (OSError, ValueError):
                continue
            successful = set(task.get("successful_scenes", []))
            scene_count = task.get("scene_count", 0)
            extension = task.get("image_extension", ".png")
            has_missing_file = any(
                not any(
                    os.path.isfile(
                        os.path.join(entry.path, f"scene_{number:02d}{ext}")
                    )
                    for ext in (extension, ".png", ".jpg", ".webp")
                )
                for number in successful
            )
            is_incomplete = (
                bool(task.get("failed_scenes"))
                or task.get("status") != "completed"
                or len(successful) < scene_count
                or has_missing_file
            )
            if is_incomplete:
                candidates.append((os.path.getmtime(task_path), entry.path))

    return max(candidates, default=(0, None))[1]


def collect_generated_images(
    task_dir: str,
    scene_numbers: list[int],
    extension: str = ".png",
) -> list[str]:
    """收集已生成的场景图片，自动匹配 .png/.jpg/.webp 扩展名。"""
    images = []
    for scene_number in sorted(scene_numbers):
        found = None
        for ext in (".png", ".jpg", ".webp", ".gif", extension):
            path = os.path.join(task_dir, f"scene_{scene_number:02d}{ext}")
            if os.path.isfile(path):
                found = path
                break
        if found:
            images.append(found)
    return images
