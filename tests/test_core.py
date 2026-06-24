import gc
import json
import os
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

import config
import image_generator
import generation_tasks
import main
import output_manager
import story_parser


class ConfigTests(unittest.TestCase):
    def test_zhipu_is_configured_as_default_llm(self):
        self.assertEqual("zhipu", config.LLM_PROVIDER)
        self.assertEqual("glm-5.1", config.LLM_MODEL)

    def test_llm_provider_config_is_complete(self):
        for provider in config.LLM_PROVIDERS:
            with self.subTest(provider=provider):
                provider_config = config.get_llm_provider_config(provider)
                self.assertTrue(provider_config["label"])
                self.assertTrue(provider_config["base_url"])
                self.assertTrue(provider_config["model"])

    def test_get_model_config_uses_requested_model(self):
        model = "Tongyi-MAI/Z-Image-Turbo"

        model_config = config.get_model_config(model)

        self.assertEqual("siliconflow", model_config["provider"])
        self.assertEqual(10, model_config["num_inference_steps"])

    def test_volcengine_model_is_registered(self):
        model_config = config.get_model_config("doubao-seedream-5-0-260128")

        self.assertEqual("volcengine", model_config["provider"])
        self.assertIn("1728x2304", model_config["image_sizes"])

    def test_all_models_have_ui_documentation(self):
        required_fields = {"label", "provider", "image_sizes", "price", "summary"}

        for model_id, model_config in config.ALL_MODELS.items():
            with self.subTest(model=model_id):
                self.assertTrue(required_fields.issubset(model_config))


class StoryParserTests(unittest.TestCase):
    def test_parse_llm_json_repairs_unescaped_dialogue_quotes(self):
        malformed_json = (
            '{"title":"月亮灯笼","scenes":[{"story_text":'
            '"阿圆问奶奶："月亮是不是迷路了？"奶奶回答。"}]}'
        )

        result = story_parser.parse_llm_json(malformed_json)

        self.assertEqual("月亮灯笼", result["title"])
        self.assertIn("月亮是不是迷路了", result["scenes"][0]["story_text"])

    def test_scene_count_merges_short_sentences(self):
        story = "。".join(f"第{i}句话" for i in range(1, 23)) + "。"

        self.assertEqual(14, story_parser.estimate_scene_count(story))

    def test_parse_story_builds_one_consistent_prompt(self):
        # 第一轮：大纲响应（无视觉细节字段）
        outline_payload = {
            "title": "测试故事",
            "characters": [
                {
                    "name": "女孩",
                    "visual": "a six-year-old girl with black braids and a yellow coat",
                }
            ],
            "style": "watercolor picture book, muted blue and gold palette",
            "scenes": [
                {
                    "scene_number": 1,
                    "story_text": "女孩推开木门。",
                    "characters_in_scene": ["女孩"],
                }
            ],
        }
        # 第二轮：细节填充响应
        detail_payload = {
            "scenes": [
                {
                    "scene_number": 1,
                    "story_text": "女孩推开木门。",
                    "characters_in_scene": ["女孩"],
                    "shot": "medium shot at eye level",
                    "visual_action": "the girl cautiously pushes open the wooden door",
                    "environment": "an old cottage entrance",
                    "composition": "the girl is the clear focal point",
                    "lighting": "cool dawn light",
                    "state_tracking": "early morning, outdoors",
                }
            ],
        }
        outline_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(outline_payload))
                )
            ]
        )
        detail_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(detail_payload))
                )
            ]
        )
        client = Mock()
        client.chat.completions.create.side_effect = [
            outline_response,
            detail_response,
        ]

        with (
            patch.object(
                story_parser,
                "get_llm_client_config",
                return_value=("https://example.test/v1", "test-key", "test-model"),
            ),
            patch.object(story_parser, "OpenAI", return_value=client),
        ):
            result = story_parser.parse_story("女孩推开木门。")

        prompt = result["scenes"][0]["prompt"]
        self.assertEqual(1, result["scenes"][0]["scene_number"])
        self.assertEqual(1, prompt.count(outline_payload["style"]))
        self.assertEqual(1, prompt.count(outline_payload["characters"][0]["visual"]))
        self.assertIn("no typography", prompt)
        # 两轮共调用 LLM 两次
        self.assertEqual(2, client.chat.completions.create.call_count)

    def test_parse_story_skips_undefined_character(self):
        # 第一轮大纲：无角色定义
        outline_payload = {
            "title": "测试故事",
            "characters": [],
            "style": "watercolor",
            "scenes": [
                {
                    "scene_number": 1,
                    "story_text": "女孩挥手。",
                    "characters_in_scene": ["女孩"],
                }
            ],
        }
        # 第二轮细节：角色不在定义表中但仍被引用
        detail_payload = {
            "scenes": [
                {
                    "scene_number": 1,
                    "story_text": "女孩挥手。",
                    "characters_in_scene": ["女孩"],
                    "shot": "medium shot",
                    "visual_action": "a girl waves her hand",
                    "environment": "a garden",
                    "composition": "girl in center",
                    "lighting": "morning sun",
                    "state_tracking": "daytime",
                }
            ],
        }
        outline_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(outline_payload))
                )
            ]
        )
        detail_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(detail_payload))
                )
            ]
        )
        client = Mock()
        client.chat.completions.create.side_effect = [
            outline_response,
            detail_response,
        ]

        with (
            patch.object(
                story_parser,
                "get_llm_client_config",
                return_value=("https://example.test/v1", "test-key", "test-model"),
            ),
            patch.object(story_parser, "OpenAI", return_value=client),
        ):
            result = story_parser.parse_story("女孩挥手。")

        # 角色未定义时不应硬失败，场景正常生成但不含角色引用
        prompt = result["scenes"][0]["prompt"]
        self.assertNotIn("女孩", prompt.split("; ")[0] if "; " in prompt else "")


class ImageGeneratorTests(unittest.TestCase):
    def test_ark_generate_disables_watermark(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"data": [{"url": "https://example.test/a.png"}]}
        response.raise_for_status.return_value = None

        with patch.object(image_generator.httpx, "post", return_value=response) as post:
            url = image_generator._ark_generate(
                "test prompt",
                "doubao-seedream-5-0-260128",
                "1728x2304",
            )

        self.assertEqual("https://example.test/a.png", url)
        payload = post.call_args.kwargs["json"]
        self.assertFalse(payload["watermark"])
        self.assertEqual("png", payload["output_format"])
        self.assertEqual("disabled", payload["sequential_image_generation"])

    def test_ark_seedream_4_requests_png_output(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"data": [{"url": "https://example.test/a.png"}]}
        response.raise_for_status.return_value = None

        with patch.object(image_generator.httpx, "post", return_value=response) as post:
            image_generator._ark_generate(
                "test prompt",
                "doubao-seedream-4-0-250828",
                "864x1152",
            )

        self.assertEqual("png", post.call_args.kwargs["json"]["output_format"])

    def test_download_image_rejects_non_image_response(self):
        response = Mock()
        response.headers = {"content-type": "application/json"}
        response.content = b'{"error":"expired"}'
        response.raise_for_status.return_value = None

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(image_generator.httpx, "get", return_value=response),
        ):
            save_path = os.path.join(directory, "scene.png")
            with self.assertRaisesRegex(ValueError, "not an image"):
                image_generator.download_image(
                    "https://example.test/expired", save_path
                )
            self.assertFalse(os.path.exists(save_path))

    def test_generate_and_save_uses_story_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    image_generator,
                    "generate_image",
                    return_value="https://example.test/scene.png",
                ),
                patch.object(
                    image_generator,
                    "download_image",
                    return_value=os.path.join(directory, "scene_01.png"),
                ) as download,
            ):
                path = image_generator.generate_and_save(
                    "prompt",
                    "scene_01.png",
                    output_dir=directory,
                )

        expected_path = os.path.join(directory, "scene_01.png")
        self.assertEqual(expected_path, path)
        download.assert_called_once_with(
            "https://example.test/scene.png",
            expected_path,
        )

    def test_download_retry_does_not_regenerate_image(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                image_generator,
                "generate_image",
                return_value="https://example.test/scene.png",
            ) as generate,
            patch.object(
                image_generator,
                "download_image",
                side_effect=[
                    httpx.ReadTimeout("download timeout"),
                    os.path.join(directory, "scene.png"),
                ],
            ) as download,
        ):
            result = image_generator.generate_and_save(
                "prompt",
                "scene.png",
                output_dir=directory,
                model="Kwai-Kolors/Kolors",
                size="1024x1024",
                sleep_fn=lambda _: None,
            )

        self.assertEqual(os.path.join(directory, "scene.png"), result)
        self.assertEqual(1, generate.call_count)
        self.assertEqual(2, download.call_count)


class OutputManagerTests(unittest.TestCase):
    def test_sanitize_story_title_replaces_windows_invalid_characters(self):
        title = ' 小红帽：森林/冒险?* '

        safe_title = output_manager.sanitize_story_title(title)

        self.assertEqual("小红帽：森林_冒险__", safe_title)

    def test_sanitize_story_title_handles_reserved_name(self):
        self.assertEqual("CON_故事", output_manager.sanitize_story_title("CON"))

    def test_create_story_output_dir_avoids_overwriting_previous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            first = output_manager.create_story_output_dir("小红帽", directory)
            second = output_manager.create_story_output_dir("小红帽", directory)

        self.assertEqual("小红帽", os.path.basename(first))
        self.assertEqual("小红帽_2", os.path.basename(second))


class GenerationTaskTests(unittest.TestCase):
    def test_find_latest_failed_task(self):
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "故事一")
            second = os.path.join(directory, "故事二")
            os.makedirs(first)
            os.makedirs(second)
            generation_tasks.save_task(
                first,
                {
                    "status": "partial",
                    "scene_count": 1,
                    "successful_scenes": [],
                    "failed_scenes": [{"scene_number": 1, "error": "timeout"}],
                },
            )
            generation_tasks.save_task(
                second,
                {
                    "status": "completed",
                    "scene_count": 0,
                    "successful_scenes": [],
                    "failed_scenes": [],
                },
            )

            latest = generation_tasks.find_latest_failed_task(directory)

        self.assertEqual(first, latest)

    def test_find_latest_failed_task_includes_interrupted_task(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = os.path.join(directory, "中断任务")
            os.makedirs(task_dir)
            generation_tasks.save_task(
                task_dir,
                {
                    "status": "running",
                    "scene_count": 3,
                    "successful_scenes": [1],
                    "failed_scenes": [],
                    "image_extension": ".png",
                },
            )
            with open(os.path.join(task_dir, "scene_01.png"), "wb") as file:
                file.write(b"image")

            latest = generation_tasks.find_latest_failed_task(directory)

        self.assertEqual(task_dir, latest)


class ImageRetryWorkflowTests(unittest.TestCase):
    def test_generation_button_updates_only_change_interactive_state(self):
        disabled = main.set_generation_buttons_enabled(False)
        enabled = main.set_generation_buttons_enabled(True)

        self.assertEqual(
            (
                {"interactive": False, "__type__": "update"},
                {"interactive": False, "__type__": "update"},
            ),
            disabled,
        )
        self.assertEqual(
            (
                {"interactive": True, "__type__": "update"},
                {"interactive": True, "__type__": "update"},
            ),
            enabled,
        )

    def test_generate_and_retry_restore_buttons_after_completion(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            app = main.build_ui()
            try:
                for target in (main.generate, main.retry_failed_images):
                    event_id, event = next(
                        (event_id, event)
                        for event_id, event in app.fns.items()
                        if event.fn is target
                    )
                    disable_event = app.fns[event.trigger_after]
                    restore_events = [
                        candidate
                        for candidate in app.fns.values()
                        if candidate.trigger_after == event_id
                    ]

                    self.assertFalse(disable_event.queue)
                    self.assertEqual(1, event.concurrency_limit)
                    self.assertEqual("image-generation", event.concurrency_id)
                    self.assertEqual(1, len(restore_events))
                    self.assertFalse(restore_events[0].queue)
                    self.assertFalse(restore_events[0].trigger_only_on_success)
            finally:
                app.close()
                del app
                gc.collect()

    def test_rejected_server_request_is_retried_until_success(self):
        request = httpx.Request("POST", "https://example.test")
        server_error = httpx.HTTPStatusError(
            "service unavailable",
            request=request,
            response=httpx.Response(503, request=request),
        )
        with patch.object(
            main,
            "generate_and_save",
            side_effect=[
                server_error,
                server_error,
                "scene.png",
            ],
        ) as generate:
            path, attempts, hit_limit = main.generate_scene_with_retry(
                "prompt",
                "scene.png",
                "output",
                "Kwai-Kolors/Kolors",
                "1024x1024",
                sleep_fn=lambda _: None,
            )

        self.assertEqual("scene.png", path)
        self.assertEqual(3, attempts)
        self.assertEqual(3, generate.call_count)

    def test_transport_timeout_is_not_resubmitted(self):
        with patch.object(
            main,
            "generate_and_save",
            side_effect=httpx.ReadTimeout("response timeout"),
        ) as generate:
            with self.assertRaisesRegex(RuntimeError, "尝试 1/4"):
                main.generate_scene_with_retry(
                    "prompt",
                    "scene.png",
                    "output",
                    "Kwai-Kolors/Kolors",
                    "1024x1024",
                    sleep_fn=lambda _: None,
                )

        self.assertEqual(1, generate.call_count)

    def test_generation_uses_explicit_model_and_size(self):
        observed = {}

        def fake_generate(*args, **kwargs):
            observed.update(kwargs)
            return "scene.png"

        with (
            patch.object(config, "IMAGE_MODEL", "doubao-seedream-5-0-260128"),
            patch.object(config, "IMAGE_SIZE", "1728x2304"),
            patch.object(main, "generate_and_save", side_effect=fake_generate),
        ):
            main.generate_scene_with_retry(
                "prompt",
                "scene.png",
                "output",
                "Kwai-Kolors/Kolors",
                "1024x1024",
            )

        self.assertEqual("Kwai-Kolors/Kolors", observed["model"])
        self.assertEqual("1024x1024", observed["size"])

    def test_permanent_http_error_is_not_retried(self):
        request = httpx.Request("POST", "https://example.test")
        response = httpx.Response(400, request=request)
        error = httpx.HTTPStatusError(
            "bad request",
            request=request,
            response=response,
        )

        with patch.object(main, "generate_and_save", side_effect=error) as generate:
            with self.assertRaisesRegex(RuntimeError, "尝试 1/4"):
                main.generate_scene_with_retry(
                    "prompt",
                    "scene.png",
                    "output",
                    "Kwai-Kolors/Kolors",
                    "1024x1024",
                    sleep_fn=lambda _: None,
                )

        self.assertEqual(1, generate.call_count)

    def test_retry_failed_images_uses_saved_prompts_without_llm(self):
        with tempfile.TemporaryDirectory() as task_dir:
            prompts = {
                "scenes": [
                    {"scene_number": 1, "prompt": "scene one"},
                    {"scene_number": 2, "prompt": "scene two"},
                ]
            }
            with open(
                os.path.join(task_dir, "prompts.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(prompts, file)
            with open(os.path.join(task_dir, "scene_01.png"), "wb") as file:
                file.write(b"image")
            generation_tasks.save_task(
                task_dir,
                {
                    "title": "测试",
                    "image_model": "Kwai-Kolors/Kolors",
                    "image_size": "1024x1024",
                    "image_extension": ".png",
                    "scene_count": 2,
                    "successful_scenes": [1],
                    "failed_scenes": [
                        {"scene_number": 2, "error": "timeout"}
                    ],
                    "status": "partial",
                },
            )

            def fake_generate(
                prompt,
                filename,
                output_dir,
                image_model,
                image_size,
            ):
                self.assertEqual("Kwai-Kolors/Kolors", image_model)
                self.assertEqual("1024x1024", image_size)
                path = os.path.join(output_dir, filename)
                with open(path, "wb") as file:
                    file.write(b"image")
                return path, 1, False

            with (
                patch.object(main, "generate_scene_with_retry", side_effect=fake_generate),
                patch.object(main, "parse_story") as parse_story,
            ):
                results = list(
                    main.retry_failed_images(
                        task_dir,
                        progress=lambda *args, **kwargs: None,
                    )
                )

            task = generation_tasks.load_task(task_dir)

        self.assertEqual(1, len(results))
        self.assertIn("未调用 LLM", results[0][0])
        self.assertEqual("completed", task["status"])
        self.assertEqual([1, 2], task["successful_scenes"])
        self.assertEqual([], task["failed_scenes"])
        parse_story.assert_not_called()

    def test_retry_refuses_task_that_is_currently_running(self):
        with tempfile.TemporaryDirectory() as task_dir:
            with generation_tasks.task_execution_lock(task_dir):
                results = list(
                    main.retry_failed_images(
                        task_dir,
                        progress=lambda *args, **kwargs: None,
                    )
                )

        self.assertIn("正在生成图片", results[0][0])


if __name__ == "__main__":
    unittest.main()
