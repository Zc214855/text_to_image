import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config
import image_generator
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
    def test_scene_count_merges_short_sentences(self):
        story = "。".join(f"第{i}句话" for i in range(1, 23)) + "。"

        self.assertEqual(14, story_parser.estimate_scene_count(story))

    def test_parse_story_builds_one_consistent_prompt(self):
        payload = {
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
                    "scene_number": 9,
                    "story_text": "女孩推开木门。",
                    "characters_in_scene": ["女孩"],
                    "shot": "medium shot at eye level",
                    "visual_action": "the girl cautiously pushes open the wooden door",
                    "environment": "an old cottage entrance",
                    "composition": "the girl is the clear focal point",
                    "lighting": "cool dawn light",
                }
            ],
        }
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload))
                )
            ]
        )
        client = Mock()
        client.chat.completions.create.return_value = response

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
        self.assertEqual(1, prompt.count(payload["style"]))
        self.assertEqual(1, prompt.count(payload["characters"][0]["visual"]))
        self.assertIn("no typography", prompt)
        client.chat.completions.create.assert_called_once()

    def test_parse_story_rejects_undefined_character(self):
        payload = {
            "title": "测试故事",
            "characters": [],
            "style": "watercolor",
            "scenes": [
                {
                    "story_text": "女孩挥手。",
                    "characters_in_scene": ["女孩"],
                    "visual_action": "a girl waves",
                }
            ],
        }
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload))
                )
            ]
        )
        client = Mock()
        client.chat.completions.create.return_value = response

        with (
            patch.object(
                story_parser,
                "get_llm_client_config",
                return_value=("https://example.test/v1", "test-key", "test-model"),
            ),
            patch.object(story_parser, "OpenAI", return_value=client),
        ):
            with self.assertRaisesRegex(ValueError, "undefined character"):
                story_parser.parse_story("女孩挥手。")


class ImageGeneratorTests(unittest.TestCase):
    def test_ark_generate_disables_watermark(self):
        response = Mock()
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


if __name__ == "__main__":
    unittest.main()
