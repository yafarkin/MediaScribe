import json
from pathlib import Path
import tempfile
import unittest

import config
import summarize
import transcribe


class ConfigTests(unittest.TestCase):
    def test_default_file_contains_every_cli_setting(self):
        data = json.loads((Path(__file__).parents[1] / "config.json").read_text())
        shared = {"provider", "base_url", "api_key", "context_length"}
        for section, factory in [("transcribe", transcribe.parser), ("summarize", summarize.parser)]:
            with self.subTest(section=section):
                actions = {action.dest for action in factory()._actions}
                expected = (actions - {"help", "config", "file", "lm_url"}) | {"base_url"}
                self.assertEqual(set(data[section]) | shared, expected)

    def test_config_values_paths_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "transcribe": {
                    "language": "en",
                    "summarize": True,
                    "summary_prompt_file": "prompt.md",
                },
            }))
            _, args = config.parse_args(transcribe.parser, "transcribe", [
                "lecture.wav", "--config", str(path), "--language", "auto", "--no-summarize",
            ])
            self.assertEqual(args.provider, "openai")
            self.assertEqual(args.lm_url, "https://api.openai.com/v1")
            self.assertEqual(args.language, "auto")
            self.assertFalse(args.summarize)
            self.assertEqual(args.summary_prompt_file, (Path(tmp) / "prompt.md").resolve())
            self.assertEqual(args.config, path.resolve())

    def test_partial_summary_config_keeps_default_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({"summarize": {"summary_type": "review"}}))
            _, args = config.parse_args(summarize.parser, "summarize", [
                "transcript.md", "--config", str(path),
            ])
            self.assertEqual(args.model, "qwen/qwen3.5-9b")
            self.assertEqual(args.summary_type, "review")

    def test_summary_main_uses_provider_and_model_from_config(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "transcript.md"
            source.write_text("Source material")
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "summarize": {"model": "gpt-test", "summary_type": "review"},
            }))
            with patch("sys.argv", ["summarize.py", str(source), "--config", str(path)]), \
                    patch.object(summarize, "chat", return_value="# Notes") as chat:
                summarize.main()
            self.assertEqual(source.with_name("summary.md").read_text(), "# Notes\n")
            self.assertEqual(chat.call_args.args[1], "gpt-test")
            self.assertEqual(chat.call_args.kwargs["provider"], "openai")
            self.assertEqual(chat.call_args.kwargs["context_length"], 32768)

    def test_unknown_and_invalid_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps({"transcribe": {"unknown": 1}}))
            with self.assertRaisesRegex(ValueError, "unknown"):
                config.parse_args(transcribe.parser, "transcribe", ["lecture.wav", "--config", str(path)])
            path.write_text(json.dumps({"transcribe": {"slides": "yes"}}))
            with self.assertRaisesRegex(ValueError, "true или false"):
                config.parse_args(transcribe.parser, "transcribe", ["lecture.wav", "--config", str(path)])


if __name__ == "__main__":
    unittest.main()
