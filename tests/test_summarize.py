import argparse
import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lmstudio
import summarize as app


class SummaryTests(unittest.TestCase):
    def test_stream_progress_and_final_result(self):
        events = [{"type": "prompt_processing.progress", "progress": 0.5},
                  {"type": "message.delta", "content": "Конспект"},
                  {"type": "chat.end", "result": {"output": [
                      {"type": "message", "content": "Конспект"}], "stats": {"total_output_tokens": 4}}}]
        stream = "\n\n".join("event: " + e["type"] + "\ndata: " + json.dumps(e, ensure_ascii=False) for e in events) + "\n\n"
        seen = []
        with patch.object(lmstudio, "open_response", return_value=io.BytesIO(stream.encode())):
            text = lmstudio.chat("http://127.0.0.1:1234/v1", "test", [], on_event=seen.append)
        self.assertEqual(text, "Конспект")
        self.assertEqual(seen, events)

    def test_stream_without_final_event_is_an_error(self):
        stream = b'event: message.delta\ndata: {"content": "partial"}\n\n'
        with patch.object(lmstudio, "open_response", return_value=io.BytesIO(stream)):
            with self.assertRaisesRegex(RuntimeError, "прерван"):
                lmstudio.chat("http://127.0.0.1:1234/v1", "test", [], on_event=lambda e: None)

    def test_interrupted_stream_is_retried_without_duplicate_checkpoint(self):
        args = argparse.Namespace(chunk_chars=1000, lm_url="http://127.0.0.1:1234/v1", model="test",
                                  max_tokens=4096, timeout=60, reasoning="off")
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "summary.parts.json"
            with patch.object(app, "chat", side_effect=[
                    lmstudio.StreamInterruptedError("interrupted"), "Complete notes"]) as chat:
                result = app.summarize("Source text", "en", "Prompt", args, checkpoint,
                                       on_event=events.append)
            self.assertEqual(result, "Complete notes")
            self.assertEqual(chat.call_count, 2)
            self.assertIn({"type": "stage.retry", "stage": "final", "attempt": 2}, events)
            self.assertEqual(len(json.loads(checkpoint.read_text())["parts"]), 1)

    def test_interrupted_stream_stops_after_three_attempts(self):
        args = argparse.Namespace(chunk_chars=1000, lm_url="http://127.0.0.1:1234/v1", model="test",
                                  max_tokens=4096, timeout=60, reasoning="off")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(app, "chat", side_effect=lmstudio.StreamInterruptedError("interrupted")) as chat:
                with self.assertRaises(lmstudio.StreamInterruptedError):
                    app.summarize("Source text", "en", "Prompt", args, Path(tmp) / "summary.parts.json")
            self.assertEqual(chat.call_count, 3)

    def test_stream_error_is_not_saved_as_successful_completion(self):
        stream = b'event: error\ndata: {"error": {"message": "Model unloaded"}}\n\n'
        with patch.object(lmstudio, "open_response", return_value=io.BytesIO(stream)):
            with self.assertRaisesRegex(RuntimeError, "Model unloaded"):
                lmstudio.chat("http://127.0.0.1:1234/v1", "test", [], on_event=lambda e: None)

    def test_chunking_preserves_all_characters_and_bounds_requests(self):
        text = ("Определение.\n\nПример с объяснением. " * 500) + "x" * 4000
        chunks = app.split_text(text, 1000)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(0 < len(c) <= 1000 for c in chunks))

    def test_summary_settings_fit_loaded_model_context(self):
        self.assertEqual(app.fit_summary_context(8192), (3584, 2560))
        self.assertEqual(app.fit_summary_context(32768), (10000, 4096))
        self.assertEqual(app.fit_summary_context(8192, 3000, 3000), (3000, 3000))
        with self.assertRaisesRegex(ValueError, "не помещаются"):
            app.fit_summary_context(8192, 10000, 4096)

    def test_reads_context_from_loaded_model_instance(self):
        response = {"models": [{"key": "test", "loaded_instances": [
            {"config": {"context_length": 8192}}]}]}
        with patch.object(lmstudio, "request_json", return_value=response) as request:
            context = lmstudio.model_context_length("http://127.0.0.1:1234/v1", "test")
        self.assertEqual(context, 8192)
        self.assertEqual(request.call_args.args, ("http://127.0.0.1:1234", "/api/v1/models"))

    def test_loads_model_with_requested_context(self):
        catalog = {"models": [{"key": "test", "max_context_length": 262144, "loaded_instances": []}]}
        loaded = {"instance_id": "test:2", "load_config": {"context_length": 32768}}
        with patch.object(lmstudio, "request_json", side_effect=[catalog, loaded]) as request:
            instance = lmstudio.load_model("http://127.0.0.1:1234/v1", "test", 32768)
        self.assertEqual(instance, "test:2")
        self.assertEqual(request.call_args_list[1].args[1], "/api/v1/models/load")
        self.assertEqual(request.call_args_list[1].args[2]["context_length"], 32768)

    def test_model_with_requested_context_is_not_reloaded(self):
        catalog = {"models": [{"key": "test", "max_context_length": 262144, "loaded_instances": [
            {"id": "test:2", "config": {"context_length": 32768}}]}]}
        with patch.object(lmstudio, "request_json", return_value=catalog) as request:
            instance = lmstudio.load_model("http://127.0.0.1:1234/v1", "test", 32768)
        self.assertEqual(instance, "test:2")
        self.assertEqual(request.call_count, 1)

    def test_requested_context_is_checked_against_model_limit(self):
        info = {"max_context_length": 8192}
        with self.assertRaisesRegex(ValueError, "превышает максимум"):
            lmstudio.validate_context_length(info, "test", 32768)

    def test_json_input_preserves_language_and_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "raw-transcript.json"
            source.write_text(json.dumps({"language": "en", "segments": [
                {"start": 60, "end": 64, "text": "A strategic decision affects other agents."}]}))
            _, text, language = app.read_transcript(source)
            self.assertEqual(language, "en")
            self.assertIn("[00:01:00]", text)
            self.assertIn("strategic decision", text)

    def test_translated_transcript_uses_output_language_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "transcript.md").write_text("Translated text")
            (folder / "transcript-original.md").write_text("Исходный текст")
            (folder / "raw-transcript.json").write_text(json.dumps({
                "language": "ru", "output_language": "en", "segments": []}))
            _, _, translated_language = app.read_transcript(folder / "transcript.md")
            _, _, original_language = app.read_transcript(folder / "transcript-original.md")
            self.assertEqual(translated_language, "en")
            self.assertEqual(original_language, "ru")

    def test_requested_output_language_is_used_at_every_summary_stage(self):
        args = argparse.Namespace(chunk_chars=1000, lm_url="http://127.0.0.1:1234/v1", model="test",
                                  max_tokens=4096, timeout=60, reasoning="off")

        def fake_chat(url, model, messages, **kwargs):
            self.assertIn("Required output language: de", messages[1]["content"])
            self.assertIn("Output in de", messages[0]["content"])
            self.assertNotIn("Output in the original transcript's language", messages[0]["content"])
            return "Kurze Notiz."

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "summary.parts.json"
            with patch.object(app, "chat", fake_chat):
                app.summarize("An example. " * 300, "en", "Study notes", args, checkpoint,
                              output_language="de")
            self.assertEqual(json.loads(checkpoint.read_text())["output_language"], "de")

    def test_every_stage_keeps_original_language_and_records_all_parts(self):
        args = argparse.Namespace(chunk_chars=1000, lm_url="http://127.0.0.1:1234/v1", model="test", max_tokens=4096, timeout=60, reasoning="off")
        calls = []
        def fake_chat(url, model, messages, **kwargs):
            self.assertIn("language code: en", messages[1]["content"])
            self.assertIn("original transcript's language", messages[0]["content"])
            calls.append(messages)
            return "# Lecture notes\n\nA specific fact."
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "summary.parts.json"
            with patch.object(app, "chat", fake_chat):
                result = app.summarize("An example. " * 300, "en", "Study notes", args, checkpoint)
            self.assertIn("Lecture notes", result)
            records = json.loads(checkpoint.read_text())["parts"]
            self.assertEqual([r["stage"] for r in records], ["part"] * 4 + ["final"])
            self.assertEqual(len(calls), 5)

    def test_successful_parts_survive_later_model_failure(self):
        args = argparse.Namespace(chunk_chars=1000, lm_url="http://127.0.0.1:1234/v1", model="test", max_tokens=4096, timeout=60, reasoning="off")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "summary.parts.json"
            with patch.object(app, "chat", side_effect=["Заметки первой части.", RuntimeError("Server stopped")]):
                with self.assertRaisesRegex(RuntimeError, "Server stopped"):
                    app.summarize("Материал лекции. " * 200, "ru", "Конспект", args, checkpoint)
            self.assertEqual(len(json.loads(checkpoint.read_text())["parts"]), 1)

    def test_refuses_to_overwrite_transcript_even_with_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "transcript.md"
            source.write_text("Source text")
            with patch("sys.argv", ["summarize.py", str(source), "--model", "test", "--output", str(source), "--force"]):
                with self.assertRaises(SystemExit):
                    app.main()
            self.assertEqual(source.read_text(), "Source text")

    def test_truncated_response_is_reported_instead_of_saved_as_complete(self):
        with patch.object(lmstudio, "request_json", return_value={"output": [
            {"type": "message", "content": "Unfinished notes"}], "stats": {"total_output_tokens": 4096}}):
            with self.assertRaisesRegex(RuntimeError, "обрезан"):
                lmstudio.chat("http://127.0.0.1:1234/v1", "test", [])

    def test_native_api_disables_thinking_and_excludes_reasoning_from_output(self):
        response = {"output": [{"type": "reasoning", "content": "Internal notes"},
                               {"type": "message", "content": "Lecture summary"}],
                    "stats": {"total_output_tokens": 30}}
        with patch.object(lmstudio, "request_json", return_value=response) as request:
            result = lmstudio.chat("http://127.0.0.1:1234/v1", "test", [
                {"role": "system", "content": "Instructions"}, {"role": "user", "content": "Source"}])
        self.assertEqual(result, "Lecture summary")
        base, endpoint, payload = request.call_args.args
        self.assertEqual(base, "http://127.0.0.1:1234")
        self.assertEqual(endpoint, "/api/v1/chat")
        self.assertEqual(payload["reasoning"], "off")
        self.assertFalse(payload["store"])

    def test_hierarchical_reduction_finishes_and_keeps_intermediate_notes(self):
        args = argparse.Namespace(chunk_chars=1000, lm_url="http://127.0.0.1:1234/v1", model="test", max_tokens=4096, timeout=60, reasoning="off")
        def fake_chat(url, model, messages, **kwargs):
            task = messages[1]["content"]
            if "Task: Extract" in task:
                self.assertIn("full multi-section outline above applies ONLY to the final", messages[0]["content"])
                return "Detailed fact. " * 50
            if "Task: Merge" in task:
                return "Compact fact. " * 10
            return "Final study notes."
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "notes.json"
            with patch.object(app, "chat", fake_chat):
                result = app.summarize("Lecture material. " * 200, "en", "Prompt", args, checkpoint)
            self.assertEqual(result, "Final study notes.")
            stages = [r["stage"] for r in json.loads(checkpoint.read_text())["parts"]]
            self.assertIn("reduce", stages)
            self.assertEqual(stages[-1], "final")


if __name__ == "__main__":
    unittest.main()
