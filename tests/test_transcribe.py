import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stderr
from unittest.mock import MagicMock, patch

import transcribe as app


class TranscriptTests(unittest.TestCase):
    def recognition_args(self, tmp):
        return argparse.Namespace(model=Path(tmp), language="auto", prompt="", device="cpu")

    def assert_safe_decoding_options(self, kwargs):
        self.assertEqual(kwargs["temperature"], (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        self.assertFalse(kwargs["condition_on_previous_text"])

    def test_mlx_decoding_does_not_reinforce_repetition_loops(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = MagicMock()
            module.transcribe.return_value = {"language": "en", "segments": []}
            with patch.object(app, "backend_name", return_value="mlx"), \
                    patch.dict(sys.modules, {"mlx_whisper": module}):
                app.recognize(Path(tmp) / "audio.wav", self.recognition_args(tmp))
            self.assert_safe_decoding_options(module.transcribe.call_args.kwargs)

    def test_faster_decoding_does_not_reinforce_repetition_loops(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = MagicMock()
            instance = module.WhisperModel.return_value
            instance.transcribe.return_value = (iter([]), MagicMock(language="en"))
            with patch.object(app, "backend_name", return_value="faster"), \
                    patch.dict(sys.modules, {"faster_whisper": module}):
                app.recognize(Path(tmp) / "audio.wav", self.recognition_args(tmp))
            self.assert_safe_decoding_options(instance.transcribe.call_args.kwargs)

    def test_language_is_detected_automatically_by_default(self):
        args = app.parser().parse_args(["lecture.wav"])
        self.assertEqual(args.language, "auto")

    def test_preflight_rejects_missing_llm_before_processing(self):
        args = argparse.Namespace(reuse_transcript=True, slides=False, pptx=False, pdf=False,
                                  speakers=False, output_language="ru", translation_model="missing",
                                  summarize=False, slides_mode="heuristic", vision_model=None,
                                  summary_model=None, lm_url="http://127.0.0.1:1234/v1",
                                  context_length=32768)
        with patch.object(app, "model_info", side_effect=RuntimeError("Модель missing не найдена")):
            with self.assertRaisesRegex(RuntimeError, "не найдена"):
                app.preflight(args)

    def test_summary_child_runs_after_transcript_and_inherits_progress_output(self):
        native_run = subprocess.run
        child_calls = []
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "lecture.wav"
            native_run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=1", str(source)], check=True)
            def run(command, *args, **kwargs):
                if len(command) > 1 and command[1] == str(app.ROOT / "summarize.py"):
                    self.assertTrue(Path(command[2]).is_file())
                    self.assertNotIn("capture_output", kwargs)
                    child_calls.append(command)
                    return subprocess.CompletedProcess(command, 0)
                return native_run(command, *args, **kwargs)
            raw = {"segments": [{"start": 0, "end": 1, "text": "Тест.", "words": []}]}
            argv = ["transcribe.py", str(source), "--no-speakers", "--summarize", "--summary-model", "Local Model",
                    "--summary-type", "review", "--summary-chunk-chars", "6000", "--output-language", "en"]
            translation = json.dumps([{"id": 0, "text": "Test."}])
            with patch("sys.argv", argv), patch.object(app, "preflight"), \
                    patch.object(app, "recognize", return_value=raw), patch.object(app.subprocess, "run", run), \
                    patch.object(app, "load_model", return_value="Local Model"), \
                    patch.object(app, "chat", return_value=translation) as translate:
                app.main()
            self.assertEqual(len(child_calls), 1)
            self.assertEqual(child_calls[0][0], sys.executable)
            self.assertIn("Local Model", child_calls[0])
            self.assertIn("6000", child_calls[0])
            self.assertEqual(child_calls[0][child_calls[0].index("--summary-type") + 1], "review")
            self.assertEqual(child_calls[0][-2:], ["--output-language", "en"])
            self.assertEqual(translate.call_args.args[1], "Local Model")
            self.assertIn("Тест.", (source.with_suffix("") / "transcript-original.md").read_text())

    def test_output_language_saves_original_and_translated_transcripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "lecture.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=1", str(source)], check=True)
            raw = {"language": "ru", "segments": [
                {"start": 0, "end": 1, "text": "Исходная реплика.", "words": []}]}

            def translate(url, model, messages, **kwargs):
                self.assertEqual(model, "Translator")
                self.assertIn("into en", messages[0]["content"])
                items = json.loads(messages[1]["content"])
                return json.dumps([{"id": item["id"], "text": "Original statement."} for item in items])

            argv = ["transcribe.py", str(source), "--no-speakers", "--output-language", "en", "--translation-model", "Translator"]
            with patch("sys.argv", argv), patch.object(app, "preflight"), \
                    patch.object(app, "recognize", return_value=raw), \
                    patch.object(app, "load_model", return_value="Translator"), patch.object(app, "chat", translate):
                app.main()
            folder = source.with_suffix("")
            self.assertIn("Исходная реплика.", (folder / "transcript-original.md").read_text())
            self.assertIn("Original statement.", (folder / "transcript.md").read_text())
            original = json.loads((folder / "transcript-original.json").read_text())
            translated = json.loads((folder / "transcript.json").read_text())
            self.assertEqual(original[0]["text"], "Исходная реплика.")
            self.assertEqual(translated[0]["text"], "Original statement.")
            self.assertEqual(translated[0]["start"], original[0]["start"])
            self.assertEqual(translated[0]["end"], original[0]["end"])
            metadata = json.loads((folder / "raw-transcript.json").read_text())
            self.assertEqual(metadata["language"], "ru")
            self.assertEqual(metadata["output_language"], "en")
            self.assertEqual(metadata["segments"][0]["text"], "Исходная реплика.")

    def test_translation_rejects_changed_utterance_count(self):
        args = argparse.Namespace(lm_url="http://127.0.0.1:1234/v1", translation_model="test",
                                  translation_max_tokens=100, translation_timeout=10,
                                  translation_reasoning="off", translation_chunk_chars=1000)
        items = [{"start": 0, "end": 1, "speaker": "Спикер 1", "text": "Текст"}]
        with patch.object(app, "chat", return_value="[]"):
            with self.assertRaisesRegex(RuntimeError, "число или порядок"):
                app.translate_utterances(items, "en", args)

    def test_translation_splits_incomplete_batch_and_preserves_order(self):
        args = argparse.Namespace(lm_url="http://127.0.0.1:1234/v1", translation_model="test",
                                  translation_max_tokens=100, translation_timeout=10,
                                  translation_reasoning="off", translation_chunk_chars=1000)
        items = [{"start": i, "end": i + 1, "speaker": "Спикер 1", "text": f"Text {i}"}
                 for i in range(3)]
        def translate(url, model, messages, **kwargs):
            source = json.loads(messages[1]["content"])
            if len(source) == 3:
                return "[]"
            return json.dumps([{"id": item["id"], "text": f"Перевод {item['id']}"} for item in source])
        with patch.object(app, "chat", side_effect=translate) as chat, patch("builtins.print") as output:
            result = app.translate_utterances(items, "ru", args)
        self.assertEqual([item["text"] for item in result], ["Перевод 0", "Перевод 1", "Перевод 2"])
        self.assertEqual(chat.call_count, 4)
        messages = [call.args[0] for call in output.call_args_list]
        self.assertTrue(any("изменила число или порядок" in message for message in messages))

    def test_translation_retries_single_utterance_after_invalid_json(self):
        args = argparse.Namespace(lm_url="http://127.0.0.1:1234/v1", translation_model="test",
                                  translation_max_tokens=100, translation_timeout=10,
                                  translation_reasoning="off", translation_chunk_chars=1000)
        items = [{"start": 0, "end": 1, "speaker": "Спикер 1", "text": "Text"}]
        responses = ["not json", json.dumps([{"id": 0, "text": "Текст"}])]
        with patch.object(app, "chat", side_effect=responses) as chat:
            result = app.translate_utterances(items, "ru", args)
        self.assertEqual(result[0]["text"], "Текст")
        self.assertEqual(chat.call_count, 2)

    def test_translation_auto_settings_fit_model_context(self):
        self.assertEqual(app.fit_translation_context(131072), (12000, 8192))
        self.assertEqual(app.fit_translation_context(8192), (2048, 2048))
        self.assertEqual(app.fit_translation_context(32768, 5000, 6000), (5000, 6000))

    def test_translation_reuses_largest_successful_batch_size(self):
        args = argparse.Namespace(lm_url="http://127.0.0.1:1234/v1", translation_model="test",
                                  translation_max_tokens=100, translation_timeout=10,
                                  translation_reasoning="off", translation_chunk_chars=1000)
        items = [{"start": i, "end": i + 1, "speaker": "Спикер 1", "text": f"Text {i}"}
                 for i in range(8)]
        sizes = []
        def translate(url, model, messages, **kwargs):
            source = json.loads(messages[1]["content"])
            sizes.append(len(source))
            if len(source) > 3:
                return "[]"
            return json.dumps([{"id": item["id"], "text": f"Перевод {item['id']}"} for item in source])
        with patch.object(app, "chat", side_effect=translate), patch("builtins.print"):
            result = app.translate_utterances(items, "ru", args)
        self.assertEqual(sizes, [8, 4, 2, 2, 2, 2])
        self.assertEqual([item["text"] for item in result], [f"Перевод {i}" for i in range(8)])

    def test_speaker_change_splits_whisper_segment_at_word(self):
        segments = [{"start": 0, "end": 4, "text": "Привет. Да.", "words": [
            {"start": 0, "end": 1, "word": " Привет."},
            {"start": 2, "end": 3, "word": " Да."}]}]
        turns = [{"start": 0, "end": 1.5, "speaker": "a"},
                 {"start": 1.5, "end": 4, "speaker": "b"}]
        result = app.utterances(segments, turns)
        self.assertEqual([r["speaker"] for r in result], ["Спикер 1", "Спикер 2"])
        self.assertEqual([r["text"] for r in result], ["Привет.", "Да."])

    def test_no_diarization_leaves_speaker_blank(self):
        segments = [{"start": 0, "end": 1, "text": "Первая фраза.", "words": []},
                    {"start": 1, "end": 2, "text": "Вторая фраза.", "words": []}]
        result = app.utterances(segments)
        self.assertEqual([item["speaker"] for item in result], ["", ""])
        self.assertNotIn("Спикер", app.markdown("Test", result, []))

    def test_no_speaker_overlap_is_not_assigned_arbitrarily(self):
        self.assertEqual(app.speaker_for(5, 6, [{"start": 0, "end": 1, "speaker": "a"}]), "unknown")

    def test_markdown_keeps_chronological_slide_links_and_literal_speech(self):
        speech = [{"start": 5, "end": 8, "speaker": "Спикер 1", "text": "Текст [пример]."}]
        slides = [{"time": 2, "path": "slides/0001_00-00-02.jpg"},
                  {"time": 9, "path": "slides/0002_00-00-09.jpg"}]
        text = app.markdown("Лекция", speech, slides)
        self.assertLess(text.index("0001"), text.index("Спикер 1"))
        self.assertLess(text.index("Спикер 1"), text.index("0002"))
        self.assertIn(r"Текст \[пример\].", text)

    def test_remote_vision_endpoint_is_rejected(self):
        for url in ["https://example.com/v1", "http://192.168.1.2:1234/v1", "http://localhost:1234/v1",
                    "http://127.0.0.1.evil.example/v1", "http://user@127.0.0.1:1234/v1"]:
            with self.assertRaises(ValueError):
                app.local_endpoint(url)
        self.assertEqual(app.local_endpoint("http://127.0.0.1:1234/v1/"), "http://127.0.0.1:1234/v1")

    def test_paragraphs_respect_speakers_pauses_and_slides(self):
        speech = [{"start": i * 2, "end": i * 2 + 2, "speaker": "Спикер 1", "text": str(i)} for i in range(5)]
        self.assertEqual(len(app.paragraphs(speech, [], 45)), 1)
        self.assertEqual(len(app.paragraphs(speech, [{"time": 3}], 45)), 2)
        speech[2]["speaker"] = "Спикер 2"
        self.assertEqual(len(app.paragraphs(speech, [], 45)), 3)
        self.assertEqual(" ".join(s["text"] for s in app.paragraphs(speech, [], 3)), "0 1 2 3 4")

    def test_speaker_labels_appear_only_at_start_changes_and_new_slides(self):
        speech = [{"start": i * 2, "end": i * 2 + 2, "speaker": speaker, "text": f"Фраза {i}."}
                  for i, speaker in enumerate(["Спикер 1", "Спикер 1", "Спикер 2", "Спикер 2", "Спикер 1", "Спикер 1"])]
        for mode in ["segment", "start", "paragraph", "interval", "none"]:
            text = app.markdown("Test", speech, [], timestamps=mode)
            self.assertEqual(text.count("Спикер 1:"), 2, mode)
            self.assertEqual(text.count("Спикер 2:"), 1, mode)
            for i in range(6):
                self.assertIn(f"Фраза {i}.", text)
        text = app.markdown("Test", speech, [{"time": 10, "path": "slides/a.jpg"}])
        self.assertEqual(text.count("Спикер 1:"), 3)

    def test_timestamp_modes_and_none_hides_slide_times(self):
        speech = [{"start": i * 20, "end": i * 20 + 20, "speaker": "Спикер 1", "text": f"Фраза {i}."} for i in range(4)]
        slides = [{"time": 5, "path": "slides/a.jpg"}]
        segment = app.markdown("Lecture", speech, [], timestamps="segment")
        paragraph = app.markdown("Lecture", speech, [], timestamps="paragraph")
        start = app.markdown("Lecture", speech, [], timestamps="start")
        interval = app.markdown("Lecture", speech, [], timestamps="interval", paragraph_seconds=20, timestamp_interval=60)
        none = app.markdown("Lecture", speech, slides, timestamps="none")
        self.assertEqual(segment.count("**["), 4)
        self.assertEqual(paragraph.count("**["), 2)
        self.assertIn("[00:00:00]", start)
        self.assertNotIn(" – ", start)
        self.assertEqual(interval.count("**["), 2)
        self.assertNotIn("00:", none)
        self.assertIn("slides/a.jpg", none)
        for i in range(4):
            self.assertIn(f"Фраза {i}.", none)

    def test_slide_progress_and_saved_full_frames(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "lecture.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
            self.assertTrue(writer.isOpened())
            for color in [255, 160]:
                frame = np.full((360, 640, 3), color, dtype=np.uint8)
                for row in range(6):
                    cv2.putText(frame, f"Lecture slide text {row}", (20, 40 + row * 48),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 1)
                for _ in range(60):
                    writer.write(frame)
            writer.release()
            args = app.parser().parse_args([str(source), "--slides"])
            progress_output = io.StringIO()
            with redirect_stderr(progress_output):
                slides = app.extract_slides(source, Path(tmp), [], args)
            self.assertEqual(len(slides), 2)
            self.assertIn("100%", progress_output.getvalue())
            self.assertEqual(cv2.imread(str(Path(tmp) / slides[0]["path"])).shape[:2], (360, 640))

    def test_slide_interval_and_speech_filters(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "lecture.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
            writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
            writer.release()
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            candidates = [(10, frame), (20, frame), (40, frame)]
            speech = [{"start": 0, "end": 12, "text": "Голос."}]
            cases = [
                (["--min-slide-interval", "25"], [], [10, 40], "min_slide_interval"),
                (["--require-speech", "--speech-window", "5"], speech, [10], "require_speech"),
            ]
            for options, segments, expected, filter_name in cases:
                with self.subTest(filter=filter_name):
                    destination = Path(tmp) / filter_name
                    destination.mkdir()
                    args = app.parser().parse_args([str(source), "--slides", *options])
                    with patch.object(app, "slide_candidates", return_value=candidates), \
                            patch.object(app, "heuristic_slide", return_value={"keep": True, "reason": "test"}), \
                            redirect_stderr(io.StringIO()):
                        slides = app.extract_slides(source, destination, segments, args)
                    self.assertEqual([slide["time"] for slide in slides], expected)
                    decisions = json.loads((destination / "slide-decisions.json").read_text())
                    self.assertEqual(decisions[1]["filter"], filter_name)

    def test_real_ffmpeg_extraction_output_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "Лекция с пробелом.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                            str(source)], check=True)
            def recognize(wav, args):
                self.assertTrue(wav.is_file())
                return {"language": "ru", "segments": [{"start": 0, "end": 1, "text": "Пример.", "words": []}]}
            with patch("sys.argv", ["transcribe.py", str(source), "--no-speakers", "--output-language", ""]), patch.object(app, "recognize", recognize), patch.object(app, "preflight"):
                app.main()
            folder = source.with_suffix("")
            self.assertIn("Пример.", (folder / "transcript.md").read_text())
            (folder / "slides").mkdir()
            (folder / "slides" / "a.jpg").write_bytes(b"cached-jpg")
            app.write_json(folder / "slides.json", [{"time": 0, "path": "slides/a.jpg"}])
            app.write_json(folder / "speakers.json", [{"start": 0, "end": 1, "speaker": "a"}])
            with patch("sys.argv", ["transcribe.py", str(source), "--reuse-transcript", "--no-speakers", "--output-language", ""]), patch.object(app, "recognize", side_effect=AssertionError):
                app.main()
            self.assertEqual(len(list((folder / "runs").glob("*/transcript.md"))), 1)
            repeated = next((folder / "runs").glob("*/transcript.md")).parent
            self.assertEqual((repeated / "slides" / "a.jpg").read_bytes(), b"cached-jpg")
            self.assertIn("slides/a.jpg", (repeated / "transcript.md").read_text())
            self.assertEqual(json.loads((repeated / "speakers.json").read_text())[0]["speaker"], "a")
            with patch("sys.argv", ["transcribe.py", str(source), "--no-speakers", "--output-language", ""]), patch.object(app, "preflight"):
                with self.assertRaises(FileExistsError):
                    app.main()
            source.touch()
            with patch("sys.argv", ["transcribe.py", str(source), "--reuse-transcript", "--no-speakers", "--output-language", ""]):
                with self.assertRaisesRegex(RuntimeError, "исходнику"):
                    app.main()


if __name__ == "__main__":
    unittest.main()
