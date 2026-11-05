import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import io
from contextlib import redirect_stderr
from unittest.mock import patch

import transcribe as app


class TranscriptTests(unittest.TestCase):
    def test_speaker_change_splits_whisper_segment_at_word(self):
        segments = [{"start": 0, "end": 4, "text": "Привет. Да.", "words": [
            {"start": 0, "end": 1, "word": " Привет."},
            {"start": 2, "end": 3, "word": " Да."}]}]
        turns = [{"start": 0, "end": 1.5, "speaker": "a"},
                 {"start": 1.5, "end": 4, "speaker": "b"}]
        result = app.utterances(segments, turns)
        self.assertEqual([r["speaker"] for r in result], ["Спикер 1", "Спикер 2"])
        self.assertEqual([r["text"] for r in result], ["Привет.", "Да."])

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

    def test_real_ffmpeg_extraction_output_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "Лекция с пробелом.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                            str(source)], check=True)
            def recognize(wav, args):
                self.assertTrue(wav.is_file())
                return {"language": "ru", "segments": [{"start": 0, "end": 1, "text": "Пример.", "words": []}]}
            with patch("sys.argv", ["transcribe.py", str(source)]), patch.object(app, "recognize", recognize), patch.object(app, "preflight"):
                app.main()
            folder = source.with_suffix("")
            self.assertIn("Пример.", (folder / "transcript.md").read_text())
            (folder / "slides").mkdir()
            (folder / "slides" / "a.jpg").write_bytes(b"cached-jpg")
            app.write_json(folder / "slides.json", [{"time": 0, "path": "slides/a.jpg"}])
            app.write_json(folder / "speakers.json", [{"start": 0, "end": 1, "speaker": "a"}])
            with patch("sys.argv", ["transcribe.py", str(source), "--reuse-transcript"]), patch.object(app, "recognize", side_effect=AssertionError):
                app.main()
            self.assertEqual(len(list((folder / "runs").glob("*/transcript.md"))), 1)
            repeated = next((folder / "runs").glob("*/transcript.md")).parent
            self.assertEqual((repeated / "slides" / "a.jpg").read_bytes(), b"cached-jpg")
            self.assertIn("slides/a.jpg", (repeated / "transcript.md").read_text())
            self.assertEqual(json.loads((repeated / "speakers.json").read_text())[0]["speaker"], "a")
            with patch("sys.argv", ["transcribe.py", str(source)]), patch.object(app, "preflight"):
                with self.assertRaises(FileExistsError):
                    app.main()
            source.touch()
            with patch("sys.argv", ["transcribe.py", str(source), "--reuse-transcript"]):
                with self.assertRaisesRegex(RuntimeError, "исходнику"):
                    app.main()


if __name__ == "__main__":
    unittest.main()
