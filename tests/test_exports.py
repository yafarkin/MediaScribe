import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from urllib.parse import unquote
from unittest.mock import patch

from PIL import Image
from pypdf import PdfReader

from export_obsidian import export_obsidian
from export_pdf import export_pdf
from export_pptx import export_deck
from pptx import Presentation
import transcribe


class ExportTests(unittest.TestCase):
    def fixture(self, folder, text="Реплика без отдельного таймкода."):
        (folder / "slides").mkdir()
        Image.new("RGB", (1600, 900), "white").save(folder / "slides" / "кадр 01.jpg")
        (folder / "slides.json").write_text(json.dumps([{"time": 2, "path": "slides/кадр 01.jpg"}]))
        (folder / "transcript.json").write_text(json.dumps([
            {"start": 0, "end": 20, "speaker": "Спикер 1", "text": text}]))
        (folder / "raw-transcript.json").write_text(json.dumps({"language": "ru", "source": {"path": "/media/Лекция.mp4"}, "duration": 30}))

    def test_obsidian_is_self_contained_after_moving_and_contains_real_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            (folder / "summary.md").write_text("# Лекция\n\n## Тезисы\n\nОсновная мысль лекции.")
            target = export_obsidian(folder)
            moved = folder / "vault" / "lecture"
            moved.parent.mkdir()
            shutil.move(target, moved)
            summary = (moved / "summary.md").read_text()
            presentation = (moved / "presentation.md").read_text()
            self.assertIn("[[presentation|", summary)
            self.assertIn("Основная мысль лекции.", summary)
            self.assertIn("[[summary|", presentation)
            self.assertIn("00:00:00 - 00:00:30", presentation)
            self.assertNotIn("[00:", presentation)
            for link in re.findall(r"!\[[^]]*\]\(([^)]+)\)", presentation):
                self.assertTrue((moved / unquote(link)).is_file())
            self.assertNotIn(str(folder), presentation)

    def test_obsidian_without_summary_has_description_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            target = export_obsidian(folder)
            self.assertIn("Слайдов: 1", (target / "summary.md").read_text())
            with self.assertRaises(FileExistsError):
                export_obsidian(folder)

    def test_all_exports_label_only_speaker_changes_and_reset_on_new_slide(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder)
            (folder / "slides.json").write_text(json.dumps([
                {"time": 0, "path": "slides/кадр 01.jpg"}, {"time": 10, "path": "slides/кадр 01.jpg"}]))
            speech = [{"start": i * 2, "end": i * 2 + 2, "speaker": speaker, "text": f"Фраза {i}."}
                      for i, speaker in enumerate(["Спикер 1", "Спикер 1", "Спикер 2", "Спикер 2", "Спикер 1", "Спикер 1"])]
            (folder / "transcript.json").write_text(json.dumps(speech))
            deck = Presentation(export_deck(folder))
            first = deck.slides[0].notes_slide.notes_text_frame.text
            second = deck.slides[1].notes_slide.notes_text_frame.text
            self.assertEqual(first.count("Спикер 1:"), 2)
            self.assertEqual(second.count("Спикер 1:"), 1)
            obsidian = export_obsidian(folder)
            markdown = (obsidian / "presentation.md").read_text()
            pdf = PdfReader(export_pdf(folder))
            pdf_text = "\n".join(p.extract_text() for p in pdf.pages)
            for text in [first + second, markdown.replace("**", ""), pdf_text]:
                self.assertEqual(text.count("Спикер 1:"), 3)
                self.assertEqual(text.count("Спикер 2:"), 1)
                for i in range(6):
                    self.assertIn(f"Фраза {i}.", text)

    def test_pdf_embeds_cyrillic_and_paginates_long_speech_without_losing_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture(folder, "Очень длинная реплика с символами < и &. " * 400 + "Конец материала.")
            output = export_pdf(folder)
            self.assertEqual(output.name, "Лекция.pdf")
            reader = PdfReader(output)
            self.assertGreater(len(reader.pages), 1)
            text = "\n".join(page.extract_text() for page in reader.pages)
            self.assertIn("00:00:00 - 00:00:30", text)
            self.assertIn("Очень длинная реплика", text)
            self.assertIn("Конец материала.", " ".join(text.split()))
            self.assertNotIn("[00:", text)
            self.assertTrue(any(list(page.images) for page in reader.pages))
            with self.assertRaises(FileExistsError):
                export_pdf(folder)

    def test_no_arguments_prints_help_successfully(self):
        output = io.StringIO()
        with patch("sys.argv", ["transcribe.py"]), redirect_stdout(output):
            self.assertIsNone(transcribe.main())
        for flag in ["--pptx", "--pdf", "--obsidian", "--summarize", "--timestamps"]:
            self.assertIn(flag, output.getvalue())
