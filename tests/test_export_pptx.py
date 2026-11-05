import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image
from pptx import Presentation

import export_pptx as app


class PowerPointTests(unittest.TestCase):
    def test_notes_preserve_opening_and_boundary_speech_exactly_once(self):
        slides = [{"time": 4}, {"time": 10}]
        speech = [{"start": 0, "end": 2, "text": "Начало."},
                  {"start": 8, "end": 12, "text": "На границе."},
                  {"start": 20, "end": 22, "text": "Конец."}]
        groups = app.assign_notes(slides, speech)
        self.assertEqual(groups, [[speech[0]], [speech[1], speech[2]]])
        with self.assertRaises(ValueError):
            app.assign_notes([], speech)

    def test_export_contains_images_notes_and_preserves_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "slides").mkdir()
            Image.new("RGB", (1600, 900), "white").save(folder / "slides" / "a.jpg")
            Image.new("RGB", (900, 1600), "black").save(folder / "slides" / "b.jpg")
            (folder / "slides.json").write_text(json.dumps([
                {"time": 1, "path": "slides/a.jpg"}, {"time": 10, "path": "slides/b.jpg"}]))
            (folder / "transcript.json").write_text(json.dumps([
                {"start": 0, "end": 3, "text": "Первая реплика.", "speaker": "Спикер 1"},
                {"start": 11, "end": 15, "text": "Вторая реплика.", "speaker": "Спикер 2"}]))
            (folder / "raw-transcript.json").write_text(json.dumps({"source": {"path": "C:\\Video\\Лекция.Часть 1.mp4"}, "duration": 20}))
            output = app.export_deck(folder)
            self.assertEqual(output.name, "Лекция.Часть 1.pptx")
            deck = Presentation(output)
            self.assertEqual(len(deck.slides), 2)
            self.assertIn("Первая реплика.", deck.slides[0].notes_slide.notes_text_frame.text)
            self.assertNotIn("Вторая реплика.", deck.slides[0].notes_slide.notes_text_frame.text)
            self.assertIn("Спикер 2: Вторая реплика.", deck.slides[1].notes_slide.notes_text_frame.text)
            first_notes = deck.slides[0].notes_slide.notes_text_frame.text
            self.assertTrue(first_notes.startswith("00:00:00 - 00:00:10\n\n"))
            self.assertNotIn("[00:", first_notes)
            self.assertTrue(deck.slides[1].notes_slide.notes_text_frame.text.startswith("00:00:10 - 00:00:20"))
            picture = deck.slides[1].shapes[0]
            self.assertAlmostEqual(picture.width / picture.height, 900 / 1600, places=5)
            self.assertLessEqual(picture.left + picture.width, deck.slide_width)
            with zipfile.ZipFile(output) as archive:
                self.assertIsNone(archive.testzip())
                self.assertIn("ppt/notesSlides/notesSlide2.xml", archive.namelist())
            with self.assertRaises(FileExistsError):
                app.export_deck(folder)
