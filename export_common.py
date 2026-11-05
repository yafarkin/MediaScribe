"""Shared lecture metadata and slide-to-speech alignment for offline exports."""
from bisect import bisect_right
from dataclasses import dataclass
import json
import math
from pathlib import Path

from transcribe import stamp


def assign_notes(slides, speech):
    if not slides:
        raise ValueError("Нет сохранённых слайдов. Сначала запустите transcribe.py с --slides.")
    starts = [float(s["time"]) for s in slides]
    if any(not math.isfinite(t) or t < 0 for t in starts) or any(a >= b for a, b in zip(starts, starts[1:])):
        raise ValueError("Время слайдов должно строго возрастать и быть неотрицательным.")
    groups = [[] for _ in slides]
    for segment in speech:
        midpoint = (float(segment["start"]) + float(segment["end"])) / 2
        groups[max(0, bisect_right(starts, midpoint) - 1)].append(segment)
    return groups


@dataclass
class Lecture:
    folder: Path
    title: str
    source_name: str
    language: str | None
    slides: list
    speech: list
    groups: list
    images: list[Path]
    duration: float | None

    def period(self, index):
        start = 0 if index == 0 else self.slides[index]["time"]
        end = self.slides[index + 1]["time"] if index + 1 < len(self.slides) else self.duration
        end_label = stamp(end) if end is not None else ("end of recording" if self.language == "en" else "конец записи")
        return f"{stamp(start)} - {end_label}"

    def notes(self, index):
        return self.period(index) + "\n\n" + "\n\n".join(
            f"{s.get('speaker', 'Спикер 1')}: {s['text'].strip()}" for s in self.groups[index])


def load_lecture(folder):
    folder = Path(folder).expanduser().resolve()
    slides = json.loads((folder / "slides.json").read_text(encoding="utf-8"))
    speech = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
    raw_path = folder / "raw-transcript.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.is_file() else {}
    source_name = str(raw.get("source", {}).get("path", "")).replace("\\", "/").rsplit("/", 1)[-1]
    title = Path(source_name).stem if source_name else folder.name
    images = []
    groups = assign_notes(slides, speech)
    for item in slides:
        path = (folder / item["path"]).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Кадр должен находиться внутри папки результата.")
        if not path.is_file():
            raise FileNotFoundError(f"Кадр не найден: {path}")
        images.append(path)
    duration = raw.get("duration")
    if duration is not None:
        duration = float(duration)
        if not math.isfinite(duration) or duration < slides[-1]["time"]:
            duration = None
    if duration is None:
        last_speech = max((float(s["end"]) for s in speech), default=0)
        duration = last_speech if last_speech > slides[-1]["time"] else None
    return Lecture(folder, title, source_name, raw.get("language"), slides, speech, groups, images, duration)


def markdown_escape(text):
    for char in ("\\", "*", "_", "[", "]", "<", ">", "`", "#"):
        text = text.replace(char, "\\" + char)
    return text
