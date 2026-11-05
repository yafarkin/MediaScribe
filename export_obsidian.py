#!/usr/bin/env python3
"""Create a self-contained Obsidian-compatible lecture folder."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.parse import quote

from export_common import load_lecture, markdown_escape


def export_obsidian(folder, output=None):
    from tqdm import tqdm
    lecture = load_lecture(folder)
    target = (Path(output).expanduser() if output else lecture.folder / "obsidian").resolve()
    if target.exists():
        raise FileExistsError(f"Папка уже существует: {target}. Выберите другую через --output.")
    en = lecture.language == "en"
    link_label = "Presentation and transcript" if en else "Презентация и транскрипция"
    title = markdown_escape(lecture.title)
    presentation = [f"# {title}", "", "[[summary|" + ("Summary" if en else "Конспект") + "]]", ""]
    target.parent.mkdir(parents=True, exist_ok=True)
    # Publish the folder only after every asset and both notes have been written.
    with tempfile.TemporaryDirectory(prefix=".obsidian-", dir=target.parent) as temporary:
        staging = Path(temporary) / "lecture"
        staging.mkdir()
        for index, image in enumerate(tqdm(lecture.images, desc="Obsidian", unit="слайд")):
            relative = image.relative_to(lecture.folder)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, destination)
            label = ("Slide" if en else "Слайд") + f" {index + 1}"
            presentation += [f"## {label}", "", lecture.period(index), "",
                             f"![{label}]({quote(relative.as_posix())})", ""]
            for speaker, text in lecture.replicas(index):
                prefix = f"**{markdown_escape(speaker)}:** " if speaker else ""
                presentation += [prefix + markdown_escape(text), ""]
        (staging / "presentation.md").write_text("\n".join(presentation), encoding="utf-8")
        metadata = ("---\n" + "title: " + json.dumps(lecture.title, ensure_ascii=False) + "\n"
                    + "source: " + json.dumps(lecture.source_name, ensure_ascii=False) + "\n"
                    + "type: lecture\n---\n\n")
        description = (f"Lecture material. Slides: {len(lecture.slides)}. Speech segments: {len(lecture.speech)}."
                       if en else f"Материал лекции. Слайдов: {len(lecture.slides)}. Реплик: {len(lecture.speech)}.")
        summary = metadata + f"# {title}\n\n{description}\n\n[[presentation|{link_label}]]\n\n"
        existing = lecture.folder / "summary.md"
        if existing.is_file():
            content = existing.read_text(encoding="utf-8-sig").strip()
            if content.startswith("# "):
                content = content.partition("\n")[2].lstrip()
            summary += content + "\n"
        (staging / "summary.md").write_text(summary, encoding="utf-8")
        staging.rename(target)
    print(f"Obsidian готов: {target}")
    return target


def main():
    p = argparse.ArgumentParser(description="Папка Obsidian с конспектом, транскрипцией и слайдами")
    p.add_argument("folder", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    export_obsidian(args.folder, args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
