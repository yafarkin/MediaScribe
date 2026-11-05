#!/usr/bin/env python3
"""Package captured slide images and speech into a local PowerPoint."""
import argparse
from pathlib import Path
import sys

from transcribe import stamp
from export_common import assign_notes, load_lecture


def export_deck(folder, output=None, force=False):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches
    from PIL import Image
    from tqdm import tqdm

    lecture = load_lecture(folder)
    folder = lecture.folder
    output = (Path(output).expanduser() if output else folder / (lecture.title + ".pptx")).resolve()
    if output.suffix.lower() != ".pptx":
        raise ValueError("Выходной файл должен иметь расширение .pptx")
    if output.exists() and not force:
        raise FileExistsError(f"Уже существует {output}. Выберите --output или --force.")
    images = []
    for path in lecture.images:
        with Image.open(path) as image:
            images.append((path, image.size))
    deck = Presentation()
    first_width, first_height = images[0][1]
    # Preserve the video's canvas ratio; pictures on later slides fit, never crop.
    ratio = first_width / first_height
    height = min(7.5, 50 / ratio)
    width = height * ratio
    if min(width, height) < 1:
        raise ValueError("Неподдерживаемое соотношение сторон кадра.")
    deck.slide_width, deck.slide_height = Inches(width), Inches(height)
    deck.core_properties.title = lecture.title
    deck.core_properties.subject = "Captured lecture slides with transcript in speaker notes"
    for index, ((path, (image_width, image_height)), item) in enumerate(
            tqdm(list(zip(images, lecture.slides)), desc="PowerPoint", unit="слайд")):
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        slide.name = f"{index + 1}: {stamp(item['time'])}"
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor(0, 0, 0)
        scale = min(deck.slide_width / image_width, deck.slide_height / image_height)
        picture_width, picture_height = round(image_width * scale), round(image_height * scale)
        slide.shapes.add_picture(str(path), (deck.slide_width - picture_width) // 2,
                                 (deck.slide_height - picture_height) // 2,
                                 width=picture_width, height=picture_height)
        slide.notes_slide.notes_text_frame.text = lecture.notes(index).rstrip()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.pptx")
    deck.save(str(temporary))
    temporary.replace(output)
    print(f"PowerPoint готов: {output}")
    return output


def main():
    p = argparse.ArgumentParser(description="Слайды JPG и транскрипция в заметках PowerPoint")
    p.add_argument("folder", type=Path, help="Папка с slides.json, transcript.json и кадрами")
    p.add_argument("--output", type=Path)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    export_deck(args.folder, args.output, args.force)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
