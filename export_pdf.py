#!/usr/bin/env python3
"""Create a readable PDF with each captured slide followed by its speech."""
import argparse
from html import escape
import os
from pathlib import Path
import sys

from export_common import load_lecture


def find_font(custom=None):
    if custom:
        path = Path(custom).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Шрифт не найден: {path}")
        return path
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Не найден шрифт с кириллицей. Укажите TTF через --font-file (в transcribe.py: --pdf-font-file).")


def export_pdf(folder, output=None, force=False, font_file=None):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer
    from PIL import Image as PILImage
    from tqdm import tqdm

    lecture = load_lecture(folder)
    output = (Path(output).expanduser() if output else lecture.folder / (lecture.title + ".pdf")).resolve()
    if output.suffix.lower() != ".pdf":
        raise ValueError("Выходной файл должен иметь расширение .pdf")
    if output.exists() and not force:
        raise FileExistsError(f"Уже существует {output}. Выберите --output или --force.")
    pdfmetrics.registerFont(TTFont("LectureText", str(find_font(font_file))))
    body = ParagraphStyle("speech", fontName="LectureText", fontSize=10.5, leading=15,
                          spaceAfter=9, alignment=TA_LEFT, splitLongWords=True)
    heading = ParagraphStyle("heading", parent=body, fontSize=14, leading=19, spaceAfter=12, keepWithNext=True)
    timing = ParagraphStyle("timing", parent=body, textColor=colors.HexColor("#555555"), keepWithNext=True)
    width, height = A4
    margin = 42
    available_width = width - 2 * margin
    story = []
    for index, image in enumerate(tqdm(lecture.images, desc="PDF", unit="слайд")):
        if index:
            story.append(PageBreak())
        label = "Slide" if lecture.language == "en" else "Слайд"
        story.append(Paragraph(escape(f"{label} {index + 1}"), heading))
        story.append(Paragraph(escape(lecture.period(index)), timing))
        with PILImage.open(image) as bitmap:
            image_width, image_height = bitmap.size
        scale = min(available_width / image_width, 300 / image_height)
        story += [Image(str(image), width=image_width * scale, height=image_height * scale), Spacer(1, 16)]
        for segment in lecture.groups[index]:
            speaker = escape(segment.get("speaker", "Спикер 1"))
            text = escape(segment["text"].strip()).replace("\n", "<br/>")
            story.append(Paragraph(f"{speaker}: {text}", body))
    def page_chrome(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, width, height, fill=1, stroke=0)
        canvas.setFont("LectureText", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        # Fit long source names without letting the running header cross the page.
        title = lecture.title
        while title and pdfmetrics.stringWidth(title, "LectureText", 8) > available_width:
            title = title[:-1]
        canvas.drawString(margin, height - 25, title)
        canvas.drawRightString(width - margin, 23, str(doc.page))
        canvas.restoreState()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.pdf")
    try:
        doc = SimpleDocTemplate(str(temporary), pagesize=A4, leftMargin=margin, rightMargin=margin,
                                topMargin=margin, bottomMargin=margin, title=lecture.title)
        doc.build(story, onFirstPage=page_chrome, onLaterPages=page_chrome)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"PDF готов: {output}")
    return output


def main():
    p = argparse.ArgumentParser(description="PDF: слайд, временной диапазон и реплики без отдельных таймкодов")
    p.add_argument("folder", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--font-file", type=Path)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    export_pdf(args.folder, args.output, args.force, args.font_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
