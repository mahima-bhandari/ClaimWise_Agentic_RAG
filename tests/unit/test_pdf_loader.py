
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from claimwise.ingestion.pdf_loader import extract_pdf


def create_digital_pdf(pdf_path: Path):
    """Create a PDF containing selectable digital text."""

    document = pymupdf.open()
    page = document.new_page()

    text = (
        "Health Insurance Policy\n"
        "Pre-existing diseases have a waiting period "
        "of thirty-six months.\n"
        "Room rent is subject to policy limits."
    )

    page.insert_textbox(
        pymupdf.Rect(50, 50, 550, 250),
        text,
        fontsize=14,
    )

    document.save(str(pdf_path))
    document.close()


def create_scanned_pdf(pdf_path: Path):
    """Create a PDF containing text as an image."""

    image = Image.new("RGB", (1800, 900), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=65)

    lines = [
        "CLAIMWISE OCR TEST",
        "Health Insurance Policy",
        "Pre-existing diseases have a waiting",
        "period of thirty-six months.",
        "Room rent is subject to policy limits.",
    ]

    y_position = 100

    for line in lines:
        draw.text(
            (100, y_position),
            line,
            fill="black",
            font=font,
        )
        y_position += 130

    image_bytes_path = pdf_path.parent / "temporary_scan.png"
    image.save(image_bytes_path)

    document = pymupdf.open()
    page = document.new_page(width=900, height=450)

    page.insert_image(
        page.rect,
        filename=str(image_bytes_path),
    )

    document.save(str(pdf_path))
    document.close()

    image_bytes_path.unlink()


def test_native_text_extraction(tmp_path):
    """Digitally generated PDFs should use native extraction."""

    pdf_path = tmp_path / "digital_policy.pdf"
    create_digital_pdf(pdf_path)

    result = extract_pdf(pdf_path)

    assert result.total_pages == 1
    assert result.pages[0].extraction_method == "native"
    assert "Health Insurance Policy" in result.pages[0].text
    assert result.pages[0].needs_ocr is False


def test_scanned_pdf_uses_ocr(tmp_path):
    """Image-based PDFs should use OCR."""

    pdf_path = tmp_path / "scanned_policy.pdf"
    create_scanned_pdf(pdf_path)

    result = extract_pdf(pdf_path)

    assert result.total_pages == 1
    assert result.pages[0].extraction_method == "ocr"
    assert "Health Insurance Policy" in result.pages[0].text
    assert result.pages[0].needs_ocr is False


def test_ocr_can_be_disabled(tmp_path):
    """OCR should not run when explicitly disabled."""

    pdf_path = tmp_path / "scanned_policy.pdf"
    create_scanned_pdf(pdf_path)

    result = extract_pdf(
        pdf_path,
        enable_ocr=False,
    )

    assert result.pages[0].extraction_method == "native"
    assert result.pages[0].needs_ocr is True


def test_missing_pdf_raises_error(tmp_path):
    """A missing file should raise FileNotFoundError."""

    pdf_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_pdf(pdf_path)


def test_wrong_file_extension_raises_error(tmp_path):
    """Non-PDF files should be rejected."""

    file_path = tmp_path / "document.txt"
    file_path.write_text("This is not a PDF.")

    with pytest.raises(ValueError):
        extract_pdf(file_path)
