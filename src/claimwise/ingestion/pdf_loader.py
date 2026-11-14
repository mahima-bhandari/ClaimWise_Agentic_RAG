
from dataclasses import dataclass
from pathlib import Path

import pymupdf


# Pages with fewer characters than this are flagged for OCR review.
MIN_TEXT_CHARS_FOR_OCR = 40


@dataclass
class PageExtraction:
    """Stores extracted information for one PDF page."""

    page_number: int
    text: str
    character_count: int
    needs_ocr: bool


@dataclass
class PDFExtractionResult:
    """Stores the complete PDF extraction result."""

    file_path: str
    file_name: str
    total_pages: int
    pages: list[PageExtraction]
    title: str | None
    author: str | None

    @property
    def pages_requiring_ocr(self) -> list[int]:
        """Return page numbers flagged for OCR."""
        return [
            page.page_number
            for page in self.pages
            if page.needs_ocr
        ]


def extract_pdf(pdf_path: str | Path) -> PDFExtractionResult:
    """
    Validate a PDF and extract its text page by page.

    Pages with very little extracted text are flagged
    for possible OCR processing.
    """

    pdf_path = Path(pdf_path)

    # 1. Validate file existence.
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    # 2. Validate file extension.
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    extracted_pages = []

    # 3. Open the PDF.
    try:
        with pymupdf.open(str(pdf_path)) as document:

            # 4. Check whether the PDF is password protected.
            if document.needs_pass:
                raise ValueError(
                    "This PDF is password protected. "
                    "Please provide an accessible PDF."
                )

            # 5. Read document metadata.
            metadata = document.metadata or {}

            # 6. Extract text page by page.
            for page_index, page in enumerate(document):

                text = page.get_text("text", sort=True)

                character_count = len(text.strip())

                # 7. Flag pages with very little extracted text.
                needs_ocr = (
                    character_count < MIN_TEXT_CHARS_FOR_OCR
                )

                page_result = PageExtraction(
                    page_number=page_index + 1,
                    text=text,
                    character_count=character_count,
                    needs_ocr=needs_ocr,
                )

                extracted_pages.append(page_result)

            # 8. Return structured extraction results.
            return PDFExtractionResult(
                file_path=str(pdf_path.resolve()),
                file_name=pdf_path.name,
                total_pages=len(document),
                pages=extracted_pages,
                title=metadata.get("title"),
                author=metadata.get("author"),
            )

    except (FileNotFoundError, ValueError):
        raise

    except pymupdf.FileDataError as error:
        raise ValueError(
            "The PDF is invalid, corrupted or unreadable."
        ) from error

    except Exception as error:
        raise RuntimeError(
            f"Unexpected error while processing PDF: {error}"
        ) from error
