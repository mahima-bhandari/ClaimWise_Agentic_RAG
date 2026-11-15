
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .ocr import extract_page_text_with_ocr


# Pages with fewer characters than this are flagged for OCR.
MIN_TEXT_CHARS_FOR_OCR = 40


@dataclass
class PageExtraction:
    """Stores extracted information for one PDF page."""

    page_number: int
    text: str
    character_count: int
    needs_ocr: bool
    extraction_method: str


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
        """Return page numbers that still need OCR."""

        return [
            page.page_number
            for page in self.pages
            if page.needs_ocr
        ]


def extract_pdf(
    pdf_path: str | Path,
    enable_ocr: bool = True,
    ocr_language: str = "eng",
    ocr_dpi: int = 200,
) -> PDFExtractionResult:
    """
    Validate a PDF, extract native text and apply OCR
    to pages with insufficient native text.
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

            # 4. Check password protection.
            if document.needs_pass:
                raise ValueError(
                    "This PDF is password protected."
                )

            # 5. Read metadata.
            metadata = document.metadata or {}

            # 6. Process each page.
            for page_index, page in enumerate(document):

                # First attempt: native text extraction.
                native_text = page.get_text(
                    "text",
                    sort=True,
                ).strip()

                final_text = native_text
                extraction_method = "native"

                # 7. Decide whether OCR is required.
                if (
                    enable_ocr
                    and len(native_text) < MIN_TEXT_CHARS_FOR_OCR
                ):
                    ocr_text = extract_page_text_with_ocr(
                        page=page,
                        language=ocr_language,
                        dpi=ocr_dpi,
                    )

                    # Keep whichever extraction produced
                    # more usable text.
                    if len(ocr_text) > len(native_text):
                        final_text = ocr_text
                        extraction_method = "ocr"

                character_count = len(final_text)

                # 8. Check whether the page still lacks text.
                needs_ocr = (
                    character_count < MIN_TEXT_CHARS_FOR_OCR
                )

                extracted_pages.append(
                    PageExtraction(
                        page_number=page_index + 1,
                        text=final_text,
                        character_count=character_count,
                        needs_ocr=needs_ocr,
                        extraction_method=extraction_method,
                    )
                )

            # 9. Return structured results.
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

    except RuntimeError:
        raise

    except Exception as error:
        raise RuntimeError(
            f"Unexpected error while processing PDF: {error}"
        ) from error
