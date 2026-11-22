import re
from dataclasses import dataclass
from math import ceil
from uuid import uuid4

from claimwise.ingestion.pdf_loader import PDFExtractionResult


@dataclass
class TextChunk:
    """Represents one searchable piece of a policy document."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    page_number: int
    estimated_tokens: int


def estimate_tokens(text: str) -> int:
    """
    Estimate token count without loading a tokenizer.

    This is a rough estimate, not an exact model token count.
    """

    word_count = len(text.split())

    return ceil(word_count * 1.3)


def split_into_paragraphs(text: str) -> list[str]:
    """Split extracted text into non-empty paragraphs."""

    # Join single line breaks but preserve blank-line boundaries.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]


def chunk_document(
    extraction_result: PDFExtractionResult,
    document_id: str,
    target_tokens: int = 400,
    overlap_tokens: int = 60,
) -> list[TextChunk]:
    """
    Convert extracted PDF pages into searchable text chunks.

    Chunks are created page by page to preserve citation accuracy.
    Paragraph boundaries are preferred where possible.
    """

    if not document_id.strip():
        raise ValueError("Document ID cannot be empty.")

    if target_tokens <= 0:
        raise ValueError("Target tokens must be positive.")

    if overlap_tokens < 0:
        raise ValueError("Overlap tokens cannot be negative.")

    if overlap_tokens >= target_tokens:
        raise ValueError(
            "Overlap must be smaller than the target chunk size."
        )

    # Convert approximate token settings into word limits.
    words_per_chunk = max(1, int(target_tokens / 1.3))
    overlap_words = int(overlap_tokens / 1.3)

    chunks = []
    chunk_index = 0

    for page in extraction_result.pages:

        if not page.text.strip():
            continue

        paragraphs = split_into_paragraphs(page.text)

        current_words = []

        def save_chunk(words: list[str]) -> None:
            nonlocal chunk_index

            if not words:
                return

            chunk_text = " ".join(words).strip()

            if not chunk_text:
                return

            chunks.append(
                TextChunk(
                    chunk_id=str(uuid4()),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    page_number=page.page_number,
                    estimated_tokens=estimate_tokens(chunk_text),
                )
            )

            chunk_index += 1

        for paragraph in paragraphs:

            paragraph_words = paragraph.split()

            # Split exceptionally long paragraphs into smaller pieces.
            while len(paragraph_words) > words_per_chunk:

                available_space = (
                    words_per_chunk - len(current_words)
                )

                if available_space <= 0:
                    save_chunk(current_words)

                    current_words = (
                        current_words[-overlap_words:]
                        if overlap_words > 0
                        else []
                    )

                    available_space = (
                        words_per_chunk - len(current_words)
                    )

                current_words.extend(
                    paragraph_words[:available_space]
                )

                paragraph_words = paragraph_words[available_space:]

                save_chunk(current_words)

                current_words = (
                    current_words[-overlap_words:]
                    if overlap_words > 0
                    else []
                )

            # Add the remaining paragraph if it fits.
            if (
                current_words
                and len(current_words) + len(paragraph_words)
                > words_per_chunk
            ):
                save_chunk(current_words)

                current_words = (
                    current_words[-overlap_words:]
                    if overlap_words > 0
                    else []
                )

            current_words.extend(paragraph_words)

        # Save the remaining words on this page.
        save_chunk(current_words)

    return chunks
