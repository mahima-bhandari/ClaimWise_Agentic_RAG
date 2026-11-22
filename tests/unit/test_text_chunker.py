
import pytest

from claimwise.ingestion.pdf_loader import (
    PageExtraction,
    PDFExtractionResult,
)
from claimwise.chunking.text_chunker import (
    chunk_document,
    estimate_tokens,
)


def create_extraction_result():
    """Create a sample two-page extraction result."""

    return PDFExtractionResult(
        file_path="sample_policy.pdf",
        file_name="sample_policy.pdf",
        total_pages=2,
        title="Sample Health Policy",
        author="ClaimWise Test",
        pages=[
            PageExtraction(
                page_number=1,
                text=(
                    "Section 1: Waiting Periods\n\n"
                    "Pre-existing diseases are covered "
                    "after thirty-six months of continuous coverage."
                ),
                character_count=120,
                needs_ocr=False,
                extraction_method="native",
            ),
            PageExtraction(
                page_number=2,
                text=(
                    "Section 2: Room Rent\n\n"
                    "Room rent is subject to the limits "
                    "mentioned in the policy schedule."
                ),
                character_count=100,
                needs_ocr=False,
                extraction_method="native",
            ),
        ],
    )


def test_chunk_document_creates_chunks():
    result = create_extraction_result()

    chunks = chunk_document(
        result,
        document_id="test-document-001",
    )

    assert len(chunks) > 0


def test_chunks_preserve_document_id():
    result = create_extraction_result()

    chunks = chunk_document(
        result,
        document_id="test-document-001",
    )

    assert all(
        chunk.document_id == "test-document-001"
        for chunk in chunks
    )


def test_chunks_preserve_page_numbers():
    result = create_extraction_result()

    chunks = chunk_document(
        result,
        document_id="test-document-001",
    )

    page_numbers = {
        chunk.page_number
        for chunk in chunks
    }

    assert page_numbers == {1, 2}


def test_chunk_indexes_are_sequential():
    result = create_extraction_result()

    chunks = chunk_document(
        result,
        document_id="test-document-001",
    )

    indexes = [
        chunk.chunk_index
        for chunk in chunks
    ]

    assert indexes == list(range(len(chunks)))


def test_chunk_ids_are_unique():
    result = create_extraction_result()

    chunks = chunk_document(
        result,
        document_id="test-document-001",
    )

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_empty_pages_are_skipped():
    result = create_extraction_result()

    result.pages[0].text = "   "
    result.pages[1].text = ""

    chunks = chunk_document(
        result,
        document_id="test-document-001",
    )

    assert chunks == []


def test_invalid_document_id_is_rejected():
    result = create_extraction_result()

    with pytest.raises(ValueError):
        chunk_document(
            result,
            document_id=" ",
        )


def test_invalid_chunk_size_is_rejected():
    result = create_extraction_result()

    with pytest.raises(ValueError):
        chunk_document(
            result,
            document_id="test-document-001",
            target_tokens=0,
        )


def test_overlap_must_be_smaller_than_chunk_size():
    result = create_extraction_result()

    with pytest.raises(ValueError):
        chunk_document(
            result,
            document_id="test-document-001",
            target_tokens=100,
            overlap_tokens=100,
        )


def test_token_estimation():
    estimated = estimate_tokens(
        "Health insurance policy waiting period"
    )

    assert estimated == 7
