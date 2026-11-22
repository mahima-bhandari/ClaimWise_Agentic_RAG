
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from claimwise.ingestion.ingestion_service import (
    IngestionService,
    IngestionResult,
)
from claimwise.ingestion.metadata import DocumentMetadata
from claimwise.ingestion.pdf_loader import PDFExtractionResult
from claimwise.chunking.text_chunker import TextChunk


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a temporary PDF placeholder for mocked tests."""
    pdf_path = tmp_path / "policy.pdf"
    pdf_path.write_bytes(b"fake pdf content")
    return pdf_path


@pytest.fixture
def sample_extraction():
    return PDFExtractionResult(
        file_path="policy.pdf",
        file_name="policy.pdf",
        total_pages=1,
        pages=[],
        title="Test Policy",
        author="Test Author",
    )


@pytest.fixture
def sample_metadata():
    return DocumentMetadata(
        document_id="doc-123",
        content_hash="abc123",
        file_name="policy.pdf",
        file_size_bytes=100,
        file_extension=".pdf",
        uploaded_at="2026-11-22T10:00:00+00:00",
        source_type="user_upload",
        title="Test Policy",
        author="Test Author",
        total_pages=1,
    )


@pytest.fixture
def sample_chunks():
    return [
        TextChunk(
            chunk_id="chunk-1",
            document_id="doc-123",
            chunk_index=0,
            text="Policy coverage details.",
            page_number=1,
            estimated_tokens=5,
        ),
        TextChunk(
            chunk_id="chunk-2",
            document_id="doc-123",
            chunk_index=1,
            text="Policy exclusion details.",
            page_number=1,
            estimated_tokens=5,
        ),
    ]


@pytest.fixture
def mock_indexing_service():
    service = Mock()
    service.index_chunks.return_value = 2
    return service


@pytest.fixture
def ingestion_service(tmp_path, mock_indexing_service):
    return IngestionService(
        indexing_service=mock_indexing_service,
        upload_dir=tmp_path / "uploads",
    )


@patch("claimwise.ingestion.ingestion_service.update_ingestion_status")
@patch("claimwise.ingestion.ingestion_service.save_chunks")
@patch("claimwise.ingestion.ingestion_service.chunk_document")
@patch("claimwise.ingestion.ingestion_service.save_document")
@patch("claimwise.ingestion.ingestion_service.create_document_metadata")
@patch("claimwise.ingestion.ingestion_service.extract_pdf")
def test_successful_ingestion(
    mock_extract_pdf,
    mock_create_metadata,
    mock_save_document,
    mock_chunk_document,
    mock_save_chunks,
    mock_update_status,
    ingestion_service,
    sample_pdf,
    sample_extraction,
    sample_metadata,
    sample_chunks,
    mock_indexing_service,
):
    mock_extract_pdf.return_value = sample_extraction
    mock_create_metadata.return_value = sample_metadata
    mock_chunk_document.return_value = sample_chunks
    mock_save_chunks.return_value = 2

    result = ingestion_service.ingest_pdf(sample_pdf)

    assert isinstance(result, IngestionResult)
    assert result.metadata == sample_metadata
    assert result.chunk_count == 2
    assert Path(result.storage_path).exists()

    mock_extract_pdf.assert_called_once_with(sample_pdf)
    mock_save_document.assert_called_once()
    mock_save_chunks.assert_called_once_with(
        document_id="doc-123",
        chunks=sample_chunks,
        database_path=None,
    )
    mock_indexing_service.index_chunks.assert_called_once_with(
        sample_chunks
    )

    assert mock_update_status.call_count == 2


@patch("claimwise.ingestion.ingestion_service.extract_pdf")
def test_pdf_extraction_failure(
    mock_extract_pdf,
    ingestion_service,
    sample_pdf,
):
    mock_extract_pdf.side_effect = ValueError("Invalid PDF")

    with pytest.raises(ValueError, match="Invalid PDF"):
        ingestion_service.ingest_pdf(sample_pdf)

    mock_extract_pdf.assert_called_once()
    assert list(ingestion_service.upload_dir.iterdir()) == []


@patch("claimwise.ingestion.ingestion_service.update_ingestion_status")
@patch("claimwise.ingestion.ingestion_service.save_document")
@patch("claimwise.ingestion.ingestion_service.create_document_metadata")
@patch("claimwise.ingestion.ingestion_service.extract_pdf")
def test_document_save_failure_removes_copied_pdf(
    mock_extract_pdf,
    mock_create_metadata,
    mock_save_document,
    mock_update_status,
    ingestion_service,
    sample_pdf,
    sample_extraction,
    sample_metadata,
    mock_indexing_service,
):
    mock_extract_pdf.return_value = sample_extraction
    mock_create_metadata.return_value = sample_metadata
    mock_save_document.side_effect = RuntimeError("Database error")

    with pytest.raises(RuntimeError, match="Database error"):
        ingestion_service.ingest_pdf(sample_pdf)

    assert list(ingestion_service.upload_dir.iterdir()) == []

    mock_indexing_service.index_chunks.assert_not_called()
    mock_update_status.assert_not_called()


@patch("claimwise.ingestion.ingestion_service.update_ingestion_status")
@patch("claimwise.ingestion.ingestion_service.delete_chunks_for_document")
@patch("claimwise.ingestion.ingestion_service.save_chunks")
@patch("claimwise.ingestion.ingestion_service.chunk_document")
@patch("claimwise.ingestion.ingestion_service.save_document")
@patch("claimwise.ingestion.ingestion_service.create_document_metadata")
@patch("claimwise.ingestion.ingestion_service.extract_pdf")
def test_indexing_failure_marks_document_failed(
    mock_extract_pdf,
    mock_create_metadata,
    mock_save_document,
    mock_chunk_document,
    mock_save_chunks,
    mock_delete_chunks,
    mock_update_status,
    ingestion_service,
    sample_pdf,
    sample_extraction,
    sample_metadata,
    sample_chunks,
    mock_indexing_service,
):
    mock_extract_pdf.return_value = sample_extraction
    mock_create_metadata.return_value = sample_metadata
    mock_chunk_document.return_value = sample_chunks
    mock_save_chunks.return_value = 2

    mock_indexing_service.index_chunks.side_effect = RuntimeError(
        "Qdrant unavailable"
    )

    with pytest.raises(RuntimeError, match="Qdrant unavailable"):
        ingestion_service.ingest_pdf(sample_pdf)

    mock_indexing_service.delete_document_vectors.assert_called_once_with(
        "doc-123"
    )
    mock_delete_chunks.assert_called_once_with(
        "doc-123",
        database_path=None,
    )

    failed_call = mock_update_status.call_args_list[-1]

    assert failed_call.kwargs["status"] == "failed"
    assert "Qdrant unavailable" in failed_call.kwargs["error_message"]


@patch("claimwise.ingestion.ingestion_service.update_ingestion_status")
@patch("claimwise.ingestion.ingestion_service.delete_chunks_for_document")
@patch("claimwise.ingestion.ingestion_service.save_chunks")
@patch("claimwise.ingestion.ingestion_service.chunk_document")
@patch("claimwise.ingestion.ingestion_service.save_document")
@patch("claimwise.ingestion.ingestion_service.create_document_metadata")
@patch("claimwise.ingestion.ingestion_service.extract_pdf")
def test_chunk_index_count_mismatch(
    mock_extract_pdf,
    mock_create_metadata,
    mock_save_document,
    mock_chunk_document,
    mock_save_chunks,
    mock_delete_chunks,
    mock_update_status,
    ingestion_service,
    sample_pdf,
    sample_extraction,
    sample_metadata,
    sample_chunks,
    mock_indexing_service,
):
    mock_extract_pdf.return_value = sample_extraction
    mock_create_metadata.return_value = sample_metadata
    mock_chunk_document.return_value = sample_chunks
    mock_save_chunks.return_value = 2
    mock_indexing_service.index_chunks.return_value = 1

    with pytest.raises(RuntimeError, match="does not match"):
        ingestion_service.ingest_pdf(sample_pdf)

    failed_call = mock_update_status.call_args_list[-1]

    assert failed_call.kwargs["status"] == "failed"
    mock_delete_chunks.assert_called_once_with(
        "doc-123",
        database_path=None,
    )


def test_close_delegates_to_indexing_service(ingestion_service):
    ingestion_service.close()

    ingestion_service.indexing_service.close.assert_called_once()
