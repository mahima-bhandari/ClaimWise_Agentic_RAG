
import pymupdf
import pytest

from claimwise.ingestion.metadata import create_document_metadata
from claimwise.ingestion.pdf_loader import extract_pdf
from claimwise.storage.database import initialize_database
from claimwise.storage.repository import (
    DuplicateDocumentError,
    save_document,
    get_document_by_id,
    get_document_by_hash,
    list_documents,
    update_ingestion_status,
    delete_document,
)


def create_test_metadata(tmp_path):
    """Create a test PDF and its metadata."""

    pdf_path = tmp_path / "policy.pdf"

    document = pymupdf.open()
    page = document.new_page()

    page.insert_text(
        (50, 50),
        "ClaimWise Health Insurance Policy Test",
    )

    document.save(str(pdf_path))
    document.close()

    extraction_result = extract_pdf(
        pdf_path,
        enable_ocr=False,
    )

    metadata = create_document_metadata(
        extraction_result
    )

    return metadata, pdf_path


def test_save_and_retrieve_document(tmp_path):
    database_path = tmp_path / "test.db"
    initialize_database(database_path)

    metadata, pdf_path = create_test_metadata(tmp_path)

    document_id = save_document(
        metadata,
        pdf_path,
        database_path,
    )

    result = get_document_by_id(
        document_id,
        database_path,
    )

    assert result is not None
    assert result["document_id"] == document_id
    assert result["file_name"] == "policy.pdf"
    assert result["storage_path"] == str(pdf_path.resolve())
    assert result["ingestion_status"] == "pending"


def test_duplicate_document_is_rejected(tmp_path):
    database_path = tmp_path / "test.db"
    initialize_database(database_path)

    metadata, pdf_path = create_test_metadata(tmp_path)

    save_document(
        metadata,
        pdf_path,
        database_path,
    )

    with pytest.raises(DuplicateDocumentError):
        save_document(
            metadata,
            pdf_path,
            database_path,
        )


def test_get_document_by_hash(tmp_path):
    database_path = tmp_path / "test.db"
    initialize_database(database_path)

    metadata, pdf_path = create_test_metadata(tmp_path)

    save_document(
        metadata,
        pdf_path,
        database_path,
    )

    result = get_document_by_hash(
        metadata.content_hash,
        metadata.source_type,
        database_path,
    )

    assert result is not None
    assert result["document_id"] == metadata.document_id


def test_list_documents(tmp_path):
    database_path = tmp_path / "test.db"
    initialize_database(database_path)

    metadata, pdf_path = create_test_metadata(tmp_path)

    save_document(
        metadata,
        pdf_path,
        database_path,
    )

    results = list_documents(
        database_path=database_path,
    )

    assert len(results) == 1
    assert results[0]["document_id"] == metadata.document_id


def test_update_ingestion_status(tmp_path):
    database_path = tmp_path / "test.db"
    initialize_database(database_path)

    metadata, pdf_path = create_test_metadata(tmp_path)

    save_document(
        metadata,
        pdf_path,
        database_path,
    )

    updated = update_ingestion_status(
        document_id=metadata.document_id,
        status="completed",
        chunk_count=12,
        database_path=database_path,
    )

    result = get_document_by_id(
        metadata.document_id,
        database_path,
    )

    assert updated is True
    assert result["ingestion_status"] == "completed"
    assert result["chunk_count"] == 12


def test_invalid_ingestion_status_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        update_ingestion_status(
            document_id="unknown",
            status="invalid",
            database_path=tmp_path / "test.db",
        )


def test_delete_document(tmp_path):
    database_path = tmp_path / "test.db"
    initialize_database(database_path)

    metadata, pdf_path = create_test_metadata(tmp_path)

    save_document(
        metadata,
        pdf_path,
        database_path,
    )

    deleted = delete_document(
        metadata.document_id,
        database_path,
    )

    result = get_document_by_id(
        metadata.document_id,
        database_path,
    )

    assert deleted is True
    assert result is None
    # The repository must not delete the original PDF.
    assert pdf_path.exists()
