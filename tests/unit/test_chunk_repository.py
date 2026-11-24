
import pymupdf
import pytest

from claimwise.ingestion.pdf_loader import extract_pdf
from claimwise.ingestion.metadata import DocumentMetadata
from claimwise.ingestion.metadata import create_document_metadata
from claimwise.chunking.text_chunker import chunk_document
from claimwise.chunking.text_chunker import TextChunk
from claimwise.storage.database import initialize_database
from claimwise.storage.repository import save_document
from claimwise.storage.chunk_repository import (
    save_chunks,
    get_chunks_by_document,
    get_chunk_by_id,
    delete_chunks_for_document,
    get_all_chunks,
)


def create_test_data(tmp_path):
    """Create a test document and its chunks."""

    database_path = tmp_path / "test.db"
    pdf_path = tmp_path / "policy.pdf"

    document = pymupdf.open()
    page = document.new_page()

    page.insert_textbox(
        pymupdf.Rect(50, 50, 550, 300),
        (
            "Section 1: Waiting Periods\n\n"
            "Pre-existing diseases are covered after "
            "thirty-six months of continuous coverage.\n\n"
            "Section 2: Room Rent\n\n"
            "Room rent is subject to the limits "
            "mentioned in the policy schedule."
        ),
        fontsize=12,
    )

    document.save(str(pdf_path))
    document.close()

    initialize_database(database_path)

    extraction_result = extract_pdf(
        pdf_path,
        enable_ocr=False,
    )

    metadata = create_document_metadata(extraction_result)

    save_document(
        metadata,
        pdf_path,
        database_path,
    )

    chunks = chunk_document(
        extraction_result,
        document_id=metadata.document_id,
    )

    return database_path, metadata, chunks

def create_test_document(document_id, database_path):
    metadata = DocumentMetadata(
        document_id=document_id,
        content_hash=f"hash-{document_id}",
        file_name=f"{document_id}.pdf",
        file_size_bytes=1024,
        file_extension=".pdf",
        uploaded_at="2026-11-24T10:00:00+00:00",
        source_type="user_upload",
        title=f"Test Policy {document_id}",
        author="Test Author",
        total_pages=2,
    )

    save_document(
        metadata=metadata,
        storage_path=f"uploads/{document_id}.pdf",
        database_path=database_path,
    )

def test_save_and_retrieve_chunks(tmp_path):
    database_path, metadata, chunks = create_test_data(tmp_path)

    saved_count = save_chunks(
        metadata.document_id,
        chunks,
        database_path,
    )

    results = get_chunks_by_document(
        metadata.document_id,
        database_path,
    )

    assert saved_count == len(chunks)
    assert len(results) == len(chunks)


def test_chunks_preserve_original_text(tmp_path):
    database_path, metadata, chunks = create_test_data(tmp_path)

    save_chunks(metadata.document_id, chunks, database_path)

    results = get_chunks_by_document(
        metadata.document_id,
        database_path,
    )

    stored_text = " ".join(row["text"] for row in results)

    assert "Pre-existing diseases" in stored_text
    assert "Room rent" in stored_text


def test_get_chunk_by_id(tmp_path):
    database_path, metadata, chunks = create_test_data(tmp_path)

    save_chunks(metadata.document_id, chunks, database_path)

    result = get_chunk_by_id(
        chunks[0].chunk_id,
        database_path,
    )

    assert result is not None
    assert result["chunk_id"] == chunks[0].chunk_id
    assert result["document_id"] == metadata.document_id


def test_delete_chunks_for_document(tmp_path):
    database_path, metadata, chunks = create_test_data(tmp_path)

    save_chunks(metadata.document_id, chunks, database_path)

    deleted_count = delete_chunks_for_document(
        metadata.document_id,
        database_path,
    )

    results = get_chunks_by_document(
        metadata.document_id,
        database_path,
    )

    assert deleted_count == len(chunks)
    assert results == []


def test_chunks_cannot_be_saved_for_unknown_document(tmp_path):
    database_path, metadata, chunks = create_test_data(tmp_path)

    with pytest.raises(ValueError, match="Document not found"):
        save_chunks(
            "unknown-document-id",
            chunks,
            database_path,
        )


def test_empty_chunk_list_returns_zero(tmp_path):
    database_path, metadata, chunks = create_test_data(tmp_path)

    result = save_chunks(
        metadata.document_id,
        [],
        database_path,
    )

    assert result == 0

def test_get_all_chunks_from_multiple_documents(tmp_path):

    database_path = tmp_path / "test.db"

    initialize_database(database_path)

    create_test_document("doc-1", database_path)
    create_test_document("doc-2", database_path)

    chunks = [
        TextChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            text="Health insurance hospitalization coverage.",
            page_number=1,
            estimated_tokens=6,
        ),
        TextChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            chunk_index=1,
            text="Pre-hospitalization expenses are covered.",
            page_number=2,
            estimated_tokens=6,
        ),
        TextChunk(
            chunk_id="chunk-3",
            document_id="doc-2",
            chunk_index=0,
            text="A waiting period applies to certain illnesses.",
            page_number=1,
            estimated_tokens=8,
        ),
    ]

    save_chunks("doc-1", chunks[:2], database_path)
    save_chunks("doc-2", chunks[2:], database_path)

    results = get_all_chunks(database_path)

    assert len(results) == 3

    retrieved_ids = {
        chunk["chunk_id"] for chunk in results
    }

    assert retrieved_ids == {
        "chunk-1",
        "chunk-2",
        "chunk-3",
    }
    
def test_get_all_chunks_when_database_is_empty(tmp_path):

    database_path = tmp_path / "test.db"

    initialize_database(database_path)

    results = get_all_chunks(database_path)

    assert results == []
    
def test_get_all_chunks_preserves_order(tmp_path):

    database_path = tmp_path / "test.db"

    initialize_database(database_path)

    create_test_document("doc-1", database_path)

    chunks = [
        TextChunk(
            chunk_id="chunk-3",
            document_id="doc-1",
            chunk_index=2,
            text="Third chunk.",
            page_number=2,
            estimated_tokens=3,
        ),
        TextChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            text="First chunk.",
            page_number=1,
            estimated_tokens=3,
        ),
        TextChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            chunk_index=1,
            text="Second chunk.",
            page_number=1,
            estimated_tokens=3,
        ),
    ]

    save_chunks("doc-1", chunks, database_path)

    results = get_all_chunks(database_path)

    retrieved_ids = [
        chunk["chunk_id"] for chunk in results
    ]

    assert retrieved_ids == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]