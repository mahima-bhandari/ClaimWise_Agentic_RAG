
from pathlib import Path

import pymupdf

from claimwise.ingestion.ingestion_service import IngestionService
from claimwise.indexing.embeddings import EmbeddingService
from claimwise.indexing.vector_store import VectorStore
from claimwise.indexing.indexing_service import IndexingService
from claimwise.storage.database import initialize_database
from claimwise.storage.repository import get_document_by_id
from claimwise.storage.chunk_repository import get_chunks_by_document


def create_sample_pdf(pdf_path: Path) -> None:
    """Create a real PDF containing sample insurance policy text."""

    document = pymupdf.open()

    page = document.new_page()

    policy_text = """
    HEALTH INSURANCE POLICY

    Hospitalization Coverage

    The insurance company covers eligible medical expenses incurred
    during hospitalization, subject to the terms and conditions of
    the policy. The insured person must be hospitalized for at least
    twenty-four consecutive hours unless an exception is specified.

    Pre-Hospitalization Expenses

    Medical expenses incurred before hospitalization are covered
    for up to sixty days immediately preceding admission, provided
    they are related to the same illness or injury.

    Post-Hospitalization Expenses

    Eligible medical expenses incurred after discharge are covered
    for up to ninety days, subject to policy conditions.

    Waiting Period

    A waiting period of thirty days applies from the policy start date
    for illnesses, except accidents. Certain specified illnesses may
    have longer waiting periods.

    Exclusions

    Cosmetic treatment undertaken solely for aesthetic purposes
    is excluded unless medically necessary following an accident
    or illness, subject to the policy terms.
    """

    page.insert_textbox(
        pymupdf.Rect(50, 50, 550, 750),
        policy_text,
        fontsize=11,
    )

    document.save(pdf_path)
    document.close()


def test_real_pdf_ingestion_pipeline(tmp_path):

    # Step 1: Create temporary paths.
    database_path = tmp_path / "claimwise_test.db"
    pdf_path = tmp_path / "sample_policy.pdf"
    upload_dir = tmp_path / "uploads"
    qdrant_path = tmp_path / "qdrant"

    # Step 2: Create the sample PDF.
    create_sample_pdf(pdf_path)

    assert pdf_path.exists()

    # Step 3: Initialize a separate SQLite database.
    initialize_database(database_path)

    # Step 4: Initialize real Ollama and Qdrant services.
    embedding_service = EmbeddingService()

    vector_store = VectorStore(
        collection_name="integration_test_chunks",
        vector_size=768,
        storage_path=qdrant_path,
    )

    indexing_service = IndexingService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    ingestion_service = IngestionService(
        indexing_service=indexing_service,
        upload_dir=upload_dir,
        database_path=database_path,
    )

    try:
        # Step 5: Run the complete ingestion pipeline.
        result = ingestion_service.ingest_pdf(
            pdf_path=pdf_path,
            source_type="user_upload",
            policy_group_id="integration-policy",
            version_label="Test Version 1",
        )

        document_id = result.metadata.document_id

        # Step 6: Verify document metadata in SQLite.
        stored_document = get_document_by_id(
            document_id,
            database_path=database_path,
        )

        assert stored_document is not None
        assert stored_document["ingestion_status"] == "completed"
        assert stored_document["chunk_count"] == result.chunk_count
        assert stored_document["source_type"] == "user_upload"

        # Step 7: Verify original PDF was copied into storage.
        assert Path(result.storage_path).exists()

        # Step 8: Verify chunks were saved in SQLite.
        stored_chunks = get_chunks_by_document(
            document_id,
            database_path=database_path,
        )

        assert len(stored_chunks) == result.chunk_count
        assert len(stored_chunks) > 0

        # Step 9: Verify semantic search using real embeddings.
        search_results = indexing_service.search(
            query="What expenses are covered before hospitalization?",
            limit=3,
            document_id=document_id,
        )

        assert len(search_results) > 0

        assert all(
            point.payload["document_id"] == document_id
            for point in search_results
        )

        # Step 10: Verify that the original PDF text was preserved.
        combined_text = " ".join(
            chunk["text"] for chunk in stored_chunks
        )

        assert "Pre-Hospitalization Expenses" in combined_text
        assert "sixty days" in combined_text

        print("\nIntegration test completed successfully!")
        print("Document ID:", document_id)
        print("Chunks created:", result.chunk_count)
        print("Semantic search results:", len(search_results))

    finally:
        ingestion_service.close()
