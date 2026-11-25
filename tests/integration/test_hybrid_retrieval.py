
from pathlib import Path

import pymupdf

from claimwise.ingestion.ingestion_service import IngestionService
from claimwise.indexing.embeddings import EmbeddingService
from claimwise.indexing.indexing_service import IndexingService
from claimwise.indexing.vector_store import VectorStore
from claimwise.retrieval.bm25_retriever import BM25Retriever
from claimwise.retrieval.hybrid_retriever import HybridRetriever
from claimwise.storage.chunk_repository import get_all_chunks
from claimwise.storage.database import initialize_database


def create_sample_policy(pdf_path: Path) -> None:
    """Create a sample insurance policy with multiple pages."""

    document = pymupdf.open()

    policy_pages = [
        """
        PRE-EXISTING DISEASES

        Pre-existing diseases are subject to a waiting period
        of thirty six months from the policy commencement date.

        Expenses related to pre-existing diseases are covered
        only after completion of the applicable waiting period,
        subject to policy terms and conditions.
        """,
        """
        HOSPITALIZATION COVERAGE

        Eligible hospitalization expenses are covered under
        the policy when the insured person is hospitalized
        for at least twenty four consecutive hours.

        Coverage is subject to the applicable policy limits
        and conditions.
        """,
        """
        PRE-HOSPITALIZATION EXPENSES

        Medical expenses incurred during the sixty days
        immediately preceding hospitalization are covered.

        Such expenses must be related to the same illness
        or injury for which hospitalization is required.
        """,
    ]

    for policy_text in policy_pages:

        page = document.new_page()

        page.insert_textbox(
            pymupdf.Rect(50, 50, 550, 750),
            policy_text,
            fontsize=12,
        )

    document.save(pdf_path)
    document.close()


def test_real_hybrid_retrieval_pipeline(tmp_path):

    # Step 1: Prepare isolated test paths.
    database_path = tmp_path / "claimwise_test.db"
    pdf_path = tmp_path / "sample_policy.pdf"
    upload_dir = tmp_path / "uploads"
    qdrant_path = tmp_path / "qdrant"

    create_sample_policy(pdf_path)

    initialize_database(database_path)

    # Step 2: Initialize real embedding and vector services.
    embedding_service = EmbeddingService()

    vector_store = VectorStore(
        collection_name="hybrid_integration_test",
        vector_size=768,
        storage_path=qdrant_path,
    )

    indexing_service = IndexingService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    # Step 3: Initialize the ingestion service.
    ingestion_service = IngestionService(
        indexing_service=indexing_service,
        upload_dir=upload_dir,
        database_path=database_path,
    )

    try:
        # Step 4: Ingest the real PDF.
        ingestion_result = ingestion_service.ingest_pdf(
            pdf_path=pdf_path,
            source_type="user_upload",
            policy_group_id="hybrid-test-policy",
            version_label="Version 1",
        )

        document_id = ingestion_result.metadata.document_id

        assert ingestion_result.chunk_count == 3

        # Step 5: Create the BM25 retriever using the test database.
        bm25_retriever = BM25Retriever(
            chunk_loader=lambda: get_all_chunks(database_path)
        )

        indexed_count = bm25_retriever.build_index()

        assert indexed_count == 3

        # Step 6: Initialize the hybrid retriever.
        hybrid_retriever = HybridRetriever(
            dense_retriever=indexing_service,
            lexical_retriever=bm25_retriever,
            chunk_loader=lambda: get_all_chunks(database_path),
        )

        # Step 7: Execute a real hybrid search.
        results = hybrid_retriever.search(
            query="What is the waiting period for pre-existing diseases?",
            top_k=3,
            document_id=document_id,
        )

        # Step 8: Verify the retrieved results.
        assert len(results) > 0

        assert all(
            result["document_id"] == document_id
            for result in results
        )

        assert all(
            result["rrf_score"] > 0
            for result in results
        )

        # Step 9: Verify that the relevant policy clause was retrieved.
        assert "pre-existing diseases" in results[0]["text"].lower()

        # Step 10: Verify retrieval metadata.
        assert results[0]["dense_rank"] is not None
        assert results[0]["lexical_rank"] is not None

        print("\nHybrid retrieval integration test passed!")
        print("Document ID:", document_id)
        print("Chunks indexed:", indexed_count)
        print("Hybrid results:", len(results))
        print("Top result:", results[0]["chunk_id"])
        print("RRF score:", results[0]["rrf_score"])

    finally:
        ingestion_service.close()
