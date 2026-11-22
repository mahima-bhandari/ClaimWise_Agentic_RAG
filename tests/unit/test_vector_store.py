
import pytest
from uuid import uuid4

from claimwise.indexing.vector_store import (
    VectorStore,
    VectorStoreError,
)
from claimwise.chunking.text_chunker import TextChunk


@pytest.fixture
def vector_store():
    store = VectorStore(
        collection_name=f"test_chunks_{uuid4().hex}",
        vector_size=3,
        client=__import__(
            "qdrant_client"
        ).QdrantClient(":memory:"),
    )

    yield store

    store.close()


@pytest.fixture
def sample_chunks():
    return [
        TextChunk(
            chunk_id=str(uuid4()),
            document_id="doc-1",
            chunk_index=0,
            text="Room rent is subject to policy limits.",
            page_number=1,
            estimated_tokens=10,
        ),
        TextChunk(
            chunk_id=str(uuid4()),
            document_id="doc-1",
            chunk_index=1,
            text="Pre-existing diseases have a waiting period.",
            page_number=2,
            estimated_tokens=12,
        ),
        TextChunk(
            chunk_id=str(uuid4()),
            document_id="doc-2",
            chunk_index=0,
            text="Day-care procedures are covered.",
            page_number=3,
            estimated_tokens=8,
        ),
    ]


@pytest.fixture
def sample_embeddings():
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_collection_is_created(vector_store):
    assert vector_store.client.collection_exists(
        vector_store.collection_name
    )


def test_upsert_vectors(
    vector_store,
    sample_chunks,
    sample_embeddings,
):
    result = vector_store.upsert_vectors(
        sample_chunks,
        sample_embeddings,
    )

    assert result == 3

    collection_info = vector_store.client.get_collection(
        vector_store.collection_name
    )

    assert collection_info.points_count == 3


def test_upsert_rejects_mismatched_counts(
    vector_store,
    sample_chunks,
):
    with pytest.raises(ValueError, match="number of chunks"):
        vector_store.upsert_vectors(
            sample_chunks,
            [[1.0, 0.0, 0.0]],
        )


def test_upsert_rejects_wrong_vector_dimensions(
    vector_store,
    sample_chunks,
):
    with pytest.raises(ValueError, match="dimensions"):
        vector_store.upsert_vectors(
            [sample_chunks[0]],
            [[1.0, 0.0]],
        )


def test_search_returns_similar_vectors(
    vector_store,
    sample_chunks,
    sample_embeddings,
):
    vector_store.upsert_vectors(
        sample_chunks,
        sample_embeddings,
    )

    results = vector_store.search(
        query_vector=[1.0, 0.0, 0.0],
        limit=2,
    )

    assert len(results) == 2
    assert results[0].id == sample_chunks[0].chunk_id
    assert results[0].payload["document_id"] == "doc-1"


def test_search_can_filter_by_document(
    vector_store,
    sample_chunks,
    sample_embeddings,
):
    vector_store.upsert_vectors(
        sample_chunks,
        sample_embeddings,
    )

    results = vector_store.search(
        query_vector=[0.0, 0.0, 1.0],
        limit=5,
        document_id="doc-1",
    )

    assert len(results) == 2

    assert all(
        result.payload["document_id"] == "doc-1"
        for result in results
    )


def test_delete_by_document(
    vector_store,
    sample_chunks,
    sample_embeddings,
):
    vector_store.upsert_vectors(
        sample_chunks,
        sample_embeddings,
    )

    vector_store.delete_by_document("doc-1")

    remaining_results = vector_store.search(
        query_vector=[1.0, 0.0, 0.0],
        limit=5,
    )

    assert len(remaining_results) == 1
    assert remaining_results[0].payload["document_id"] == "doc-2"


def test_delete_rejects_empty_document_id(vector_store):
    with pytest.raises(ValueError, match="Document ID"):
        vector_store.delete_by_document("   ")


def test_empty_chunks_return_zero(vector_store):
    result = vector_store.upsert_vectors([], [])

    assert result == 0


def test_search_rejects_invalid_limit(vector_store):
    with pytest.raises(ValueError, match="positive"):
        vector_store.search(
            query_vector=[1.0, 0.0, 0.0],
            limit=0,
        )


def test_search_rejects_wrong_vector_dimensions(vector_store):
    with pytest.raises(ValueError, match="dimensions"):
        vector_store.search(
            query_vector=[1.0, 0.0],
        )
