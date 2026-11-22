
from unittest.mock import Mock
from uuid import uuid4

import pytest

from claimwise.chunking.text_chunker import TextChunk
from claimwise.indexing.indexing_service import IndexingService


@pytest.fixture
def mock_embedding_service():
    return Mock()


@pytest.fixture
def mock_vector_store():
    return Mock()


@pytest.fixture
def indexing_service(
    mock_embedding_service,
    mock_vector_store,
):
    return IndexingService(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
    )


@pytest.fixture
def sample_chunks():
    document_id = str(uuid4())

    return [
        TextChunk(
            chunk_id=str(uuid4()),
            document_id=document_id,
            chunk_index=0,
            text="The policy covers hospitalisation expenses.",
            page_number=1,
            estimated_tokens=8,
        ),
        TextChunk(
            chunk_id=str(uuid4()),
            document_id=document_id,
            chunk_index=1,
            text="The waiting period is thirty days.",
            page_number=2,
            estimated_tokens=7,
        ),
    ]


def test_index_chunks(
    indexing_service,
    mock_embedding_service,
    mock_vector_store,
    sample_chunks,
):
    mock_embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    mock_embedding_service.embed_texts.return_value = (
        mock_embeddings
    )

    mock_vector_store.upsert_vectors.return_value = 2

    result = indexing_service.index_chunks(sample_chunks)

    expected_texts = [
        chunk.text for chunk in sample_chunks
    ]

    mock_embedding_service.embed_texts.assert_called_once_with(
        expected_texts
    )

    mock_vector_store.upsert_vectors.assert_called_once_with(
        chunks=sample_chunks,
        embeddings=mock_embeddings,
    )

    assert result == 2


def test_index_empty_chunks(
    indexing_service,
    mock_embedding_service,
    mock_vector_store,
):
    result = indexing_service.index_chunks([])

    assert result == 0

    mock_embedding_service.embed_texts.assert_not_called()
    mock_vector_store.upsert_vectors.assert_not_called()


def test_search(
    indexing_service,
    mock_embedding_service,
    mock_vector_store,
):
    query = "What is the waiting period?"
    query_embedding = [0.1, 0.2, 0.3]
    mock_results = [Mock(), Mock()]

    mock_embedding_service.embed_text.return_value = (
        query_embedding
    )

    mock_vector_store.search.return_value = mock_results

    result = indexing_service.search(
        query=query,
        limit=5,
        document_id="document-123",
    )

    mock_embedding_service.embed_text.assert_called_once_with(
        query
    )

    mock_vector_store.search.assert_called_once_with(
        query_vector=query_embedding,
        limit=5,
        document_id="document-123",
    )

    assert result == mock_results


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "\n",
        None,
    ],
)
def test_search_rejects_empty_query(
    indexing_service,
    mock_embedding_service,
    query,
):
    with pytest.raises(
        ValueError,
        match="Search query cannot be empty",
    ):
        indexing_service.search(query)

    mock_embedding_service.embed_text.assert_not_called()


@pytest.mark.parametrize(
    "limit",
    [
        0,
        -1,
        -5,
    ],
)
def test_search_rejects_invalid_limit(
    indexing_service,
    mock_embedding_service,
    limit,
):
    with pytest.raises(
        ValueError,
        match="Search limit must be positive",
    ):
        indexing_service.search(
            query="What is covered?",
            limit=limit,
        )

    mock_embedding_service.embed_text.assert_not_called()


def test_delete_document_vectors(
    indexing_service,
    mock_vector_store,
):
    document_id = "document-123"

    indexing_service.delete_document_vectors(document_id)

    mock_vector_store.delete_by_document.assert_called_once_with(
        document_id
    )


def test_close(
    indexing_service,
    mock_vector_store,
):
    indexing_service.close()

    mock_vector_store.close.assert_called_once()
