
from unittest.mock import Mock

import pytest

from claimwise.indexing.embeddings import (
    EmbeddingService,
    EmbeddingError,
)


@pytest.fixture
def mock_ollama_client():
    return Mock()


@pytest.fixture
def embedding_service(mock_ollama_client):
    return EmbeddingService(
        model="nomic-embed-text",
        client=mock_ollama_client,
    )


def test_embed_single_text(
    embedding_service,
    mock_ollama_client,
):
    mock_ollama_client.embed.return_value = Mock(
        embeddings=[[0.1, 0.2, 0.3]]
    )

    result = embedding_service.embed_text(
        "Room rent is subject to policy limits."
    )

    assert result == [0.1, 0.2, 0.3]

    mock_ollama_client.embed.assert_called_once_with(
        model="nomic-embed-text",
        input=["Room rent is subject to policy limits."],
    )


def test_embed_multiple_texts(
    embedding_service,
    mock_ollama_client,
):
    mock_ollama_client.embed.return_value = Mock(
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )

    texts = [
        "Room rent is subject to policy limits.",
        "Pre-existing diseases have a waiting period.",
    ]

    result = embedding_service.embed_texts(texts)

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]

    mock_ollama_client.embed.assert_called_once_with(
        model="nomic-embed-text",
        input=texts,
    )


def test_empty_text_list_returns_empty_list(
    embedding_service,
    mock_ollama_client,
):
    result = embedding_service.embed_texts([])

    assert result == []

    mock_ollama_client.embed.assert_not_called()


def test_empty_text_is_rejected(embedding_service):
    with pytest.raises(ValueError, match="non-empty strings"):
        embedding_service.embed_texts(
            ["Valid text", "   "]
        )


def test_ollama_failure_raises_embedding_error(
    embedding_service,
    mock_ollama_client,
):
    mock_ollama_client.embed.side_effect = Exception(
        "Ollama is unavailable"
    )

    with pytest.raises(EmbeddingError, match="Failed to generate"):
        embedding_service.embed_texts(["Some policy text"])


def test_wrong_number_of_vectors_is_rejected(
    embedding_service,
    mock_ollama_client,
):
    mock_ollama_client.embed.return_value = Mock(
        embeddings=[[0.1, 0.2, 0.3]]
    )

    with pytest.raises(EmbeddingError, match="unexpected number"):
        embedding_service.embed_texts(
            ["First text", "Second text"]
        )


def test_inconsistent_vector_dimensions_are_rejected(
    embedding_service,
    mock_ollama_client,
):
    mock_ollama_client.embed.return_value = Mock(
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5],
        ]
    )

    with pytest.raises(EmbeddingError, match="inconsistent dimensions"):
        embedding_service.embed_texts(
            ["First text", "Second text"]
        )


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_invalid_embedding_values_are_rejected(
    embedding_service,
    mock_ollama_client,
    invalid_value,
):
    mock_ollama_client.embed.return_value = Mock(
        embeddings=[
            [0.1, invalid_value, 0.3]
        ]
    )

    with pytest.raises(EmbeddingError, match="invalid values"):
        embedding_service.embed_texts(["Some policy text"])


def test_empty_embedding_vector_is_rejected(
    embedding_service,
    mock_ollama_client,
):
    mock_ollama_client.embed.return_value = Mock(
        embeddings=[[]]
    )

    with pytest.raises(EmbeddingError, match="empty vector"):
        embedding_service.embed_texts(["Some policy text"])
