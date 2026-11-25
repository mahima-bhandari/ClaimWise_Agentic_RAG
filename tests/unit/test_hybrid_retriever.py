
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from claimwise.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridRetrieverError,
)


@pytest.fixture
def sample_chunks():
    """Provide canonical chunk metadata."""

    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "text": "Pre-existing diseases have a waiting period of 36 months.",
            "page_number": 5,
            "estimated_tokens": 12,
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-1",
            "chunk_index": 1,
            "text": "Hospitalization expenses are covered.",
            "page_number": 8,
            "estimated_tokens": 7,
        },
        {
            "chunk_id": "chunk-3",
            "document_id": "doc-2",
            "chunk_index": 0,
            "text": "Cosmetic surgery is not covered.",
            "page_number": 12,
            "estimated_tokens": 7,
        },
    ]


@pytest.fixture
def mock_retrievers():
    """Create mock dense and lexical retrievers."""

    dense_retriever = Mock()
    lexical_retriever = Mock()

    return dense_retriever, lexical_retriever


@pytest.fixture
def hybrid_retriever(
    sample_chunks,
    mock_retrievers,
):
    """Create a HybridRetriever with mocked dependencies."""

    dense_retriever, lexical_retriever = mock_retrievers

    return HybridRetriever(
        dense_retriever=dense_retriever,
        lexical_retriever=lexical_retriever,
        chunk_loader=lambda: sample_chunks,
    )


def test_rrf_score_calculation(hybrid_retriever):

    score = hybrid_retriever._calculate_rrf_score(1)

    assert score == pytest.approx(1 / 61)


def test_hybrid_search_combines_results(
    hybrid_retriever,
    mock_retrievers,
):

    dense_retriever, lexical_retriever = mock_retrievers

    dense_retriever.search.return_value = [
        SimpleNamespace(id="chunk-1", score=0.95),
        SimpleNamespace(id="chunk-2", score=0.80),
    ]

    lexical_retriever.search.return_value = [
        {
            "chunk_id": "chunk-2",
            "score": 2.5,
        },
        {
            "chunk_id": "chunk-3",
            "score": 1.8,
        },
    ]

    results = hybrid_retriever.search(
        query="insurance coverage",
        top_k=3,
    )

    assert len(results) == 3

    result_ids = [
        result["chunk_id"] for result in results
    ]

    assert set(result_ids) == {
        "chunk-1",
        "chunk-2",
        "chunk-3",
    }


def test_chunk_found_in_both_retrievers_gets_combined_score(
    hybrid_retriever,
    mock_retrievers,
):

    dense_retriever, lexical_retriever = mock_retrievers

    dense_retriever.search.return_value = [
        SimpleNamespace(id="chunk-1", score=0.95),
    ]

    lexical_retriever.search.return_value = [
        {
            "chunk_id": "chunk-1",
            "score": 2.5,
        },
    ]

    results = hybrid_retriever.search(
        query="waiting period",
    )

    assert len(results) == 1

    result = results[0]

    expected_score = (1 / 61) + (1 / 61)

    assert result["rrf_score"] == pytest.approx(expected_score)
    assert result["dense_rank"] == 1
    assert result["lexical_rank"] == 1


def test_hybrid_results_are_sorted_by_rrf_score(
    hybrid_retriever,
    mock_retrievers,
):

    dense_retriever, lexical_retriever = mock_retrievers

    dense_retriever.search.return_value = [
        SimpleNamespace(id="chunk-1", score=0.95),
        SimpleNamespace(id="chunk-2", score=0.80),
    ]

    lexical_retriever.search.return_value = [
        {
            "chunk_id": "chunk-2",
            "score": 2.5,
        },
        {
            "chunk_id": "chunk-1",
            "score": 1.8,
        },
    ]

    results = hybrid_retriever.search(
        query="insurance coverage",
    )

    assert results[0]["chunk_id"] == "chunk-1"
    assert results[1]["chunk_id"] == "chunk-2"

    assert (
        results[0]["rrf_score"]
        == pytest.approx((1 / 61) + (1 / 62))
    )

    assert (
        results[1]["rrf_score"]
        == pytest.approx((1 / 62) + (1 / 61))
    )


def test_hybrid_search_handles_dense_only_results(
    hybrid_retriever,
    mock_retrievers,
):

    dense_retriever, lexical_retriever = mock_retrievers

    dense_retriever.search.return_value = [
        SimpleNamespace(id="chunk-1", score=0.95),
    ]

    lexical_retriever.search.return_value = []

    results = hybrid_retriever.search(
        query="waiting period",
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[0]["dense_rank"] == 1
    assert results[0]["lexical_rank"] is None


def test_hybrid_search_handles_lexical_only_results(
    hybrid_retriever,
    mock_retrievers,
):

    dense_retriever, lexical_retriever = mock_retrievers

    dense_retriever.search.return_value = []

    lexical_retriever.search.return_value = [
        {
            "chunk_id": "chunk-2",
            "score": 2.5,
        },
    ]

    results = hybrid_retriever.search(
        query="hospitalization expenses",
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk-2"
    assert results[0]["dense_rank"] is None
    assert results[0]["lexical_rank"] == 1


def test_hybrid_search_handles_empty_results(
    hybrid_retriever,
    mock_retrievers,
):

    dense_retriever, lexical_retriever = mock_retrievers

    dense_retriever.search.return_value = []
    lexical_retriever.search.return_value = []

    results = hybrid_retriever.search(
        query="unknown policy clause",
    )

    assert results == []


def test_hybrid_search_respects_top_k(
    hybrid_retriever,
    mock_retrievers,
):

    dense_retriever, lexical_retriever = mock_retrievers

    dense_retriever.search.return_value = [
        SimpleNamespace(id="chunk-1", score=0.95),
        SimpleNamespace(id="chunk-2", score=0.80),
    ]

    lexical_retriever.search.return_value = [
        {
            "chunk_id": "chunk-3",
            "score": 2.5,
        },
    ]

    results = hybrid_retriever.search(
        query="insurance",
        top_k=2,
    )

    assert len(results) == 2


def test_document_filter_is_passed_to_both_retrievers(
    hybrid_retriever,
    mock_retrievers,
):

    dense_retriever, lexical_retriever = mock_retrievers

    dense_retriever.search.return_value = []
    lexical_retriever.search.return_value = []

    hybrid_retriever.search(
        query="waiting period",
        document_id="doc-1",
    )

    dense_retriever.search.assert_called_once_with(
        query="waiting period",
        limit=20,
        document_id="doc-1",
    )

    lexical_retriever.search.assert_called_once_with(
        query="waiting period",
        top_k=20,
        document_id="doc-1",
    )


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        None,
    ],
)
def test_search_rejects_empty_query(
    hybrid_retriever,
    query,
):

    with pytest.raises(ValueError, match="Search query"):
        hybrid_retriever.search(query=query)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("top_k", 0),
        ("top_k", -1),
        ("dense_top_k", 0),
        ("dense_top_k", -1),
        ("lexical_top_k", 0),
        ("lexical_top_k", -1),
    ],
)
def test_search_rejects_invalid_limits(
    hybrid_retriever,
    parameter,
    value,
):

    with pytest.raises(ValueError):
        hybrid_retriever.search(
            query="waiting period",
            **{parameter: value},
        )


def test_search_rejects_empty_document_id(hybrid_retriever):

    with pytest.raises(ValueError, match="Document ID"):
        hybrid_retriever.search(
            query="waiting period",
            document_id="   ",
        )


@pytest.mark.parametrize(
    "rrf_k",
    [
        0,
        -1,
    ],
)
def test_constructor_rejects_invalid_rrf_k(rrf_k):

    with pytest.raises(ValueError, match="RRF constant"):
        HybridRetriever(
            dense_retriever=Mock(),
            lexical_retriever=Mock(),
            rrf_k=rrf_k,
        )


def test_search_raises_when_dense_retrieval_fails(
    hybrid_retriever,
    mock_retrievers,
):

    dense_retriever, _ = mock_retrievers

    dense_retriever.search.side_effect = RuntimeError(
        "Qdrant connection failed"
    )

    with pytest.raises(
        HybridRetrieverError,
        match="Dense retrieval failed",
    ):
        hybrid_retriever.search("waiting period")


def test_search_raises_when_lexical_retrieval_fails(
    hybrid_retriever,
    mock_retrievers,
):

    _, lexical_retriever = mock_retrievers

    lexical_retriever.search.side_effect = RuntimeError(
        "BM25 index failed"
    )

    with pytest.raises(
        HybridRetrieverError,
        match="Lexical retrieval failed",
    ):
        hybrid_retriever.search("waiting period")


def test_search_raises_when_chunk_loader_fails(
    sample_chunks,
    mock_retrievers,
):

    dense_retriever, lexical_retriever = mock_retrievers

    dense_retriever.search.return_value = [
        SimpleNamespace(id="chunk-1", score=0.95),
    ]

    lexical_retriever.search.return_value = []

    def failing_loader():
        raise RuntimeError("SQLite connection failed")

    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        lexical_retriever=lexical_retriever,
        chunk_loader=failing_loader,
    )

    with pytest.raises(
        HybridRetrieverError,
        match="Failed to load chunk metadata",
    ):
        retriever.search("waiting period")


def test_search_raises_when_dense_chunk_metadata_is_missing(
    hybrid_retriever,
    mock_retrievers,
):

    dense_retriever, lexical_retriever = mock_retrievers

    dense_retriever.search.return_value = [
        SimpleNamespace(id="unknown-chunk", score=0.95),
    ]

    lexical_retriever.search.return_value = []

    with pytest.raises(
        HybridRetrieverError,
        match="Chunk metadata not found",
    ):
        hybrid_retriever.search("waiting period")


def test_search_raises_when_lexical_chunk_id_is_invalid(
    hybrid_retriever,
    mock_retrievers,
):

    dense_retriever, lexical_retriever = mock_retrievers

    dense_retriever.search.return_value = []

    lexical_retriever.search.return_value = [
        {
            "chunk_id": None,
            "score": 2.5,
        },
    ]

    with pytest.raises(
        HybridRetrieverError,
        match="invalid chunk ID",
    ):
        hybrid_retriever.search("waiting period")


def test_search_raises_when_lexical_chunk_metadata_is_missing(
    hybrid_retriever,
    mock_retrievers,
):

    dense_retriever, lexical_retriever = mock_retrievers

    dense_retriever.search.return_value = []

    lexical_retriever.search.return_value = [
        {
            "chunk_id": "unknown-chunk",
            "score": 2.5,
        },
    ]

    with pytest.raises(
        HybridRetrieverError,
        match="Chunk metadata not found",
    ):
        hybrid_retriever.search("waiting period")
