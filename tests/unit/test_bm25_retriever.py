
import pytest

from claimwise.retrieval.bm25_retriever import (
    BM25Retriever,
    BM25RetrieverError,
)


@pytest.fixture
def sample_chunks():
    """Provide sample insurance policy chunks."""

    return [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "chunk_index": 0,
            "text": (
                "Pre-existing diseases are subject to "
                "a waiting period of thirty six months."
            ),
            "page_number": 5,
            "estimated_tokens": 14,
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-1",
            "chunk_index": 1,
            "text": (
                "Hospitalization expenses are covered "
                "for eligible medical treatments."
            ),
            "page_number": 8,
            "estimated_tokens": 12,
        },
        {
            "chunk_id": "chunk-3",
            "document_id": "doc-2",
            "chunk_index": 0,
            "text": (
                "Cosmetic surgery is not covered "
                "unless medically necessary."
            ),
            "page_number": 12,
            "estimated_tokens": 10,
        },
        {
            "chunk_id": "chunk-4",
            "document_id": "doc-2",
            "chunk_index": 1,
            "text": (
                "The policy provides coverage for "
                "pre-hospitalization expenses."
            ),
            "page_number": 15,
            "estimated_tokens": 11,
        },
    ]


def test_build_index(sample_chunks):

    retriever = BM25Retriever(
        chunk_loader=lambda: sample_chunks
    )

    indexed_count = retriever.build_index()

    assert indexed_count == 4
    assert retriever.is_indexed is True
    assert retriever.bm25 is not None
    assert len(retriever.chunks) == 4


def test_build_index_with_empty_corpus():

    retriever = BM25Retriever(
        chunk_loader=lambda: []
    )

    indexed_count = retriever.build_index()

    assert indexed_count == 0
    assert retriever.is_indexed is True
    assert retriever.bm25 is None
    assert retriever.chunks == []


def test_search_returns_relevant_chunks(sample_chunks):

    retriever = BM25Retriever(
        chunk_loader=lambda: sample_chunks
    )

    retriever.build_index()

    results = retriever.search(
        query="What is the waiting period for pre-existing diseases?",
        top_k=2,
    )

    assert len(results) == 2

    assert results[0]["chunk_id"] == "chunk-1"

    assert all("score" in result for result in results)


def test_search_respects_top_k(sample_chunks):

    retriever = BM25Retriever(
        chunk_loader=lambda: sample_chunks
    )

    retriever.build_index()

    results = retriever.search(
        query="insurance policy coverage",
        top_k=2,
    )

    assert len(results) == 2


def test_search_filters_by_document_id(sample_chunks):

    retriever = BM25Retriever(
        chunk_loader=lambda: sample_chunks
    )

    retriever.build_index()

    results = retriever.search(
        query="insurance coverage",
        top_k=5,
        document_id="doc-1",
    )

    assert len(results) <= 2

    assert all(
        result["document_id"] == "doc-1"
        for result in results
    )


def test_search_returns_empty_for_unknown_document(sample_chunks):

    retriever = BM25Retriever(
        chunk_loader=lambda: sample_chunks
    )

    retriever.build_index()

    results = retriever.search(
        query="insurance coverage",
        document_id="unknown-doc",
    )

    assert results == []


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        None,
    ],
)
def test_search_rejects_empty_query(sample_chunks, query):

    retriever = BM25Retriever(
        chunk_loader=lambda: sample_chunks
    )

    retriever.build_index()

    with pytest.raises(ValueError, match="Search query"):
        retriever.search(query=query)


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_search_rejects_invalid_top_k(sample_chunks, top_k):

    retriever = BM25Retriever(
        chunk_loader=lambda: sample_chunks
    )

    retriever.build_index()

    with pytest.raises(ValueError, match="top_k"):
        retriever.search(
            query="waiting period",
            top_k=top_k,
        )


def test_search_before_building_index(sample_chunks):

    retriever = BM25Retriever(
        chunk_loader=lambda: sample_chunks
    )

    with pytest.raises(
        BM25RetrieverError,
        match="index has not been built",
    ):
        retriever.search("waiting period")


def test_build_index_handles_loader_failure():

    def failing_loader():
        raise RuntimeError("Database connection failed")

    retriever = BM25Retriever(
        chunk_loader=failing_loader
    )

    with pytest.raises(
        BM25RetrieverError,
        match="Failed to build BM25 index",
    ):
        retriever.build_index()

    assert retriever.is_indexed is False
    assert retriever.bm25 is None
