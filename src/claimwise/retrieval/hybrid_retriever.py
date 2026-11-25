
from claimwise.indexing.indexing_service import IndexingService
from claimwise.retrieval.bm25_retriever import BM25Retriever
from claimwise.storage.chunk_repository import get_all_chunks


class HybridRetrieverError(RuntimeError):
    """Raised when hybrid retrieval fails."""


class HybridRetriever:
    """
    Combines dense and lexical retrieval using
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: IndexingService | None = None,
        lexical_retriever: BM25Retriever | None = None,
        chunk_loader=None,
        rrf_k: int = 60,
    ):
        if rrf_k <= 0:
            raise ValueError("RRF constant must be positive.")

        self.dense_retriever = (
            dense_retriever or IndexingService()
        )

        self.lexical_retriever = (
            lexical_retriever or BM25Retriever()
        )

        self.chunk_loader = chunk_loader or get_all_chunks
        self.rrf_k = rrf_k

    def _calculate_rrf_score(self, rank: int) -> float:
        """Calculate the reciprocal rank contribution."""

        return 1.0 / (self.rrf_k + rank)

    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_top_k: int = 20,
        lexical_top_k: int = 20,
        document_id: str | None = None,
    ) -> list[dict]:
        """
        Perform dense and lexical retrieval,
        then combine their rankings using RRF.
        """

        # Step 1: Validate inputs.
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if dense_top_k <= 0:
            raise ValueError("dense_top_k must be positive.")

        if lexical_top_k <= 0:
            raise ValueError("lexical_top_k must be positive.")

        if document_id is not None and (
            not isinstance(document_id, str)
            or not document_id.strip()
        ):
            raise ValueError("Document ID cannot be empty.")

        # Step 2: Execute dense retrieval.
        try:
            dense_results = self.dense_retriever.search(
                query=query,
                limit=dense_top_k,
                document_id=document_id,
            )

        except Exception as error:
            raise HybridRetrieverError(
                f"Dense retrieval failed: {error}"
            ) from error

        # Step 3: Execute lexical retrieval.
        try:
            lexical_results = self.lexical_retriever.search(
                query=query,
                top_k=lexical_top_k,
                document_id=document_id,
            )

        except Exception as error:
            raise HybridRetrieverError(
                f"Lexical retrieval failed: {error}"
            ) from error

        # Step 4: Load canonical chunk metadata.
        try:
            chunks = self.chunk_loader()

        except Exception as error:
            raise HybridRetrieverError(
                f"Failed to load chunk metadata: {error}"
            ) from error

        chunk_lookup = {
            chunk["chunk_id"]: chunk
            for chunk in chunks
        }

        # Step 5: Prepare the combined result collection.
        combined_results = {}

        # Step 6: Add dense retrieval rankings.
        for rank, point in enumerate(dense_results, start=1):

            chunk_id = str(point.id)

            if chunk_id not in chunk_lookup:
                raise HybridRetrieverError(
                    f"Chunk metadata not found for: {chunk_id}"
                )

            result = combined_results.setdefault(
                chunk_id,
                {
                    **chunk_lookup[chunk_id],
                    "dense_rank": None,
                    "lexical_rank": None,
                    "dense_score": None,
                    "lexical_score": None,
                    "rrf_score": 0.0,
                },
            )

            result["dense_rank"] = rank
            result["dense_score"] = float(point.score)

            result["rrf_score"] += (
                self._calculate_rrf_score(rank)
            )

        # Step 7: Add lexical retrieval rankings.
        for rank, chunk in enumerate(lexical_results, start=1):

            chunk_id = chunk.get("chunk_id")

            if not isinstance(chunk_id, str) or not chunk_id:
                raise HybridRetrieverError(
                    "Lexical result contains an invalid chunk ID."
                )

            if chunk_id not in chunk_lookup:
                raise HybridRetrieverError(
                    f"Chunk metadata not found for: {chunk_id}"
                )

            result = combined_results.setdefault(
                chunk_id,
                {
                    **chunk_lookup[chunk_id],
                    "dense_rank": None,
                    "lexical_rank": None,
                    "dense_score": None,
                    "lexical_score": None,
                    "rrf_score": 0.0,
                },
            )

            result["lexical_rank"] = rank
            result["lexical_score"] = float(chunk["score"])

            result["rrf_score"] += (
                self._calculate_rrf_score(rank)
            )

        # Step 8: Sort by combined RRF score.
        ranked_results = sorted(
            combined_results.values(),
            key=lambda result: result["rrf_score"],
            reverse=True,
        )

        # Step 9: Return the requested number of results.
        return ranked_results[:top_k]
