
from claimwise.chunking.text_chunker import TextChunk
from claimwise.indexing.embeddings import EmbeddingService
from claimwise.indexing.vector_store import VectorStore


class IndexingService:
    """
    Coordinates embedding generation and Qdrant indexing.

    Also converts natural-language questions into vectors
    and retrieves semantically similar document chunks.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.embedding_service = (
            embedding_service or EmbeddingService()
        )

        self.vector_store = (
            vector_store or VectorStore()
        )

    def index_chunks(
        self,
        chunks: list[TextChunk],
    ) -> int:
        """
        Generate embeddings for document chunks
        and store them in Qdrant.

        Returns the number of indexed chunks.
        """

        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedding_service.embed_texts(
            texts
        )

        indexed_count = self.vector_store.upsert_vectors(
            chunks=chunks,
            embeddings=embeddings,
        )

        return indexed_count

    def search(
        self,
        query: str,
        limit: int = 5,
        document_id: str | None = None,
    ) -> list:
        """
        Search for semantically similar document chunks.

        Optionally restrict the search to a particular document.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        if limit <= 0:
            raise ValueError(
                "Search limit must be positive."
            )

        query_embedding = self.embedding_service.embed_text(
            query
        )

        results = self.vector_store.search(
            query_vector=query_embedding,
            limit=limit,
            document_id=document_id,
        )

        return results

    def delete_document_vectors(
        self,
        document_id: str,
    ) -> None:
        """
        Remove all indexed vectors belonging to a document.
        """

        self.vector_store.delete_by_document(
            document_id
        )

    def close(self) -> None:
        """Close the underlying vector store connection."""

        self.vector_store.close()
