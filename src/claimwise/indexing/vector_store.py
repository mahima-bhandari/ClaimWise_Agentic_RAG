
from pathlib import Path
import math

from qdrant_client import QdrantClient, models

from claimwise.config import settings
from claimwise.chunking.text_chunker import TextChunk


class VectorStoreError(RuntimeError):
    """Raised when a vector store operation fails."""


class VectorStore:
    """
    Stores and searches document chunk embeddings using Qdrant.
    """

    def __init__(
        self,
        collection_name: str = "claimwise_chunks",
        vector_size: int = 768,
        storage_path: str | Path | None = None,
        client: QdrantClient | None = None,
    ):
        if vector_size <= 0:
            raise ValueError("Vector size must be positive.")

        self.collection_name = collection_name
        self.vector_size = vector_size

        self.storage_path = Path(
            storage_path
            or settings.absolute_data_dir / "indexes" / "qdrant"
        )

        self.client = client or QdrantClient(
            path=str(self.storage_path)
        )

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert_vectors(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> int:
        """
        Insert or update chunk embeddings in Qdrant.

        Each embedding is associated with its corresponding chunk.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return 0

        points = []

        for chunk, embedding in zip(chunks, embeddings):

            if len(embedding) != self.vector_size:
                raise ValueError(
                    f"Expected vectors with {self.vector_size} dimensions, "
                    f"received {len(embedding)}."
                )

            if not all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                for value in embedding
            ):
                raise ValueError(
                    "Embedding vectors must contain finite numerical values."
                )

            points.append(
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding,
                    payload={
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "page_number": chunk.page_number,
                    },
                )
            )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        except Exception as error:
            raise VectorStoreError(
                f"Failed to insert vectors into Qdrant: {error}"
            ) from error

        return len(points)

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        document_id: str | None = None,
    ) -> list:
        """
        Find the most similar vectors in Qdrant.

        Optionally restrict the search to one document.
        """

        if len(query_vector) != self.vector_size:
            raise ValueError(
                f"Expected a query vector with {self.vector_size} dimensions."
            )

        if not all(
            isinstance(value, (int, float))
            and math.isfinite(value)
            for value in query_vector
        ):
            raise ValueError(
                "Query vector must contain finite numerical values."
            )

        if limit <= 0:
            raise ValueError("Search limit must be positive.")

        query_filter = None

        if document_id is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

            return response.points

        except Exception as error:
            raise VectorStoreError(
                f"Qdrant similarity search failed: {error}"
            ) from error

    def delete_by_document(self, document_id: str) -> None:
        """Delete all vectors belonging to a document."""

        if not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(
                                    value=document_id
                                ),
                            )
                        ]
                    )
                ),
                wait=True,
            )

        except Exception as error:
            raise VectorStoreError(
                f"Failed to delete document vectors: {error}"
            ) from error

    def close(self) -> None:
        """Close the Qdrant client."""

        self.client.close()
