
import bm25s

from claimwise.storage.chunk_repository import get_all_chunks


class BM25RetrieverError(RuntimeError):
    """Raised when BM25 retrieval fails."""


class BM25Retriever:
    """
    Performs lexical search over document chunks using BM25S.
    """

    def __init__(self, chunk_loader=None):
        self.chunk_loader = chunk_loader or get_all_chunks

        self.chunks = []
        self.bm25 = None

        self.is_indexed = False

    def build_index(self) -> int:
        """
        Load all chunks and build the BM25 index.

        Returns the number of chunks indexed.
        """

        try:
            chunks = self.chunk_loader()

            if not chunks:
                self.chunks = []
                self.bm25 = None
                self.is_indexed = True
                return 0

            # Extract the text of each chunk.
            corpus = [
                chunk["text"]
                for chunk in chunks
            ]

            # Tokenize the corpus.
            # We intentionally retain stopwords because terms
            # such as "not", "no", and "without" matter in policies.
            tokenized_corpus = bm25s.tokenize(
                corpus,
                stopwords=None,
            )

            # Build the BM25 index.
            retriever = bm25s.BM25()
            retriever.index(tokenized_corpus)

            self.chunks = chunks
            self.bm25 = retriever
            self.is_indexed = True

            return len(chunks)

        except Exception as error:
            self.chunks = []
            self.bm25 = None
            self.is_indexed = False

            raise BM25RetrieverError(
                f"Failed to build BM25 index: {error}"
            ) from error

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.

        Optionally restrict the search to one document.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not self.is_indexed:
            raise BM25RetrieverError(
                "BM25 index has not been built. "
                "Call build_index() first."
            )

        # Apply optional document filtering.
        if document_id is not None:
            searchable_chunks = [
                chunk
                for chunk in self.chunks
                if chunk["document_id"] == document_id
            ]
        else:
            searchable_chunks = self.chunks

        if not searchable_chunks:
            return []

        # Reuse the existing index when searching all documents.
        if document_id is None:
            active_bm25 = self.bm25

        else:
            # Build a temporary index for the selected document.
            filtered_corpus = [
                chunk["text"]
                for chunk in searchable_chunks
            ]

            tokenized_corpus = bm25s.tokenize(
                filtered_corpus,
                stopwords=None,
            )

            active_bm25 = bm25s.BM25()
            active_bm25.index(tokenized_corpus)

        # Tokenize the user's query.
        tokenized_query = bm25s.tokenize(
            [query],
            stopwords=None,
        )

        try:
            results, scores = active_bm25.retrieve(
                tokenized_query,
                k=min(top_k, len(searchable_chunks)),
            )

        except Exception as error:
            raise BM25RetrieverError(
                f"BM25 search failed: {error}"
            ) from error

        retrieved_chunks = []

        for index, score in zip(results[0], scores[0]):

            chunk = searchable_chunks[int(index)]

            result = chunk.copy()
            result["score"] = float(score)

            retrieved_chunks.append(result)

        return retrieved_chunks
