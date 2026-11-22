
import math

import ollama

from claimwise.config import settings


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class EmbeddingService:
    """
    Generates text embeddings using a local Ollama model.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        client: ollama.Client | None = None,
    ):
        self.model = model or settings.ollama_embedding_model

        self.client = client or ollama.Client(
            host=base_url or settings.ollama_base_url
        )

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text.
        """

        embeddings = self.embed_texts([text])

        return embeddings[0]

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Returns one vector for each input text,
        preserving the original input order.
        """

        if not texts:
            return []

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "All input texts must be non-empty strings."
            )

        try:
            response = self.client.embed(
                model=self.model,
                input=texts,
            )

        except Exception as error:
            raise EmbeddingError(
                f"Failed to generate embeddings using model "
                f"'{self.model}': {error}"
            ) from error

        embeddings = response.embeddings

        if len(embeddings) != len(texts):
            raise EmbeddingError(
                "The embedding model returned an unexpected "
                "number of vectors."
            )

        if not embeddings:
            raise EmbeddingError(
                "The embedding model returned no vectors."
            )

        expected_dimensions = len(embeddings[0])

        if expected_dimensions == 0:
            raise EmbeddingError(
                "The embedding model returned an empty vector."
            )

        validated_embeddings = []

        for vector in embeddings:

            if len(vector) != expected_dimensions:
                raise EmbeddingError(
                    "Embedding vectors have inconsistent dimensions."
                )

            if not all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                for value in vector
            ):
                raise EmbeddingError(
                    "An embedding vector contains invalid values."
                )

            validated_embeddings.append(
                [float(value) for value in vector]
            )

        return validated_embeddings
