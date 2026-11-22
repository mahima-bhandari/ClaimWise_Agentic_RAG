
import shutil
from dataclasses import dataclass
from pathlib import Path

from claimwise.config import settings
from claimwise.ingestion.pdf_loader import extract_pdf
from claimwise.ingestion.metadata import (
    DocumentMetadata,
    create_document_metadata,
)
from claimwise.chunking.text_chunker import (
    TextChunk,
    chunk_document,
)
from claimwise.storage.repository import (
    save_document,
    update_ingestion_status,
)
from claimwise.storage.chunk_repository import (
    save_chunks,
    delete_chunks_for_document,
)
from claimwise.indexing.indexing_service import IndexingService


@dataclass
class IngestionResult:
    """Summary of a successfully ingested document."""

    metadata: DocumentMetadata
    chunk_count: int
    storage_path: str


class IngestionService:
    """
    Coordinates the complete PDF ingestion pipeline.
    """

    def __init__(
        self,
        indexing_service: IndexingService | None = None,
        upload_dir: str | Path | None = None,
        database_path: str | Path | None = None,
    ):
        self.indexing_service = (
            indexing_service or IndexingService()
        )

        self.upload_dir = Path(
            upload_dir
            or settings.absolute_data_dir / "uploads"
        )

        self.database_path = database_path

        self.upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def ingest_pdf(
        self,
        pdf_path: str | Path,
        source_type: str = "user_upload",
        policy_group_id: str | None = None,
        version_label: str | None = None,
    ) -> IngestionResult:

        pdf_path = Path(pdf_path)

        # Step 1: Extract PDF text.
        extraction_result = extract_pdf(pdf_path)

        # Step 2: Create document metadata.
        metadata = create_document_metadata(
            extraction_result=extraction_result,
            source_type=source_type,
            policy_group_id=policy_group_id,
            version_label=version_label,
        )

        # Step 3: Copy original PDF into ClaimWise storage.
        storage_path = self.upload_dir / (
            f"{metadata.document_id}{pdf_path.suffix.lower()}"
        )

        shutil.copy2(pdf_path, storage_path)

        document_saved = False

        try:
            # Step 4: Save document metadata.
            save_document(
                metadata=metadata,
                storage_path=str(storage_path),
                database_path=self.database_path,
            )

            document_saved = True

            update_ingestion_status(
                document_id=metadata.document_id,
                status="processing",
                database_path=self.database_path,
            )

            # Step 5: Create text chunks.
            chunks = chunk_document(
                extraction_result=extraction_result,
                document_id=metadata.document_id,
            )

            # Step 6: Save chunks in SQLite.
            saved_count = save_chunks(
                document_id=metadata.document_id,
                chunks=chunks,
                database_path=self.database_path,
            )

            # Step 7: Generate embeddings and index in Qdrant.
            indexed_count = self.indexing_service.index_chunks(
                chunks
            )

            if indexed_count != saved_count:
                raise RuntimeError(
                    "The number of indexed chunks does not match "
                    "the number of saved chunks."
                )

            # Step 8: Mark ingestion as completed.
            update_ingestion_status(
                document_id=metadata.document_id,
                status="completed",
                chunk_count=saved_count,
                error_message=None,
                database_path=self.database_path,
            )

            return IngestionResult(
                metadata=metadata,
                chunk_count=saved_count,
                storage_path=str(storage_path),
            )

        except Exception as error:

            if document_saved:

                try:
                    self.indexing_service.delete_document_vectors(
                        metadata.document_id
                    )
                except Exception:
                    pass

                try:
                    delete_chunks_for_document(
                        metadata.document_id,
                        database_path=self.database_path,
                    )
                except Exception:
                    pass

                try:
                    update_ingestion_status(
                        document_id=metadata.document_id,
                        status="failed",
                        error_message=str(error)[:1000],
                        database_path=self.database_path,
                    )
                except Exception:
                    pass

            else:
                if storage_path.exists():
                    storage_path.unlink()

            raise

    def close(self) -> None:
        """Close the underlying indexing service."""

        self.indexing_service.close()
