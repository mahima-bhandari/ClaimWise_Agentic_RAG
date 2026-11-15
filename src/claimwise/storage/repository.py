
import sqlite3
from pathlib import Path

from claimwise.ingestion.metadata import DocumentMetadata
from claimwise.storage.database import (
    get_connection,
)


VALID_INGESTION_STATUSES = {
    "pending",
    "processing",
    "completed",
    "failed",
}


class DuplicateDocumentError(Exception):
    """Raised when an identical document already exists."""


def save_document(
    metadata: DocumentMetadata,
    storage_path: str | Path,
    database_path: str | Path | None = None,
) -> str:
    """
    Save document metadata and its local PDF storage path.

    Returns the document ID.
    """

    connection = get_connection(database_path)

    try:
        # Check whether this document already exists.
        existing = connection.execute(
            """
            SELECT document_id
            FROM documents
            WHERE content_hash = ?
            AND source_type = ?
            """,
            (
                metadata.content_hash,
                metadata.source_type,
            ),
        ).fetchone()

        if existing:
            raise DuplicateDocumentError(
                f"Document already exists with ID: "
                f"{existing['document_id']}"
            )

        connection.execute(
            """
            INSERT INTO documents (
                document_id,
                content_hash,
                file_name,
                file_size_bytes,
                file_extension,
                uploaded_at,
                source_type,
                title,
                author,
                total_pages,
                policy_group_id,
                version_label,
                storage_path,
                ingestion_status,
                chunk_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.document_id,
                metadata.content_hash,
                metadata.file_name,
                metadata.file_size_bytes,
                metadata.file_extension,
                metadata.uploaded_at,
                metadata.source_type,
                metadata.title,
                metadata.author,
                metadata.total_pages,
                metadata.policy_group_id,
                metadata.version_label,
                str(Path(storage_path).resolve()),
                "pending",
                0,
            ),
        )

        connection.commit()

        return metadata.document_id

    except sqlite3.IntegrityError as error:
        connection.rollback()

        raise DuplicateDocumentError(
            "Document could not be saved because it violates "
            "a database uniqueness or validation constraint."
        ) from error

    finally:
        connection.close()


def get_document_by_id(
    document_id: str,
    database_path: str | Path | None = None,
) -> dict | None:
    """Retrieve one document using its ID."""

    connection = get_connection(database_path)

    try:
        result = connection.execute(
            """
            SELECT *
            FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

        return dict(result) if result else None

    finally:
        connection.close()


def get_document_by_hash(
    content_hash: str,
    source_type: str,
    database_path: str | Path | None = None,
) -> dict | None:
    """Find an existing document using its content hash."""

    connection = get_connection(database_path)

    try:
        result = connection.execute(
            """
            SELECT *
            FROM documents
            WHERE content_hash = ?
            AND source_type = ?
            """,
            (content_hash, source_type),
        ).fetchone()

        return dict(result) if result else None

    finally:
        connection.close()


def list_documents(
    limit: int = 100,
    offset: int = 0,
    database_path: str | Path | None = None,
) -> list[dict]:
    """Return documents, newest uploads first."""

    if limit < 1 or offset < 0:
        raise ValueError(
            "Limit must be positive and offset cannot be negative."
        )

    connection = get_connection(database_path)

    try:
        results = connection.execute(
            """
            SELECT *
            FROM documents
            ORDER BY uploaded_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        return [dict(row) for row in results]

    finally:
        connection.close()


def update_ingestion_status(
    document_id: str,
    status: str,
    chunk_count: int | None = None,
    error_message: str | None = None,
    database_path: str | Path | None = None,
) -> bool:
    """Update a document's ingestion status."""

    if status not in VALID_INGESTION_STATUSES:
        raise ValueError(
            f"Invalid ingestion status: {status}"
        )

    if chunk_count is not None and chunk_count < 0:
        raise ValueError("Chunk count cannot be negative.")

    connection = get_connection(database_path)

    try:
        existing = connection.execute(
            """
            SELECT document_id
            FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

        if existing is None:
            return False

        if chunk_count is None:
            connection.execute(
                """
                UPDATE documents
                SET ingestion_status = ?,
                    error_message = ?
                WHERE document_id = ?
                """,
                (status, error_message, document_id),
            )
        else:
            connection.execute(
                """
                UPDATE documents
                SET ingestion_status = ?,
                    error_message = ?,
                    chunk_count = ?
                WHERE document_id = ?
                """,
                (
                    status,
                    error_message,
                    chunk_count,
                    document_id,
                ),
            )

        connection.commit()

        return True

    finally:
        connection.close()


def delete_document(
    document_id: str,
    database_path: str | Path | None = None,
) -> bool:
    """
    Delete a document's database record only.

    Original files and search indexes are not deleted here.
    """

    connection = get_connection(database_path)

    try:
        cursor = connection.execute(
            """
            DELETE FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        )

        connection.commit()

        return cursor.rowcount > 0

    finally:
        connection.close()
