
from pathlib import Path

from claimwise.chunking.text_chunker import TextChunk
from claimwise.storage.database import get_connection


def save_chunks(
    document_id: str,
    chunks: list[TextChunk],
    database_path: str | Path | None = None,
) -> int:
    """
    Save all chunks belonging to a document.

    Returns the number of chunks saved.
    """

    if not document_id.strip():
        raise ValueError("Document ID cannot be empty.")

    connection = get_connection(database_path)

    try:
        # Step 1: Verify that the document exists.
        document = connection.execute(
            """
            SELECT document_id
            FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

        if document is None:
            raise ValueError(
                f"Document not found: {document_id}"
            )

        # Step 2: Verify that all chunks belong to this document.
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError(
                "All chunks must belong to the supplied document ID."
            )

        # Step 3: Handle an empty chunk list.
        if not chunks:
            return 0

        # Step 4: Insert all chunks into the database.
        connection.executemany(
            """
            INSERT INTO document_chunks (
                chunk_id,
                document_id,
                chunk_index,
                text,
                page_number,
                estimated_tokens
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.page_number,
                    chunk.estimated_tokens,
                )
                for chunk in chunks
            ],
        )

        # Step 5: Commit the transaction.
        connection.commit()

        return len(chunks)

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def get_chunks_by_document(
    document_id: str,
    database_path: str | Path | None = None,
) -> list[dict]:
    """Retrieve all chunks for a document in their original order."""

    connection = get_connection(database_path)

    try:
        results = connection.execute(
            """
            SELECT *
            FROM document_chunks
            WHERE document_id = ?
            ORDER BY chunk_index ASC
            """,
            (document_id,),
        ).fetchall()

        return [dict(row) for row in results]

    finally:
        connection.close()


def get_chunk_by_id(
    chunk_id: str,
    database_path: str | Path | None = None,
) -> dict | None:
    """Retrieve one chunk using its unique ID."""

    connection = get_connection(database_path)

    try:
        result = connection.execute(
            """
            SELECT *
            FROM document_chunks
            WHERE chunk_id = ?
            """,
            (chunk_id,),
        ).fetchone()

        return dict(result) if result else None

    finally:
        connection.close()


def delete_chunks_for_document(
    document_id: str,
    database_path: str | Path | None = None,
) -> int:
    """Delete all chunks belonging to a document."""

    connection = get_connection(database_path)

    try:
        cursor = connection.execute(
            """
            DELETE FROM document_chunks
            WHERE document_id = ?
            """,
            (document_id,),
        )

        connection.commit()

        return cursor.rowcount

    finally:
        connection.close()


def get_all_chunks(
    database_path: str | Path | None = None,
) -> list[dict]:
    """
    Retrieve all chunks from all documents.

    Returns chunks ordered by document ID
    and their original chunk order.
    """

    connection = get_connection(database_path)

    try:
        results = connection.execute(
            """
            SELECT
                chunk_id,
                document_id,
                chunk_index,
                text,
                page_number,
                estimated_tokens
            FROM document_chunks
            ORDER BY document_id, chunk_index
            """
        ).fetchall()

        return [dict(row) for row in results]

    finally:
        connection.close()
