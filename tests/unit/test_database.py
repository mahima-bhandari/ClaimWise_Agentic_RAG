import pytest
import sqlite3

from claimwise.storage.database import (
    get_connection,
    initialize_database,
)


def test_database_initialization(tmp_path):
    database_path = tmp_path / "test_claimwise.db"

    initialize_database(database_path)

    assert database_path.exists()

    connection = get_connection(database_path)

    try:
        result = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name = 'documents'
            """
        ).fetchone()

        assert result is not None

    finally:
        connection.close()


def test_documents_table_has_expected_columns(tmp_path):
    database_path = tmp_path / "test_claimwise.db"

    initialize_database(database_path)

    connection = get_connection(database_path)

    try:
        columns = connection.execute(
            "PRAGMA table_info(documents)"
        ).fetchall()

        column_names = {
            column["name"]
            for column in columns
        }

        expected_columns = {
            "document_id",
            "content_hash",
            "file_name",
            "file_size_bytes",
            "file_extension",
            "uploaded_at",
            "source_type",
            "title",
            "author",
            "total_pages",
            "policy_group_id",
            "version_label",
        }

        assert expected_columns.issubset(column_names)

    finally:
        connection.close()


def test_database_initialization_is_idempotent(tmp_path):
    database_path = tmp_path / "test_claimwise.db"

    initialize_database(database_path)
    initialize_database(database_path)

    connection = get_connection(database_path)

    try:
        result = connection.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
            AND name = 'documents'
            """
        ).fetchone()

        assert result[0] == 1

    finally:
        connection.close()


def test_invalid_source_type_is_rejected(tmp_path):
    database_path = tmp_path / "test_claimwise.db"

    initialize_database(database_path)

    connection = get_connection(database_path)

    try:
        with pytest.raises(sqlite3.IntegrityError):
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
                    total_pages
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "test-id",
                    "a" * 64,
                    "policy.pdf",
                    1000,
                    ".pdf",
                    "2026-11-15T12:00:00+00:00",
                    "invalid_source",
                    10,
                ),
            )

    finally:
        connection.close()
