
import sqlite3
from pathlib import Path

from claimwise.config import settings


DEFAULT_DATABASE_PATH = (
    settings.absolute_data_dir / "claimwise.db"
)


def get_connection(
    database_path: str | Path | None = None,
) -> sqlite3.Connection:
    """Create and return a SQLite database connection."""

    path = (
        Path(database_path)
        if database_path is not None
        else DEFAULT_DATABASE_PATH
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        str(path),
        timeout=10,
    )

    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_database(
    database_path: str | Path | None = None,
) -> None:
    """Create database tables and apply basic schema migrations."""

    connection = get_connection(database_path)

    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (

                document_id TEXT PRIMARY KEY,

                content_hash TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL
                    CHECK (file_size_bytes >= 0),

                file_extension TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,

                source_type TEXT NOT NULL
                    CHECK (
                        source_type IN ('public', 'user_upload')
                    ),

                title TEXT,
                author TEXT,
                total_pages INTEGER NOT NULL
                    CHECK (total_pages >= 0),

                policy_group_id TEXT,
                version_label TEXT,

                storage_path TEXT NOT NULL DEFAULT '',

                ingestion_status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (
                        ingestion_status IN (
                            'pending',
                            'processing',
                            'completed',
                            'failed'
                        )
                    ),

                error_message TEXT,

                chunk_count INTEGER NOT NULL DEFAULT 0
                    CHECK (chunk_count >= 0),

                UNIQUE (content_hash, source_type)
            );

            CREATE INDEX IF NOT EXISTS
                idx_documents_policy_group
            ON documents(policy_group_id);

            CREATE INDEX IF NOT EXISTS
                idx_documents_uploaded_at
            ON documents(uploaded_at);
            """
        )

        # Add columns to databases created using the earlier schema.
        existing_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(documents)"
            ).fetchall()
        }

        migrations = {
            "storage_path": (
                "ALTER TABLE documents "
                "ADD COLUMN storage_path TEXT NOT NULL DEFAULT ''"
            ),
            "ingestion_status": (
                "ALTER TABLE documents "
                "ADD COLUMN ingestion_status TEXT NOT NULL "
                "DEFAULT 'pending'"
            ),
            "error_message": (
                "ALTER TABLE documents "
                "ADD COLUMN error_message TEXT"
            ),
            "chunk_count": (
                "ALTER TABLE documents "
                "ADD COLUMN chunk_count INTEGER NOT NULL DEFAULT 0"
            ),
        }

        for column_name, migration_sql in migrations.items():
            if column_name not in existing_columns:
                connection.execute(migration_sql)

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_documents_ingestion_status
            ON documents(ingestion_status)
            """
        )
        
        connection.commit()

    finally:
        connection.close()
