
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from claimwise.ingestion.pdf_loader import PDFExtractionResult


@dataclass
class DocumentMetadata:
    """Stores identification and metadata for an uploaded document."""

    document_id: str
    content_hash: str
    file_name: str
    file_size_bytes: int
    file_extension: str
    uploaded_at: str
    source_type: str

    title: str | None
    author: str | None
    total_pages: int

    # Documents belonging to the same policy can share this ID.
    policy_group_id: str | None = None

    # Example: "2024", "2025", "Version 2"
    version_label: str | None = None


def calculate_file_hash(file_path: str | Path) -> str:
    """
    Calculate the SHA-256 hash of a file.

    Reads the file in small chunks to avoid loading
    the entire document into memory.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    hasher = sha256()

    with file_path.open("rb") as file:

        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def create_document_metadata(
    extraction_result: PDFExtractionResult,
    source_type: str = "user_upload",
    policy_group_id: str | None = None,
    version_label: str | None = None,
) -> DocumentMetadata:
    """
    Create metadata for an extracted PDF document.
    """

    # 1. Validate the source type.
    allowed_sources = {"public", "user_upload"}

    if source_type not in allowed_sources:
        raise ValueError(
            f"Invalid source_type: {source_type}. "
            f"Allowed values: {allowed_sources}"
        )

    # 2. Get the original file path.
    file_path = Path(extraction_result.file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    # 3. Generate a unique document ID.
    document_id = str(uuid4())

    # 4. Calculate the file's content hash.
    content_hash = calculate_file_hash(file_path)

    # 5. Read file information.
    file_size_bytes = file_path.stat().st_size
    file_name = file_path.name
    file_extension = file_path.suffix.lower()

    # 6. Record the upload timestamp in UTC.
    uploaded_at = datetime.now(
        timezone.utc
    ).isoformat()

    # 7. Build and return the metadata object.
    return DocumentMetadata(
        document_id=document_id,
        content_hash=content_hash,
        file_name=file_name,
        file_size_bytes=file_size_bytes,
        file_extension=file_extension,
        uploaded_at=uploaded_at,
        source_type=source_type,
        title=extraction_result.title,
        author=extraction_result.author,
        total_pages=extraction_result.total_pages,
        policy_group_id=policy_group_id,
        version_label=version_label,
    )
