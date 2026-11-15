
import shutil

import pymupdf
import pytest

from claimwise.ingestion.metadata import (
    calculate_file_hash,
    create_document_metadata,
)
from claimwise.ingestion.pdf_loader import extract_pdf


def create_test_pdf(pdf_path):
    """Create a small digital PDF for testing."""

    document = pymupdf.open()
    page = document.new_page()

    page.insert_text(
        (50, 50),
        "ClaimWise Health Insurance Policy Test",
    )

    document.save(str(pdf_path))
    document.close()


def test_metadata_creation(tmp_path):
    """Metadata should contain correct document information."""

    pdf_path = tmp_path / "policy.pdf"
    create_test_pdf(pdf_path)

    extraction_result = extract_pdf(pdf_path)

    metadata = create_document_metadata(
        extraction_result
    )

    assert metadata.document_id
    assert len(metadata.content_hash) == 64
    assert metadata.file_name == "policy.pdf"
    assert metadata.file_extension == ".pdf"
    assert metadata.file_size_bytes > 0
    assert metadata.total_pages == 1
    assert metadata.source_type == "user_upload"
    assert metadata.uploaded_at


def test_identical_files_have_same_hash(tmp_path):
    """Identical files should produce the same SHA-256 hash."""

    first_file = tmp_path / "policy.pdf"
    second_file = tmp_path / "policy_copy.pdf"

    create_test_pdf(first_file)

    shutil.copy2(
        first_file,
        second_file,
    )

    first_hash = calculate_file_hash(first_file)
    second_hash = calculate_file_hash(second_file)

    assert first_hash == second_hash


def test_identical_uploads_get_different_document_ids(tmp_path):
    """Each metadata record should have its own document ID."""

    pdf_path = tmp_path / "policy.pdf"
    create_test_pdf(pdf_path)

    extraction_result = extract_pdf(pdf_path)

    first_metadata = create_document_metadata(
        extraction_result
    )

    second_metadata = create_document_metadata(
        extraction_result
    )

    assert first_metadata.document_id != second_metadata.document_id
    assert first_metadata.content_hash == second_metadata.content_hash


def test_policy_version_information(tmp_path):
    """Policy group and version information should be preserved."""

    pdf_path = tmp_path / "policy.pdf"
    create_test_pdf(pdf_path)

    extraction_result = extract_pdf(pdf_path)

    metadata = create_document_metadata(
        extraction_result,
        policy_group_id="star-health-policy-001",
        version_label="2025",
    )

    assert metadata.policy_group_id == "star-health-policy-001"
    assert metadata.version_label == "2025"


def test_invalid_source_type_raises_error(tmp_path):
    """Unknown document source types should be rejected."""

    pdf_path = tmp_path / "policy.pdf"
    create_test_pdf(pdf_path)

    extraction_result = extract_pdf(pdf_path)

    with pytest.raises(ValueError):
        create_document_metadata(
            extraction_result,
            source_type="unknown",
        )


def test_missing_file_hash_raises_error(tmp_path):
    """Hashing a missing file should raise FileNotFoundError."""

    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        calculate_file_hash(missing_file)
