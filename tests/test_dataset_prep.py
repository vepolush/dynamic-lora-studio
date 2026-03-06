"""Tests for dataset_prep module."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

# Add backend to path before importing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dataset_prep import (
    DatasetValidationError,
    _validate_member_path,
    prepare_entity_dataset,
    add_images_from_zip,
    remove_dataset_images,
)


def _make_valid_png_bytes() -> bytes:
    """Create valid 2x2 PNG via PIL."""
    from PIL import Image
    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_zip_with_images(*filenames: str, content: bytes | None = None) -> bytes:
    """Create ZIP bytes with given filenames. content = valid PNG if None."""
    img_content = content or _make_valid_png_bytes()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in filenames:
            ext = Path(name).suffix.lower()
            data = img_content if ext in {".png", ".jpg", ".jpeg", ".webp"} else b"plain text"
            zf.writestr(name, data)
    return buf.getvalue()


class TestValidateMemberPath:
    """Path traversal and absolute path validation."""

    def test_safe_relative_path(self) -> None:
        _validate_member_path("images/photo.png")
        _validate_member_path("a/b/c.png")

    def test_absolute_path_rejected(self) -> None:
        with pytest.raises(DatasetValidationError, match="absolute"):
            _validate_member_path("/etc/passwd")

    def test_path_traversal_rejected(self) -> None:
        with pytest.raises(DatasetValidationError, match="traversal"):
            _validate_member_path("../../../etc/passwd")
        with pytest.raises(DatasetValidationError, match="traversal"):
            _validate_member_path("images/../../secret.png")


class TestPrepareEntityDataset:
    """ZIP validation and dataset preparation."""

    def test_missing_zip(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetValidationError, match="missing"):
            prepare_entity_dataset(
                entity_dir=tmp_path / "entity",
                zip_path=tmp_path / "nonexistent.zip",
                entity_id="test_entity",
            )

    def test_empty_zip(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            pass
        with pytest.raises(DatasetValidationError, match="not contain files"):
            prepare_entity_dataset(
                entity_dir=tmp_path / "entity",
                zip_path=zip_path,
                entity_id="test_entity",
            )

    def test_no_valid_images(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "no_images.zip"
        zip_path.write_bytes(_make_zip_with_images("readme.txt"))
        with pytest.raises(DatasetValidationError, match="No valid images"):
            prepare_entity_dataset(
                entity_dir=tmp_path / "entity",
                zip_path=zip_path,
                entity_id="test_entity",
            )

    def test_valid_single_image(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "images.zip"
        zip_path.write_bytes(_make_zip_with_images("photo.png"))
        entity_dir = tmp_path / "entity"
        manifest = prepare_entity_dataset(
            entity_dir=entity_dir,
            zip_path=zip_path,
            entity_id="test_entity",
        )
        assert manifest["image_count"] == 1
        assert (entity_dir / "dataset").exists()
        assert len(list((entity_dir / "dataset").glob("*.png"))) == 1
