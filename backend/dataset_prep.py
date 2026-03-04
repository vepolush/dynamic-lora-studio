"""Safe ZIP ingest and dataset preprocessing for entity training."""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_ZIP_SIZE_BYTES = 100 * 1024 * 1024
MAX_FILES_IN_ZIP = 50
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 300 * 1024 * 1024
TARGET_IMAGE_SIZE = (512, 512)
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
LANCZOS_RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


class DatasetValidationError(ValueError):
    """Raised when user-provided ZIP or images are invalid."""


def _safe_stem(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    normalized = cleaned.strip("_")
    return normalized[:48] if normalized else fallback


def _validate_member_path(member_name: str) -> None:
    normalized = member_name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute():
        raise DatasetValidationError(f"ZIP contains absolute path entry: {member_name}")
    if any(part == ".." for part in path.parts):
        raise DatasetValidationError(f"ZIP contains unsafe path traversal entry: {member_name}")


def _prepare_dataset_dir(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for child in dataset_dir.iterdir():
        if child.is_file():
            child.unlink()


def prepare_entity_dataset(*, entity_dir: Path, zip_path: Path, entity_id: str) -> dict[str, Any]:
    """Validate ZIP and save normalized 512x512 RGB images + manifest."""
    if not zip_path.exists():
        raise DatasetValidationError("Uploaded ZIP is missing on server")
    if zip_path.stat().st_size > MAX_ZIP_SIZE_BYTES:
        raise DatasetValidationError("ZIP file exceeds 100 MB limit")

    dataset_dir = entity_dir / "dataset"
    _prepare_dataset_dir(dataset_dir)

    processed_images: list[dict[str, Any]] = []
    total_uncompressed = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            if not members:
                raise DatasetValidationError("ZIP archive does not contain files")
            if len(members) > MAX_FILES_IN_ZIP:
                raise DatasetValidationError(f"ZIP contains too many files (max {MAX_FILES_IN_ZIP})")

            image_index = 0
            for member in members:
                _validate_member_path(member.filename)
                if member.file_size <= 0:
                    continue
                if member.file_size > MAX_FILE_SIZE_BYTES:
                    raise DatasetValidationError(
                        f"File '{member.filename}' exceeds max size of 20 MB"
                    )

                total_uncompressed += member.file_size
                if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
                    raise DatasetValidationError("Uncompressed dataset exceeds size limit")

                with archive.open(member, "r") as file_obj:
                    raw = file_obj.read(MAX_FILE_SIZE_BYTES + 1)
                if len(raw) > MAX_FILE_SIZE_BYTES:
                    raise DatasetValidationError(
                        f"File '{member.filename}' exceeds max size of 20 MB"
                    )

                try:
                    image = Image.open(io.BytesIO(raw))
                except UnidentifiedImageError:
                    # Allow auxiliary files (e.g., .txt captions) in ZIP archive.
                    continue

                image_format = (image.format or "").upper()
                if image_format not in ALLOWED_IMAGE_FORMATS:
                    continue

                image = ImageOps.exif_transpose(image).convert("RGB")
                prepared = ImageOps.fit(
                    image,
                    TARGET_IMAGE_SIZE,
                    method=LANCZOS_RESAMPLE,
                    centering=(0.5, 0.5),
                )
                stem = _safe_stem(Path(member.filename).stem, fallback=f"image_{image_index + 1}")
                output_name = f"{image_index + 1:04d}_{stem}.png"
                output_path = dataset_dir / output_name
                prepared.save(output_path, format="PNG", optimize=True)

                processed_images.append(
                    {
                        "filename": output_name,
                        "source": member.filename,
                        "width": TARGET_IMAGE_SIZE[0],
                        "height": TARGET_IMAGE_SIZE[1],
                        "color_mode": "RGB",
                    }
                )
                image_index += 1

    except zipfile.BadZipFile as exc:
        raise DatasetValidationError("Uploaded file is not a valid ZIP archive") from exc

    if not processed_images:
        raise DatasetValidationError("No valid images found in ZIP archive")

    manifest = {
        "entity_id": entity_id,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "source_zip": zip_path.name,
        "target_size": {"width": TARGET_IMAGE_SIZE[0], "height": TARGET_IMAGE_SIZE[1]},
        "image_count": len(processed_images),
        "images": processed_images,
    }
    manifest_path = entity_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def _load_manifest(entity_dir: Path) -> dict[str, Any]:
    manifest_path = entity_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        return {"images": [], "image_count": 0}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(entity_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = entity_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def add_images_from_zip(
    *,
    entity_dir: Path,
    zip_path: Path,
    entity_id: str,
) -> dict[str, Any]:
    """Add images from ZIP to existing dataset. Updates manifest."""
    if not zip_path.exists():
        raise DatasetValidationError("Uploaded ZIP is missing on server")
    if zip_path.stat().st_size > MAX_ZIP_SIZE_BYTES:
        raise DatasetValidationError("ZIP file exceeds 100 MB limit")

    dataset_dir = entity_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(entity_dir)
    existing_images = manifest.get("images", [])
    if not isinstance(existing_images, list):
        existing_images = []

    max_index = 0
    for img in existing_images:
        fname = str(img.get("filename", ""))
        match = re.match(r"(\d+)", fname)
        if match:
            max_index = max(max_index, int(match.group(1)))

    processed: list[dict[str, Any]] = []
    total_uncompressed = 0
    for img in existing_images:
        p = dataset_dir / str(img.get("filename", ""))
        if p.exists():
            total_uncompressed += p.stat().st_size

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            if not members:
                raise DatasetValidationError("ZIP archive does not contain files")
            current_count = len(existing_images) + len(processed)
            if current_count + len(members) > MAX_FILES_IN_ZIP:
                raise DatasetValidationError(
                    f"Adding these images would exceed max {MAX_FILES_IN_ZIP} files"
                )

            image_index = max_index
            for member in members:
                _validate_member_path(member.filename)
                if member.file_size <= 0:
                    continue
                if member.file_size > MAX_FILE_SIZE_BYTES:
                    raise DatasetValidationError(
                        f"File '{member.filename}' exceeds max size of 20 MB"
                    )
                total_uncompressed += member.file_size
                if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
                    raise DatasetValidationError("Uncompressed dataset exceeds size limit")

                with archive.open(member, "r") as file_obj:
                    raw = file_obj.read(MAX_FILE_SIZE_BYTES + 1)
                if len(raw) > MAX_FILE_SIZE_BYTES:
                    raise DatasetValidationError(
                        f"File '{member.filename}' exceeds max size of 20 MB"
                    )

                try:
                    image = Image.open(io.BytesIO(raw))
                except UnidentifiedImageError:
                    continue

                image_format = (image.format or "").upper()
                if image_format not in ALLOWED_IMAGE_FORMATS:
                    continue

                image = ImageOps.exif_transpose(image).convert("RGB")
                prepared = ImageOps.fit(
                    image,
                    TARGET_IMAGE_SIZE,
                    method=LANCZOS_RESAMPLE,
                    centering=(0.5, 0.5),
                )
                image_index += 1
                stem = _safe_stem(Path(member.filename).stem, fallback=f"image_{image_index}")
                output_name = f"{image_index:04d}_{stem}.png"
                output_path = dataset_dir / output_name
                prepared.save(output_path, format="PNG", optimize=True)

                processed.append({
                    "filename": output_name,
                    "source": member.filename,
                    "width": TARGET_IMAGE_SIZE[0],
                    "height": TARGET_IMAGE_SIZE[1],
                    "color_mode": "RGB",
                })

    except zipfile.BadZipFile as exc:
        raise DatasetValidationError("Uploaded file is not a valid ZIP archive") from exc

    if not processed:
        raise DatasetValidationError("No valid images found in ZIP archive")

    manifest["images"] = existing_images + processed
    manifest["image_count"] = len(manifest["images"])
    _save_manifest(entity_dir, manifest)
    return {"added": len(processed), "total": manifest["image_count"]}


def remove_dataset_images(
    *,
    entity_dir: Path,
    filenames: list[str],
    entity_id: str,
) -> dict[str, Any]:
    """Remove images from dataset. Updates manifest."""
    dataset_dir = entity_dir / "dataset"
    manifest = _load_manifest(entity_dir)
    images = manifest.get("images", [])
    if not isinstance(images, list):
        images = []

    to_remove = set(f.strip() for f in filenames if f and f.strip())
    if not to_remove:
        return {"removed": 0, "total": len(images)}

    remaining = [img for img in images if str(img.get("filename", "")) not in to_remove]
    removed_count = len(images) - len(remaining)

    for fname in to_remove:
        img_path = dataset_dir / fname
        if img_path.exists():
            img_path.unlink()
        caption_path = img_path.with_suffix(".txt")
        if caption_path.exists():
            caption_path.unlink()

    manifest["images"] = remaining
    manifest["image_count"] = len(remaining)
    _save_manifest(entity_dir, manifest)

    if not remaining:
        raise DatasetValidationError("Cannot remove all images; dataset would be empty")

    return {"removed": removed_count, "total": len(remaining)}
