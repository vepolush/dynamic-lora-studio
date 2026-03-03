"""Caption generation and ingestion for entity datasets."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
CAPTION_MODES = {"none", "auto", "manual_zip"}


class CaptioningError(RuntimeError):
    """Raised when caption processing fails."""


class BlipBaseCaptioner:
    """Lazy BLIP-base captioner for automatic image descriptions."""

    def __init__(self) -> None:
        self.model_id = BLIP_MODEL_ID
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor: BlipProcessor | None = None
        self._model: BlipForConditionalGeneration | None = None

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        _ensure_torch_version_for_secure_load()
        self._processor = BlipProcessor.from_pretrained(self.model_id)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()

    def caption(self, image_path: Path) -> str:
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=40)
        text = self._processor.decode(output_ids[0], skip_special_tokens=True).strip().lower()
        return text


_BLIP_CAPTIONER: BlipBaseCaptioner | None = None


def _get_blip_captioner() -> BlipBaseCaptioner:
    global _BLIP_CAPTIONER
    if _BLIP_CAPTIONER is None:
        _BLIP_CAPTIONER = BlipBaseCaptioner()
    return _BLIP_CAPTIONER


def _validate_member_path(member_name: str) -> None:
    normalized = member_name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute():
        raise CaptioningError(f"ZIP contains absolute path entry: {member_name}")
    if any(part == ".." for part in path.parts):
        raise CaptioningError(f"ZIP contains unsafe path traversal entry: {member_name}")


def _caption_file_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def _normalize_caption(raw: str) -> str:
    return " ".join(raw.replace("\n", " ").split()).strip()


def _ensure_trigger(caption: str, trigger_word: str) -> str:
    cleaned = _normalize_caption(caption)
    if not cleaned:
        return f"photo of {trigger_word}"
    if trigger_word.lower() in cleaned.lower():
        return cleaned
    return f"photo of {trigger_word}, {cleaned}"


def _load_manifest_images(entity_dir: Path) -> list[dict[str, Any]]:
    manifest_path = entity_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise CaptioningError("dataset_manifest.json is missing")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    images = manifest.get("images", [])
    if not isinstance(images, list):
        raise CaptioningError("dataset_manifest.json has invalid images list")
    return [item for item in images if isinstance(item, dict)]


def _load_provided_captions(zip_path: Path) -> dict[str, str]:
    captions: dict[str, str] = {}
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                _validate_member_path(member.filename)
                if not member.filename.lower().endswith(".txt"):
                    continue
                with archive.open(member, "r") as f:
                    raw_text = f.read().decode("utf-8", errors="ignore")
                caption = _normalize_caption(raw_text)
                if not caption:
                    continue
                normalized_name = member.filename.replace("\\", "/").lower()
                captions[normalized_name] = caption
                captions[PurePosixPath(normalized_name).name] = caption
    except zipfile.BadZipFile as exc:
        raise CaptioningError("Uploaded file is not a valid ZIP archive") from exc
    return captions


def apply_caption_mode(
    *,
    entity_dir: Path,
    zip_path: Path,
    trigger_word: str,
    caption_mode: str,
) -> dict[str, Any]:
    """Create or consume captions next to preprocessed dataset images."""
    mode = caption_mode.strip().lower()
    if mode not in CAPTION_MODES:
        raise CaptioningError(f"Unsupported caption mode: {caption_mode}")

    dataset_dir = entity_dir / "dataset"
    images_manifest = _load_manifest_images(entity_dir)
    image_paths = sorted([p for p in dataset_dir.glob("*.png") if p.is_file()])
    if not image_paths:
        raise CaptioningError("Dataset has no images to caption")

    if mode == "none":
        return {
            "caption_mode": "none",
            "caption_files": 0,
            "auto_generated": 0,
            "provided_used": 0,
            "fallback_default": len(image_paths),
            "model": None,
        }

    auto_generated = 0
    provided_used = 0
    fallback_default = 0

    provided_captions: dict[str, str] = {}
    if mode == "manual_zip":
        provided_captions = _load_provided_captions(zip_path)

    captioner = _get_blip_captioner() if mode == "auto" else None
    for image_info in images_manifest:
        output_name = str(image_info.get("filename", ""))
        if not output_name:
            continue
        image_path = dataset_dir / output_name
        if not image_path.exists():
            continue
        caption_raw = ""
        if mode == "auto":
            assert captioner is not None
            caption_raw = captioner.caption(image_path)
            auto_generated += 1
        else:
            source = str(image_info.get("source", "")).replace("\\", "/").lower()
            source_txt = str(PurePosixPath(source).with_suffix(".txt")) if source else ""
            fallback_name = f"{Path(source).stem}.txt".lower() if source else ""
            local_name = f"{Path(output_name).stem}.txt".lower()
            caption_raw = (
                provided_captions.get(source_txt, "")
                or provided_captions.get(PurePosixPath(source_txt).name, "")
                or provided_captions.get(fallback_name, "")
                or provided_captions.get(local_name, "")
            )
            if caption_raw:
                provided_used += 1
            else:
                fallback_default += 1

        final_caption = _ensure_trigger(caption_raw, trigger_word)
        with open(_caption_file_path(image_path), "w", encoding="utf-8") as f:
            f.write(final_caption)

    return {
        "caption_mode": mode,
        "caption_files": len(image_paths),
        "auto_generated": auto_generated,
        "provided_used": provided_used,
        "fallback_default": fallback_default,
        "model": BLIP_MODEL_ID if mode == "auto" else None,
    }


def _parse_major_minor(version: str) -> tuple[int, int]:
    cleaned = version.split("+", 1)[0]
    parts = cleaned.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


def _ensure_torch_version_for_secure_load() -> None:
    major, minor = _parse_major_minor(torch.__version__)
    if (major, minor) < (2, 6):
        raise CaptioningError(
            "Auto caption mode requires torch>=2.6 due to secure model loading policy. "
            f"Current torch version: {torch.__version__}. "
            "Upgrade backend dependencies or use caption mode 'none'/'manual_zip'."
        )
