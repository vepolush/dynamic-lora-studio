"""Dynamic LoRA Studio — FastAPI backend."""

from __future__ import annotations

import time
import traceback
from contextlib import asynccontextmanager

from loguru import logger

from logger import setup_logging
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

import filetype

from auth import get_current_user, login as auth_login, register as auth_register, require_auth

from captioning import CAPTION_MODES, CaptioningError, apply_caption_mode
from dataset_prep import (
    DatasetValidationError,
    add_images_from_zip,
    prepare_entity_dataset,
    remove_dataset_images,
)
from entity_store import (
    DATA_DIR,
    create_entity,
    delete_entity as store_delete_entity,
    entity_dataset_dir,
    get_entity as store_get_entity,
    get_entity_metadata,
    list_dataset_images,
    list_entities as store_list_entities,
    load_dataset_image_bytes,
    load_entity_preview_bytes,
    update_entity_metadata,
)
from model_manager import ml_manager
from prompt_builder import (
    build_enhanced_prompt,
    build_negative_prompt,
    get_quality_scheduler,
    get_quality_steps,
)
from session_store import (
    add_message,
    create_session as store_create_session,
    delete_session as store_delete_session,
    get_session as store_get_session,
    list_sessions as store_list_sessions,
    load_image_bytes,
    new_message_id,
    save_image,
    update_session as store_update_session,
)
from training_queue import training_queue
from gallery_store import (
    get_gallery_image,
    get_published_filenames_for_session,
    list_gallery,
    load_gallery_image_bytes,
    publish_image,
    toggle_like,
)
from gallery_lora_store import (
    add_gallery_lora,
    get_gallery_lora,
    list_gallery_loras,
    load_gallery_lora_preview_bytes,
    publish_lora,
    unpublish_lora,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    from db import init_db
    init_db()
    logger.info("Database initialized")
    training_queue.start()
    logger.info("Training queue started")
    try:
        ml_manager.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error("Model loading failed: {}", e)
    yield
    training_queue.stop()
    logger.info("Server shutting down")


app = FastAPI(lifespan=lifespan)

TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    "fast": {
        "steps": 700,
        "rank": 8,
        "learning_rate": 1e-4,
        "lr_scheduler": "constant",
        "warmup_ratio": 0.03,
    },
    "balanced": {
        "steps": 1200,
        "rank": 16,
        "learning_rate": 1e-4,
        "lr_scheduler": "polynomial",
        "warmup_ratio": 0.06,
    },
    "strong": {
        "steps": 1800,
        "rank": 16,
        "learning_rate": 8e-5,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.08,
    },
}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    title: str = "New session"


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    favourite: bool | None = None
    favourite_image_filenames: list[str] | None = None
    archived: bool | None = None


class GenerateRequest(BaseModel):
    session_id: str
    prompt: str
    negative_prompt: str = ""
    steps: int | None = None
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: int | None = None
    num_images: int = 1
    scheduler: str | None = None
    quality: str = "Normal"
    style: str | None = None
    lighting: str | None = None
    color: str | None = None
    entity_id: str | None = None
    entity_version: str | None = None
    lora_strength: float = 0.8


class UpdateEntityRequest(BaseModel):
    name: str | None = None
    trigger_word: str | None = None


class RemoveDatasetRequest(BaseModel):
    filenames: list[str] = []


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class PublishRequest(BaseModel):
    session_id: str
    filename: str
    prompt: str
    settings: dict | None = None


class PublishLoraRequest(BaseModel):
    entity_id: str
    name: str
    trigger_word: str
    description: str | None = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    status = "ready" if ml_manager.pipe is not None else "loading"
    return {"status": status}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.post("/api/auth/register")
def register(req: RegisterRequest):
    return auth_register(req.username, req.password)


@app.post("/api/auth/login")
def login(req: LoginRequest):
    return auth_login(req.username, req.password)


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------

@app.get("/api/gallery")
def get_gallery(
    sort: str = "newest",
    limit: int = 50,
    offset: int = 0,
    user=Depends(get_current_user),
):
    return {"images": list_gallery(sort=sort, limit=limit, offset=offset, current_user_id=user["user_id"] if user else None)}


@app.post("/api/gallery/loras/publish")
def publish_lora_to_gallery(req: PublishLoraRequest, user=Depends(require_auth)):
    try:
        return publish_lora(
            user_id=user["user_id"],
            entity_id=req.entity_id,
            name=req.name,
            trigger_word=req.trigger_word,
            description=req.description,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/gallery/loras")
def get_gallery_loras(
    sort: str = "newest",
    limit: int = 50,
    offset: int = 0,
    user=Depends(get_current_user),
):
    return {
        "loras": list_gallery_loras(
            sort=sort,
            limit=limit,
            offset=offset,
            current_user_id=user["user_id"] if user else None,
        )
    }


@app.get("/api/gallery/loras/{lora_id}")
def get_gallery_lora_item(lora_id: str, user=Depends(get_current_user)):
    item = get_gallery_lora(lora_id, current_user_id=user["user_id"] if user else None)
    if not item:
        raise HTTPException(status_code=404, detail="LoRA not found")
    return item


@app.get("/api/gallery/loras/{lora_id}/preview")
def get_gallery_lora_preview(lora_id: str):
    data = load_gallery_lora_preview_bytes(lora_id)
    if not data:
        raise HTTPException(status_code=404, detail="LoRA preview not found")
    return Response(content=data, media_type="image/png")


@app.post("/api/gallery/loras/{lora_id}/add")
def add_gallery_lora_to_entities(lora_id: str, user=Depends(require_auth)):
    try:
        return add_gallery_lora(lora_id, user["user_id"])
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="LoRA not found")


@app.delete("/api/gallery/loras/{lora_id}")
def unpublish_gallery_lora(lora_id: str, user=Depends(require_auth)):
    if not unpublish_lora(lora_id, user["user_id"]):
        raise HTTPException(status_code=404, detail="LoRA not found or you are not the creator")
    return {"status": "unpublished"}


@app.get("/api/gallery/{image_id}")
def get_gallery_item(image_id: str, user=Depends(get_current_user)):
    item = get_gallery_image(image_id, current_user_id=user["user_id"] if user else None)
    if not item:
        raise HTTPException(status_code=404, detail="Image not found")
    return item


@app.get("/api/gallery/{image_id}/image")
def get_gallery_image_bytes(image_id: str):
    data = load_gallery_image_bytes(image_id)
    if not data:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=data, media_type="image/png")


@app.get("/api/gallery/published-filenames")
def get_published_filenames(session_id: str, user=Depends(get_current_user)):
    """Return filenames from session that are already published to gallery."""
    return {"filenames": list(get_published_filenames_for_session(session_id))}


@app.post("/api/gallery/publish")
def publish_to_gallery(req: PublishRequest, user=Depends(require_auth)):
    try:
        return publish_image(
            user_id=user["user_id"],
            session_id=req.session_id,
            filename=req.filename,
            prompt=req.prompt,
            settings=req.settings,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/gallery/{image_id}/like")
def like_gallery_image(image_id: str, user=Depends(require_auth)):
    result = toggle_like(image_id, user["user_id"])
    if not result:
        raise HTTPException(status_code=404, detail="Image not found")
    return result


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
def get_sessions(user=Depends(get_current_user)):
    return {"sessions": store_list_sessions(user_id=user["user_id"] if user else None)}


@app.post("/api/sessions")
def create_session(req: CreateSessionRequest, user=Depends(require_auth)):
    return store_create_session(title=req.title, user_id=user["user_id"])


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, user=Depends(get_current_user)):
    session = store_get_session(session_id, user_id=user["user_id"] if user else None)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.put("/api/sessions/{session_id}")
def update_session(session_id: str, req: UpdateSessionRequest, user=Depends(get_current_user)):
    updates: dict[str, Any] = {}
    if req.title is not None:
        updates["title"] = req.title
    if req.favourite is not None:
        updates["favourite"] = req.favourite
    if req.favourite_image_filenames is not None:
        updates["favourite_image_filenames"] = req.favourite_image_filenames
    if req.archived is not None:
        updates["archived"] = req.archived
    result = store_update_session(session_id, updates, user_id=user["user_id"] if user else None)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, user=Depends(get_current_user)):
    if not store_delete_session(session_id, user_id=user["user_id"] if user else None):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

@app.get("/api/images/{session_id}/{filename}")
def get_image(session_id: str, filename: str):
    data = load_image_bytes(session_id, filename)
    if data is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=data, media_type="image/png")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@app.post("/api/generate")
def generate_image(req: GenerateRequest, user=Depends(get_current_user)):
    if ml_manager.pipe is None:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to load")

    session = store_get_session(req.session_id, user_id=user["user_id"] if user else None)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        enhanced_prompt = build_enhanced_prompt(
            req.prompt,
            style=req.style,
            lighting=req.lighting,
            color=req.color,
            quality=req.quality,
        )
        negative = build_negative_prompt(
            req.negative_prompt,
            style=req.style,
            quality=req.quality,
        )
        steps = get_quality_steps(req.quality, req.steps)
        scheduler = get_quality_scheduler(req.quality, req.scheduler)

        logger.info(
            "Generating: prompt_len={} steps={} sched={} entity={}",
            len(enhanced_prompt), steps, scheduler, req.entity_id,
        )
        start_time = time.time()

        results = ml_manager.generate(
            prompt=enhanced_prompt,
            negative_prompt=negative,
            steps=steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            seed=req.seed,
            num_images=min(req.num_images, 4),
            scheduler=scheduler,
            entity_id=req.entity_id,
            entity_version=req.entity_version,
            lora_strength=req.lora_strength,
        )

        gen_time = time.time() - start_time
        logger.info("Generation done in {:.2f}s — {} image(s)", gen_time, len(results))

        images_response: list[dict[str, Any]] = []
        image_filenames: list[dict[str, Any]] = []
        for i, item in enumerate(results):
            fname = f"{new_message_id()}_{i}.png"
            save_image(req.session_id, item["png_bytes"], fname)
            images_response.append({
                "base64": item["base64"],
                "seed": item["seed"],
                "filename": fname,
            })
            image_filenames.append({
                "filename": fname,
                "seed": item["seed"],
            })

        message = {
            "id": new_message_id(),
            "prompt": req.prompt,
            "enhanced_prompt": enhanced_prompt,
            "negative_prompt": negative,
            "settings": {
                "steps": steps,
                "guidance_scale": req.guidance_scale,
                "width": req.width,
                "height": req.height,
                "seed": req.seed,
                "num_images": req.num_images,
                "scheduler": scheduler,
                "quality": req.quality,
                "style": req.style,
                "lighting": req.lighting,
                "color": req.color,
            },
            "images": image_filenames,
            "generation_time": round(gen_time, 2),
        }
        add_message(req.session_id, message)

        return {
            "status": "success",
            "images": images_response,
            "enhanced_prompt": enhanced_prompt,
            "negative_prompt": negative,
            "generation_time": round(gen_time, 2),
            "message": message,
        }

    except Exception as e:
        logger.exception("Generation error: {}", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entities (placeholder)
# ---------------------------------------------------------------------------

@app.get("/api/entities")
def get_entities(user=Depends(get_current_user)):
    return {"entities": store_list_entities(user_id=user["user_id"] if user else None)}


@app.get("/api/entities/{entity_id}")
def get_entity(entity_id: str, user=Depends(get_current_user)):
    entity = store_get_entity(entity_id, user_id=user["user_id"] if user else None)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return entity


@app.get("/api/entities/{entity_id}/preview")
def get_entity_preview(entity_id: str, user=Depends(get_current_user)):
    entity = store_get_entity(entity_id, user_id=user["user_id"] if user else None)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    data = load_entity_preview_bytes(entity_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Entity preview not found")
    return Response(content=data, media_type="image/png")


@app.get("/api/entities/{entity_id}/dataset")
def get_entity_dataset(entity_id: str, user=Depends(get_current_user)):
    entity = store_get_entity(entity_id, user_id=user["user_id"] if user else None)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    images = list_dataset_images(entity_id)
    return {"images": images}


@app.get("/api/entities/{entity_id}/dataset/{filename:path}")
def get_dataset_image(entity_id: str, filename: str, user=Depends(get_current_user)):
    entity = store_get_entity(entity_id, user_id=user["user_id"] if user else None)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    data = load_dataset_image_bytes(entity_id, filename)
    if data is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=data, media_type="image/png")


@app.delete("/api/entities/{entity_id}/dataset")
def remove_entity_dataset_images(entity_id: str, req: RemoveDatasetRequest, user=Depends(get_current_user)):
    entity = store_get_entity(entity_id, user_id=user["user_id"] if user else None)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    entity_dir = Path(DATA_DIR) / "storage" / "entities" / entity_id
    try:
        result = remove_dataset_images(
            entity_dir=entity_dir,
            filenames=req.filenames,
            entity_id=entity_id,
        )
    except DatasetValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    update_entity_metadata(entity_id, {"image_count": result["total"]}, user_id=user["user_id"] if user else None)
    return result


@app.post("/api/entities")
def upload_entity(
    name: str = Form(...),
    trigger_word: str = Form(...),
    user=Depends(require_auth),
    training_profile: str = Form("balanced"),
    caption_mode: str = Form("auto"),
    file: UploadFile = File(...),
):
    clean_name = name.strip()
    clean_trigger = trigger_word.strip()
    profile_key = training_profile.strip().lower()
    caption_mode_key = caption_mode.strip().lower()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Field 'name' is required")
    if not clean_trigger:
        raise HTTPException(status_code=400, detail="Field 'trigger_word' is required")
    if profile_key not in TRAINING_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid training profile: {training_profile}. Use one of: {', '.join(TRAINING_PROFILES)}",
        )
    if caption_mode_key not in CAPTION_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid caption mode: {caption_mode}. Use one of: {', '.join(sorted(CAPTION_MODES))}",
        )
    if not file.filename:
        raise HTTPException(status_code=400, detail="ZIP file is required")

    allowed_content_types = {
        "application/zip",
        "application/x-zip-compressed",
        "application/octet-stream",
    }
    filename_lower = file.filename.lower()
    is_zip_by_name = filename_lower.endswith(".zip")
    if file.content_type not in allowed_content_types and not is_zip_by_name:
        raise HTTPException(status_code=400, detail="Only ZIP uploads are supported")

    tmp_dir = Path(DATA_DIR) / "tmp" / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"entity_upload_{new_message_id()}.zip"
    try:
        file_bytes = file.file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded ZIP is empty")
        kind = filetype.guess(file_bytes)
        zip_mimes = {"application/zip", "application/x-zip-compressed"}
        if kind is None or kind.mime not in zip_mimes:
            raise HTTPException(
                status_code=400,
                detail="File is not a valid ZIP archive (magic bytes check failed). Use .zip format.",
            )
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        entity = create_entity(
            name=clean_name,
            trigger_word=clean_trigger,
            uploaded_filename=file.filename,
            temp_zip_path=tmp_path,
            user_id=user["user_id"],
        )
        entity_id = str(entity["id"])
        raw_metadata = get_entity_metadata(entity_id)
        if raw_metadata is None:
            raise HTTPException(status_code=500, detail="Failed to read entity metadata")

        uploaded_zip_raw = raw_metadata.get("uploaded_zip_path")
        if not isinstance(uploaded_zip_raw, str) or not uploaded_zip_raw.strip():
            update_entity_metadata(entity_id, {"status": "failed", "error": "Missing uploaded ZIP path"}, user_id=user["user_id"])
            raise HTTPException(status_code=500, detail="Failed to prepare dataset")

        uploaded_zip_path = Path(uploaded_zip_raw)
        entity_dir = Path(DATA_DIR) / "storage" / "entities" / entity_id
        try:
            manifest = prepare_entity_dataset(
                entity_dir=entity_dir,
                zip_path=uploaded_zip_path,
                entity_id=entity_id,
            )
        except DatasetValidationError as e:
            update_entity_metadata(
                entity_id,
                {
                    "status": "failed",
                    "error": str(e),
                    "image_count": 0,
                },
                user_id=user["user_id"],
            )
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            update_entity_metadata(
                entity_id,
                {
                    "status": "failed",
                    "error": "Dataset preprocessing failed",
                    "image_count": 0,
                },
                user_id=user["user_id"],
            )
            raise HTTPException(status_code=500, detail=f"Failed to prepare dataset: {e}") from e

        try:
            caption_stats = apply_caption_mode(
                entity_dir=entity_dir,
                zip_path=uploaded_zip_path,
                trigger_word=clean_trigger,
                caption_mode=caption_mode_key,
            )
        except CaptioningError as e:
            update_entity_metadata(
                entity_id,
                {
                    "status": "failed",
                    "error": str(e),
                    "image_count": int(manifest.get("image_count", 0)),
                },
                user_id=user["user_id"],
            )
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            update_entity_metadata(
                entity_id,
                {
                    "status": "failed",
                    "error": f"Caption processing failed: {e}",
                    "image_count": int(manifest.get("image_count", 0)),
                },
                user_id=user["user_id"],
            )
            raise HTTPException(status_code=500, detail=f"Failed to process captions: {e}") from e

        updated_entity = update_entity_metadata(
            entity_id,
            {
                "image_count": int(manifest.get("image_count", 0)),
                "caption_mode": caption_mode_key,
                "caption_stats": caption_stats,
            },
            user_id=user["user_id"],
        )
        queue_info = training_queue.enqueue(
            entity_id=entity_id,
            trigger_word=clean_trigger,
            steps=int(TRAINING_PROFILES[profile_key]["steps"]),
            rank=int(TRAINING_PROFILES[profile_key]["rank"]),
            learning_rate=float(TRAINING_PROFILES[profile_key]["learning_rate"]),
            lr_scheduler=str(TRAINING_PROFILES[profile_key]["lr_scheduler"]),
            warmup_ratio=float(TRAINING_PROFILES[profile_key]["warmup_ratio"]),
        )

        started_entity = update_entity_metadata(
            entity_id,
            {
                "status": "queued",
                "training_job_id": queue_info["job_id"],
                "training_profile": profile_key,
                "training_params": TRAINING_PROFILES[profile_key],
                "error": None,
            },
            user_id=user["user_id"],
        )
        response_entity = started_entity or updated_entity or entity
        return {
            "status": "training_started",
            "job_id": queue_info["job_id"],
            "entity": response_entity,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload entity error: {}", e)
        raise HTTPException(status_code=500, detail=f"Failed to upload entity: {e}") from e
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/api/entities/{entity_id}/retrain")
def retrain_entity(
    entity_id: str,
    training_profile: str = Form("balanced"),
    caption_mode: str = Form("auto"),
    use_custom: str = Form("false"),
    steps: int = Form(1200),
    rank: int = Form(16),
    learning_rate: float = Form(1e-4),
    lr_scheduler: str = Form("polynomial"),
    warmup_ratio: float = Form(0.06),
    remove_filenames: str = Form("[]"),
    file: UploadFile | None = File(None),
    user=Depends(require_auth),
):
    entity = store_get_entity(entity_id, user_id=user["user_id"])
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    if entity.get("status") in ("queued", "training"):
        raise HTTPException(
            status_code=400,
            detail="Entity is already training. Wait for completion.",
        )

    trigger_word = str(entity.get("trigger_word", "")).strip()
    if not trigger_word:
        raise HTTPException(status_code=400, detail="Entity has no trigger word")

    entity_dir = Path(DATA_DIR) / "storage" / "entities" / entity_id
    tmp_dir = Path(DATA_DIR) / "tmp" / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    zip_path: Path | None = None

    try:
        import json as _json
        to_remove: list[str] = []
        try:
            to_remove = _json.loads(remove_filenames or "[]")
        except (_json.JSONDecodeError, TypeError):
            pass
        if not isinstance(to_remove, list):
            to_remove = []

        if to_remove:
            remove_result = remove_dataset_images(
                entity_dir=entity_dir,
                filenames=to_remove,
                entity_id=entity_id,
            )
            update_entity_metadata(entity_id, {"image_count": remove_result["total"]}, user_id=user["user_id"])

        if file and file.filename and file.filename.lower().endswith(".zip"):
            tmp_path = tmp_dir / f"retrain_{new_message_id()}.zip"
            file_bytes = file.file.read()
            if file_bytes:
                with open(tmp_path, "wb") as f:
                    f.write(file_bytes)
                zip_path = tmp_path
                add_result = add_images_from_zip(
                    entity_dir=entity_dir,
                    zip_path=zip_path,
                    entity_id=entity_id,
                )
                update_entity_metadata(entity_id, {"image_count": add_result["total"]}, user_id=user["user_id"])

        use_custom_bool = str(use_custom).strip().lower() in ("1", "true", "yes")
        if use_custom_bool:
            profile_key = "custom"
            steps_val = max(100, min(5000, steps))
            rank_val = max(4, min(64, rank))
            lr_val = max(1e-6, min(1e-2, learning_rate))
            lr_sched = str(lr_scheduler or "polynomial").strip().lower()
            warmup_val = max(0.0, min(0.25, warmup_ratio))
        else:
            profile_key = training_profile.strip().lower()
            if profile_key not in TRAINING_PROFILES:
                profile_key = "balanced"
            steps_val = TRAINING_PROFILES[profile_key]["steps"]
            rank_val = TRAINING_PROFILES[profile_key]["rank"]
            lr_val = TRAINING_PROFILES[profile_key]["learning_rate"]
            lr_sched = TRAINING_PROFILES[profile_key]["lr_scheduler"]
            warmup_val = TRAINING_PROFILES[profile_key]["warmup_ratio"]

        caption_mode_key = caption_mode.strip().lower()
        if caption_mode_key not in CAPTION_MODES:
            caption_mode_key = "auto"

        zip_for_caption = zip_path
        if not zip_for_caption:
            raw_meta = get_entity_metadata(entity_id) or {}
            saved = raw_meta.get("uploaded_zip_path")
            if saved and Path(saved).exists():
                zip_for_caption = Path(saved)

        if zip_for_caption:
            try:
                apply_caption_mode(
                    entity_dir=entity_dir,
                    zip_path=zip_for_caption,
                    trigger_word=trigger_word,
                    caption_mode=caption_mode_key,
                )
            except CaptioningError as e:
                update_entity_metadata(entity_id, {"status": "failed", "error": str(e)}, user_id=user["user_id"])
                raise HTTPException(status_code=400, detail=str(e)) from e

        update_entity_metadata(
            entity_id,
            {
                "caption_mode": caption_mode_key,
                "image_count": len(list_dataset_images(entity_id)),
            },
            user_id=user["user_id"],
        )

        queue_info = training_queue.enqueue(
            entity_id=entity_id,
            trigger_word=trigger_word,
            steps=steps_val,
            rank=rank_val,
            learning_rate=lr_val,
            lr_scheduler=lr_sched,
            warmup_ratio=warmup_val,
        )

        started_entity = update_entity_metadata(
            entity_id,
            {
                "status": "queued",
                "training_job_id": queue_info["job_id"],
                "training_profile": profile_key,
                "training_params": {
                    "steps": steps_val,
                    "rank": rank_val,
                    "learning_rate": lr_val,
                    "lr_scheduler": lr_sched,
                    "warmup_ratio": warmup_val,
                },
                "error": None,
            },
            user_id=user["user_id"],
        )

        return {
            "status": "training_started",
            "job_id": queue_info["job_id"],
            "entity": started_entity,
        }
    except HTTPException:
        raise
    except DatasetValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        print(f"Retrain error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if zip_path and zip_path.exists():
            zip_path.unlink(missing_ok=True)


@app.put("/api/entities/{entity_id}")
def update_entity(entity_id: str, req: UpdateEntityRequest, user=Depends(get_current_user)):
    entity = store_get_entity(entity_id, user_id=user["user_id"] if user else None)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")

    updates: dict[str, Any] = {}
    if req.name is not None:
        clean_name = req.name.strip()
        if not clean_name:
            raise HTTPException(status_code=400, detail="Field 'name' cannot be empty")
        updates["name"] = clean_name
    if req.trigger_word is not None:
        clean_trigger = req.trigger_word.strip()
        if not clean_trigger:
            raise HTTPException(status_code=400, detail="Field 'trigger_word' cannot be empty")
        updates["trigger_word"] = clean_trigger

    if not updates:
        return entity

    result = update_entity_metadata(entity_id, updates, user_id=user["user_id"] if user else None)
    if result is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return result


@app.delete("/api/entities/{entity_id}")
def delete_entity(entity_id: str, user=Depends(get_current_user)):
    if not store_delete_entity(entity_id, user_id=user["user_id"] if user else None):
        raise HTTPException(status_code=404, detail="Entity not found")
    return {"status": "deleted"}
