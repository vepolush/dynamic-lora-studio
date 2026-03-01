"""Dynamic LoRA Studio — FastAPI backend."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from entity_store import DATA_DIR, create_entity, get_entity as store_get_entity, list_entities as store_list_entities
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_manager.load_model()
    except Exception as e:
        print(f"Model loading error: {e}")
    yield
    print("Server shutting down...")


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    title: str = "New session"


class UpdateSessionRequest(BaseModel):
    title: str | None = None
    favourite: bool | None = None


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
    lora_strength: float = 0.8


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    status = "ready" if ml_manager.pipe is not None else "loading"
    return {"status": status}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
def get_sessions():
    return {"sessions": store_list_sessions()}


@app.post("/api/sessions")
def create_session(req: CreateSessionRequest):
    return store_create_session(title=req.title)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = store_get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.put("/api/sessions/{session_id}")
def update_session(session_id: str, req: UpdateSessionRequest):
    updates: dict[str, Any] = {}
    if req.title is not None:
        updates["title"] = req.title
    if req.favourite is not None:
        updates["favourite"] = req.favourite
    result = store_update_session(session_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    if not store_delete_session(session_id):
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
def generate_image(req: GenerateRequest):
    if ml_manager.pipe is None:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to load")

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

        print(f"Generating: '{enhanced_prompt}' | neg: '{negative[:80]}...' | steps={steps} sched={scheduler}")
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
        )

        gen_time = time.time() - start_time
        print(f"Done in {gen_time:.2f}s — {len(results)} image(s)")

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
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Entities (placeholder)
# ---------------------------------------------------------------------------

@app.get("/api/entities")
def get_entities():
    return {"entities": store_list_entities()}


@app.get("/api/entities/{entity_id}")
def get_entity(entity_id: str):
    entity = store_get_entity(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return entity


@app.post("/api/entities")
def upload_entity(
    name: str = Form(...),
    trigger_word: str = Form(...),
    file: UploadFile = File(...),
):
    clean_name = name.strip()
    clean_trigger = trigger_word.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Field 'name' is required")
    if not clean_trigger:
        raise HTTPException(status_code=400, detail="Field 'trigger_word' is required")
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
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        entity = create_entity(
            name=clean_name,
            trigger_word=clean_trigger,
            uploaded_filename=file.filename,
            temp_zip_path=tmp_path,
        )
        return entity
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload entity: {e}") from e
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
