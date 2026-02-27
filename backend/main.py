from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import os
import uuid

from model_manager import ml_manager

_sessions: list[dict] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_manager.load_model()
    except Exception as e:
        print(f"❌ Model loading error: {e}")
    yield
    print("Server shutting down...")

app = FastAPI(lifespan=lifespan)

class CreateSessionRequest(BaseModel):
    title: str = "New session"


class GenerateRequest(BaseModel):
    session_id: str
    prompt: str
    steps: int = 25
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: int | None = None
    lora_strength: float = 0.8

@app.get("/health")
def health_check():
    """Health check endpoint"""
    status = "ready" if ml_manager.pipe is not None else "loading"
    return {"status": status}

@app.get("/api/sessions")
def get_sessions():
    """Return all sessions"""
    return {"sessions": _sessions}


@app.post("/api/sessions")
def create_session(req: CreateSessionRequest):
    """Create a new session"""
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    session = {
        "id": session_id,
        "title": req.title,
        "created_at": now,
        "prompt": "",
        "helper_specs": "",
        "images": 0,
        "favourite": False,
    }
    _sessions.insert(0, session)
    return session

@app.get("/api/entities")
def get_entities():
    """Placeholder for entities"""
    return {"entities": []}

@app.post("/api/generate")
def generate_image(req: GenerateRequest):
    """Image generation endpoint"""
    if ml_manager.pipe is None:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to load")
    
    try:
        print(f"🎨 Generating: '{req.prompt}' (Steps: {req.steps})")
        start_time = time.time()
        
        image = ml_manager.generate(req.prompt, req.steps, req.guidance_scale)
        
        os.makedirs("/workspace/backend/output", exist_ok=True)
        filename = f"{req.session_id}_{int(time.time())}.png"
        filepath = f"/workspace/backend/output/{filename}"
        image.save(filepath)
        
        gen_time = time.time() - start_time
        print(f"✅ Done in {gen_time:.2f} seconds!")
        
        return {"status": "success", "image_url": filepath, "generation_time": gen_time}
        
    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))