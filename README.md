# Dynamic LoRA Studio

**Dynamic LoRA Management for Customizable Text-to-Image Generation** — Streamlit app for generating images from text prompts, uploading custom image sets (ZIP), training LoRA per entity, and managing entities (add/remove).

## Architecture

- **Frontend** (`frontend/`): Streamlit UI — runs anywhere (local, cloud)
- **Backend** (`backend/`): FastAPI, LoRA training, image generation — requires GPU

```
┌─────────────────┐     HTTP      ┌─────────────────┐
│  Streamlit      │ ────────────► │  FastAPI        │
│  (port 8501)    │               │  (port 8000)    │
│                 │               │  - Generation   │
│  - Prompts      │               │  - LoRA train   │
│  - Entities UI  │               │  - Sessions     │
│  - Settings     │               │  - Diffusers    │
└─────────────────┘               └─────────────────┘
```

## Tech Stack

- **Frontend**: Streamlit, Pillow, httpx
- **Backend**: FastAPI, Diffusers (SD 1.5), PyTorch, PEFT (LoRA), Transformers (BLIP captions)
- **Storage**: File-based (entities, sessions, base model, LoRA weights)

---

## Local Development

### Prerequisites

- Python 3.10+
- GPU with CUDA (for backend)
- 16+ GB VRAM recommended

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Base model (`runwayml/stable-diffusion-v1-5`) downloads on first run to `~/.cache/huggingface` or `MODEL_DIR`.

### Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

In Settings, set Backend URL to `http://localhost:8000` (default).

---

## RunPod Deployment

### Template

- **Image**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- **GPU**: T4 / A10G / A100 (8+ GB VRAM)
- **Expose ports**: 8000 (API), 8501 (Streamlit)
- **Volume**: Mount `/workspace` for persistence (models, data, cache)

### Setup

1. Create a RunPod Pod with the template above.
2. SSH or use the web terminal.
3. Clone the repo:

```bash
cd /workspace
git clone https://github.com/vepolush/dynamic-lora-studio.git
cd dynamic-lora-studio
```

4. Run the startup script:

```bash
chmod +x start.sh
./start.sh
```

**Optional**: In RunPod Pod template, set **Start Command** to:
```bash
cd /workspace/dynamic-lora-studio && ./start.sh
```
(requires repo to be cloned to `/workspace/dynamic-lora-studio` before first start)

Or run services manually in two terminals:

**Terminal 1 — Backend:**
```bash
cd /workspace/dynamic-lora-studio/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd /workspace/dynamic-lora-studio/frontend
streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false
```

### Access

- **Streamlit UI**: `https://<pod-id>-8501.proxy.runpod.net`
- **API**: `https://<pod-id>-8000.proxy.runpod.net`

The frontend uses `BACKEND_URL=http://localhost:8000` by default (both run on the same pod, so localhost works for server-side API calls).

### Environment Variables

| Variable      | Default                    | Description                    |
|---------------|----------------------------|--------------------------------|
| `DATA_DIR`    | `/workspace/data`          | Entities, sessions, DB, temp  |
| `HF_HOME`     | `/workspace/.cache/huggingface` | HuggingFace cache        |
| `BACKEND_URL` | `http://localhost:8000`    | Backend API URL (frontend)     |

Base model path: `/workspace/models/sd-1-5` (or downloaded from HuggingFace on first run).

---

## Project Structure

```
dynamic-lora-studio/
├── backend/
│   ├── main.py           # FastAPI app, generation, entities, sessions
│   ├── model_manager.py  # SD 1.5 pipeline, LoRA load/unload
│   ├── lora_trainer.py    # LoRA fine-tuning
│   ├── training_queue.py  # Background training worker
│   ├── dataset_prep.py   # ZIP ingest, image preprocessing
│   ├── captioning.py      # BLIP captions for training
│   ├── db.py             # SQLAlchemy models, SQLite
│   ├── entity_store.py   # Entity storage (DB + disk)
│   ├── session_store.py  # Session/message storage (DB + disk)
├── frontend/
│   ├── app.py            # Streamlit entry
│   ├── components/       # Sidebar, workspace, prompt_helper, settings
│   ├── services/         # API clients
│   └── state/            # Session state
├── start.sh              # RunPod startup script
└── README.md
```

---

## Features

- **Text-to-image generation** — with or without LoRA
- **Entity management** — upload ZIP (5–50 images), train LoRA, select for generation
- **Training profiles** — fast / balanced / strong (steps, rank, LR)
- **Caption modes** — Auto (BLIP), None, Use provided (.txt in ZIP)
- **Settings** — steps, guidance scale, size, LoRA strength, seed, quality, style, lighting
- **Sessions** — multiple generation sessions, favourites

---

## Testing

```bash
pip install pytest httpx
pip install -r backend/requirements.txt   # for API tests
python -m pytest tests/ -v
```

- `tests/test_dataset_prep.py` — dataset validation, path traversal, ZIP handling
- `tests/test_api.py` — health, auth, generation API (requires backend deps)
