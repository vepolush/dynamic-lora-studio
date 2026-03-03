#!/bin/bash
# Dynamic LoRA Studio — RunPod startup script
# Run from project root: /workspace/dynamic-lora-studio/

set -e
cd /workspace/dynamic-lora-studio

# Optional: set data/model dirs (defaults: /workspace/data, /workspace/models/sd-1-5)
export DATA_DIR="${DATA_DIR:-/workspace/data}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

echo "Installing dependencies..."
pip install -q -r backend/requirements.txt
pip install -q -r frontend/requirements.txt

echo "Starting backend on port 8000..."
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

echo "Waiting for backend (model loading)..."
python -c "
import time, urllib.request, json
for _ in range(90):
    try:
        r = urllib.request.urlopen('http://localhost:8000/health', timeout=2)
        d = json.loads(r.read().decode())
        if d.get('status') == 'ready':
            print('Backend ready.')
            break
    except Exception:
        pass
    time.sleep(2)
else:
    print('Backend still loading; frontend will retry.')
"

echo "Starting frontend on port 8501..."
cd frontend
export BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false

kill $BACKEND_PID 2>/dev/null || true
