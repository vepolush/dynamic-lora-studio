"""Shared utilities for experiment scripts."""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FRONTEND = ROOT / "frontend"
if str(FRONTEND) not in sys.path:
    sys.path.insert(0, str(FRONTEND))

import httpx

EXPERIMENTS_DIR = Path(__file__).parent
PROMPT_LORA_FILE = EXPERIMENTS_DIR / "prompt_lora.txt"
PROMPT_NO_LORA_FILE = EXPERIMENTS_DIR / "prompt_no_lora.txt"
# Legacy single-file fallback
PROMPT_FILE = EXPERIMENTS_DIR / "prompt.txt"
OUTPUT_DIR = EXPERIMENTS_DIR / "output"

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 120.0
GEN_TIMEOUT = 180.0

PLACEHOLDER = "{trigger}"


def _load_and_replace(path: Path, trigger: str | None) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    text = path.read_text(encoding="utf-8").strip()
    repl = (trigger or "").strip()
    result = text.replace(PLACEHOLDER, repl)
    while "  " in result:
        result = result.replace("  ", " ")
    return result.strip()


def load_prompt_lora(trigger: str) -> str:
    """Prompt for generation WITH LoRA — shorter, relies on trigger word."""
    if PROMPT_LORA_FILE.exists():
        return _load_and_replace(PROMPT_LORA_FILE, trigger)
    return _load_and_replace(PROMPT_FILE, trigger)


def load_prompt_no_lora(trigger: str | None = None) -> str:
    """Prompt for generation WITHOUT LoRA — detailed description of the subject."""
    if PROMPT_NO_LORA_FILE.exists():
        return _load_and_replace(PROMPT_NO_LORA_FILE, trigger)
    return _load_and_replace(PROMPT_FILE, None)


def load_prompt(trigger: str | None = None) -> str:
    """Legacy: load from prompt.txt (or prompt_lora.txt if trigger given)."""
    if trigger:
        return load_prompt_lora(trigger)
    return load_prompt_no_lora(trigger)


def save_images_from_response(
    response: dict,
    session_id: str,
    subdir: str = "",
) -> list[Path]:
    """Decode base64 images from generate response, save to output/, return paths."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_sub = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
    out_sub.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for img in response.get("images", []):
        b64 = img.get("base64")
        fname = img.get("filename", "out.png")
        seed = img.get("seed", "?")
        if not b64:
            continue
        if isinstance(b64, str) and b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        try:
            raw = base64.b64decode(b64)
        except Exception:
            continue
        stem = Path(fname).stem
        out_name = f"{stem}_seed{seed}.png"
        out_path = out_sub / out_name
        out_path.write_bytes(raw)
        paths.append(out_path)
        print(f"    -> {out_path}")
    return paths


def get_client(token: str | None = None) -> "ExperimentClient":
    return ExperimentClient(BACKEND_URL, token=token)


def ensure_auth(client: "ExperimentClient", username: str = "exp_user", password: str = "exp_pass_123") -> str:
    try:
        client.register(username, password)
    except Exception:
        pass
    data = client.login(username, password)
    return data["token"]


def wait_for_entity_ready(
    client: "ExperimentClient",
    entity_id: str,
    poll_interval: float = 3.0,
    max_wait: float = 600.0,
) -> tuple[bool, float]:
    """Poll until entity ready/failed. Returns (success, elapsed_seconds)."""
    start = time.time()
    while time.time() - start < max_wait:
        entities = client.get_entities()
        for e in entities:
            if e.get("id") == entity_id:
                status = e.get("status", "")
                if status == "ready":
                    return True, time.time() - start
                if status == "failed":
                    return False, time.time() - start
        time.sleep(poll_interval)
    return False, time.time() - start


class ExperimentClient:
    """Minimal API client for experiments."""

    def __init__(self, base_url: str = BACKEND_URL, token: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        timeout = kwargs.pop("timeout", self.timeout)
        with httpx.Client(timeout=timeout) as c:
            r = c.request(method, url, headers=self._headers(), **kwargs)
        if r.status_code >= 400:
            raise RuntimeError(f"API error {r.status_code}: {r.text}")
        return r.json() if r.content else {}

    def register(self, username: str, password: str) -> dict:
        return self._request("POST", "/api/auth/register", json={"username": username, "password": password})

    def login(self, username: str, password: str) -> dict:
        return self._request("POST", "/api/auth/login", json={"username": username, "password": password})

    def create_session(self, title: str = "Experiment") -> dict:
        return self._request("POST", "/api/sessions", json={"title": title})

    def get_entities(self) -> list[dict]:
        data = self._request("GET", "/api/entities")
        return data.get("entities", [])

    def get_entity(self, entity_id: str) -> dict | None:
        try:
            return self._request("GET", f"/api/entities/{entity_id}")
        except RuntimeError:
            return None

    def upload_entity(
        self,
        name: str,
        trigger_word: str,
        zip_path: Path,
        training_profile: str = "balanced",
        caption_mode: str = "auto",
    ) -> dict:
        with open(zip_path, "rb") as f:
            zip_bytes = f.read()
        files = {"file": (zip_path.name, zip_bytes, "application/zip")}
        data = {
            "name": name,
            "trigger_word": trigger_word,
            "training_profile": training_profile,
            "caption_mode": caption_mode,
        }
        with httpx.Client(timeout=300.0) as c:
            r = c.post(
                f"{self.base_url}/api/entities",
                headers=self._headers(),
                data=data,
                files=files,
            )
        if r.status_code >= 400:
            raise RuntimeError(f"Upload error {r.status_code}: {r.text}")
        out = r.json()
        return out.get("entity", out)

    def retrain_entity(
        self,
        entity_id: str,
        *,
        steps: int = 1200,
        rank: int = 16,
        learning_rate: float = 1e-4,
        lr_scheduler: str = "polynomial",
        warmup_ratio: float = 0.06,
        use_custom: bool = True,
    ) -> dict:
        import json as _json
        data = {
            "training_profile": "balanced",
            "caption_mode": "auto",
            "use_custom": "true" if use_custom else "false",
            "steps": str(steps),
            "rank": str(rank),
            "learning_rate": str(learning_rate),
            "lr_scheduler": lr_scheduler,
            "warmup_ratio": str(warmup_ratio),
            "remove_filenames": _json.dumps([]),
        }
        return self._request("POST", f"/api/entities/{entity_id}/retrain", data=data, timeout=300.0)

    def generate(
        self,
        session_id: str,
        prompt: str,
        *,
        entity_id: str | None = None,
        seed: int = -1,
        num_images: int = 1,
    ) -> dict:
        payload: dict = {
            "session_id": session_id,
            "prompt": prompt,
            "width": 512,
            "height": 512,
            "num_images": num_images,
        }
        if seed >= 0:
            payload["seed"] = seed
        if entity_id:
            payload["entity_id"] = entity_id
        return self._request("POST", "/api/generate", json=payload, timeout=GEN_TIMEOUT)
