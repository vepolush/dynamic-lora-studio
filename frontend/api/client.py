"""HTTP client for backend API."""

from __future__ import annotations

from typing import Any

import httpx

from config import BACKEND_URL

DEFAULT_TIMEOUT = 120.0
QUICK_TIMEOUT = 5.0
ENTITY_UPLOAD_TIMEOUT = 120.0


class BackendError(Exception):
    """Raised when backend returns an error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class APIClient:
    """Sync HTTP client for Dynamic LoRA Studio backend."""

    def __init__(
        self,
        base_url: str = BACKEND_URL,
        timeout: float = DEFAULT_TIMEOUT,
        token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token = token

    def _headers(self) -> dict[str, str]:
        h = {}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        data: dict[str, str] | None = None,
        files: dict[str, tuple[str, bytes]] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any] | list[Any]:
        url = f"{self.base_url}{path}"
        req_timeout = timeout if timeout is not None else self.timeout
        headers = self._headers()
        try:
            with httpx.Client(timeout=req_timeout) as client:
                if files:
                    form_data = dict(data) if data else {}
                    if json:
                        form_data.update({k: str(v) for k, v in json.items()})
                    response = client.request(
                        method, url, data=form_data, files=files, headers=headers, timeout=req_timeout
                    )
                elif data:
                    response = client.request(method, url, data=data, headers=headers, timeout=req_timeout)
                else:
                    response = client.request(method, url, json=json, headers=headers, timeout=req_timeout)
        except httpx.ConnectError as e:
            raise BackendError(f"Cannot connect to backend at {self.base_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise BackendError(f"Request timed out: {e}") from e

        if response.status_code >= 400:
            raise BackendError(
                f"Backend error: {response.text or response.reason_phrase}",
                status_code=response.status_code,
            )

        if response.status_code == 204 or not response.content:
            return {}

        return response.json()

    def _get_bytes(self, path: str, timeout: float | None = None) -> bytes | None:
        """Fetch raw bytes (e.g. for images). Returns None on error."""
        url = f"{self.base_url}{path}"
        req_timeout = timeout if timeout is not None else QUICK_TIMEOUT
        try:
            with httpx.Client(timeout=req_timeout) as client:
                response = client.get(url, headers=self._headers())
                if response.status_code != 200:
                    return None
                return response.content
        except (httpx.ConnectError, httpx.TimeoutException):
            return None

    # ---- Auth ----

    def register(self, username: str, password: str) -> dict[str, Any]:
        return self._request("POST", "/api/auth/register", json={"username": username, "password": password})  # type: ignore

    def login(self, username: str, password: str) -> dict[str, Any]:
        return self._request("POST", "/api/auth/login", json={"username": username, "password": password})  # type: ignore

    # ---- Gallery ----

    def get_gallery(self, sort: str = "newest", limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        data = self._request("GET", f"/api/gallery?sort={sort}&limit={limit}&offset={offset}", timeout=QUICK_TIMEOUT)
        return data.get("images", []) if isinstance(data, dict) else []

    def get_gallery_image(self, image_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/gallery/{image_id}", timeout=QUICK_TIMEOUT)  # type: ignore

    def get_gallery_image_bytes(self, image_id: str) -> bytes | None:
        return self._get_bytes(f"/api/gallery/{image_id}/image", timeout=QUICK_TIMEOUT)

    def get_published_filenames(self, session_id: str) -> list[str]:
        """Get filenames from session that are already published to gallery."""
        data = self._request("GET", f"/api/gallery/published-filenames?session_id={session_id}", timeout=QUICK_TIMEOUT)
        return data.get("filenames", []) if isinstance(data, dict) else []

    def publish_to_gallery(self, session_id: str, filename: str, prompt: str, settings: dict | None = None) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/gallery/publish",
            json={"session_id": session_id, "filename": filename, "prompt": prompt, "settings": settings},
            timeout=QUICK_TIMEOUT,
        )  # type: ignore

    def like_gallery_image(self, image_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/gallery/{image_id}/like", timeout=QUICK_TIMEOUT)  # type: ignore

    # ---- Gallery LoRAs ----

    def publish_lora_to_gallery(
        self,
        entity_id: str,
        name: str,
        trigger_word: str,
        description: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "entity_id": entity_id,
            "name": name,
            "trigger_word": trigger_word,
        }
        if description is not None:
            payload["description"] = description
        return self._request("POST", "/api/gallery/loras/publish", json=payload, timeout=QUICK_TIMEOUT)  # type: ignore

    def get_gallery_loras(
        self,
        sort: str = "newest",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        data = self._request(
            "GET",
            f"/api/gallery/loras?sort={sort}&limit={limit}&offset={offset}",
            timeout=QUICK_TIMEOUT,
        )
        return data.get("loras", []) if isinstance(data, dict) else []

    def get_gallery_lora(self, lora_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/gallery/loras/{lora_id}", timeout=QUICK_TIMEOUT)  # type: ignore

    def get_gallery_lora_preview_bytes(self, lora_id: str) -> bytes | None:
        return self._get_bytes(f"/api/gallery/loras/{lora_id}/preview", timeout=QUICK_TIMEOUT)

    def add_gallery_lora(self, lora_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/gallery/loras/{lora_id}/add", timeout=QUICK_TIMEOUT)  # type: ignore

    def unpublish_gallery_lora(self, lora_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/api/gallery/loras/{lora_id}", timeout=QUICK_TIMEOUT)  # type: ignore

    def get_entity_preview_bytes(self, entity_id: str) -> bytes | None:
        """Fetch entity preview image as bytes. Returns None if not found."""
        return self._get_bytes(f"/api/entities/{entity_id}/preview", timeout=QUICK_TIMEOUT)

    def regenerate_entity_preview(self, entity_id: str) -> dict[str, Any]:
        """Regenerate entity preview image. Returns dict with preview_url or raises BackendError."""
        return self._request("POST", f"/api/entities/{entity_id}/regenerate-preview", timeout=60)  # type: ignore

    def get_session_image_bytes(self, session_id: str, filename: str) -> bytes | None:
        """Fetch session image as bytes. Returns None if not found."""
        return self._get_bytes(f"/api/images/{session_id}/{filename}", timeout=QUICK_TIMEOUT)

    # ---- Sessions ----

    def get_sessions(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/sessions", timeout=QUICK_TIMEOUT)
        return data if isinstance(data, list) else data.get("sessions", [])

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/sessions/{session_id}")  # type: ignore

    def create_session(self, title: str = "New session") -> dict[str, Any]:
        return self._request("POST", "/api/sessions", json={"title": title}, timeout=QUICK_TIMEOUT)  # type: ignore

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        favourite: bool | None = None,
        favourite_image_filenames: list[str] | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if title is not None:
            payload["title"] = title
        if favourite is not None:
            payload["favourite"] = favourite
        if favourite_image_filenames is not None:
            payload["favourite_image_filenames"] = favourite_image_filenames
        if archived is not None:
            payload["archived"] = archived
        return self._request("PUT", f"/api/sessions/{session_id}", json=payload, timeout=QUICK_TIMEOUT)  # type: ignore

    def delete_session(self, session_id: str) -> None:
        self._request("DELETE", f"/api/sessions/{session_id}", timeout=QUICK_TIMEOUT)

    # ---- Entities ----

    def get_entities(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/entities", timeout=QUICK_TIMEOUT)
        return data if isinstance(data, list) else data.get("entities", [])

    def upload_entity(
        self,
        name: str,
        trigger_word: str,
        zip_bytes: bytes,
        training_profile: str = "balanced",
        caption_mode: str = "auto",
        filename: str = "images.zip",
        subject_type: str | None = None,
    ) -> dict[str, Any]:
        files = {"file": (filename, zip_bytes, "application/zip")}
        payload: dict[str, Any] = {
            "name": name,
            "trigger_word": trigger_word,
            "training_profile": training_profile,
            "caption_mode": caption_mode,
        }
        if subject_type:
            payload["subject_type"] = subject_type
        data = self._request(
            "POST",
            "/api/entities",
            data=payload,
            files=files,
            timeout=ENTITY_UPLOAD_TIMEOUT,
        )  # type: ignore
        if isinstance(data, dict):
            entity = data.get("entity")
            if isinstance(entity, dict):
                if "status" in data:
                    entity["_upload_status"] = data["status"]
                if "job_id" in data:
                    entity["_training_job_id"] = data["job_id"]
                return entity
        return data  # type: ignore

    def update_entity(
        self,
        entity_id: str,
        *,
        name: str | None = None,
        trigger_word: str | None = None,
        subject_type: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if trigger_word is not None:
            payload["trigger_word"] = trigger_word
        if subject_type is not None:
            payload["subject_type"] = subject_type
        return self._request("PUT", f"/api/entities/{entity_id}", json=payload, timeout=QUICK_TIMEOUT)  # type: ignore

    def get_entity_dataset(self, entity_id: str) -> list[dict[str, Any]]:
        data = self._request("GET", f"/api/entities/{entity_id}/dataset", timeout=QUICK_TIMEOUT)
        return data.get("images", []) if isinstance(data, dict) else []

    def get_dataset_image_bytes(self, entity_id: str, filename: str) -> bytes | None:
        from urllib.parse import quote
        path = f"/api/entities/{entity_id}/dataset/{quote(filename, safe='')}"
        return self._get_bytes(path, timeout=QUICK_TIMEOUT)

    def remove_dataset_images(self, entity_id: str, filenames: list[str]) -> dict[str, Any]:
        return self._request(
            "DELETE",
            f"/api/entities/{entity_id}/dataset",
            json={"filenames": filenames},
            timeout=QUICK_TIMEOUT,
        )  # type: ignore

    def retrain_entity(
        self,
        entity_id: str,
        *,
        zip_bytes: bytes | None = None,
        filename: str = "images.zip",
        remove_filenames: list[str] | None = None,
        training_profile: str = "balanced",
        caption_mode: str = "auto",
        use_custom: bool = False,
        steps: int = 1200,
        rank: int = 16,
        learning_rate: float = 1e-4,
        lr_scheduler: str = "polynomial",
        warmup_ratio: float = 0.06,
    ) -> dict[str, Any]:
        import json as _json
        form_data: dict[str, str] = {
            "training_profile": training_profile,
            "caption_mode": caption_mode,
            "use_custom": "true" if use_custom else "false",
            "steps": str(steps),
            "rank": str(rank),
            "learning_rate": str(learning_rate),
            "lr_scheduler": lr_scheduler,
            "warmup_ratio": str(warmup_ratio),
            "remove_filenames": _json.dumps(remove_filenames or []),
        }
        files: dict[str, tuple[str, bytes]] = {}
        if zip_bytes:
            files["file"] = (filename, zip_bytes, "application/zip")
        return self._request(
            "POST",
            f"/api/entities/{entity_id}/retrain",
            data=form_data,
            files=files if files else None,
            timeout=ENTITY_UPLOAD_TIMEOUT,
        )  # type: ignore

    def delete_entity(self, entity_id: str) -> None:
        self._request("DELETE", f"/api/entities/{entity_id}")

    # ---- Generation ----

    def generate(
        self,
        session_id: str,
        prompt: str,
        *,
        negative_prompt: str = "",
        steps: int | None = None,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        num_images: int = 1,
        scheduler: str | None = None,
        quality: str = "Normal",
        entity_id: str | None = None,
        entity_version: str | None = None,
        lora_strength: float = 0.8,
        style: str | None = None,
        lighting: str | None = None,
        color: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": session_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed if seed >= 0 else None,
            "num_images": num_images,
            "quality": quality,
            "lora_strength": lora_strength,
        }
        if steps and steps > 0:
            payload["steps"] = steps
        if scheduler:
            payload["scheduler"] = scheduler
        if entity_id:
            payload["entity_id"] = entity_id
        if entity_version:
            payload["entity_version"] = entity_version
        if style and style != "None":
            payload["style"] = style
        if lighting and lighting != "None":
            payload["lighting"] = lighting
        if color and color != "Default":
            payload["color"] = color

        return self._request("POST", "/api/generate", json=payload)  # type: ignore

    # ---- Health ----

    def health(self) -> bool:
        try:
            self._request("GET", "/health", timeout=QUICK_TIMEOUT)
            return True
        except BackendError:
            return False
