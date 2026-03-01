"""HTTP client for backend API."""

from __future__ import annotations

from typing import Any

import httpx

from config import BACKEND_URL

DEFAULT_TIMEOUT = 120.0
QUICK_TIMEOUT = 5.0


class BackendError(Exception):
    """Raised when backend returns an error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class APIClient:
    """Sync HTTP client for Dynamic LoRA Studio backend."""

    def __init__(self, base_url: str = BACKEND_URL, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

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
        try:
            with httpx.Client(timeout=req_timeout) as client:
                if files:
                    form_data = dict(data) if data else {}
                    if json:
                        form_data.update({k: str(v) for k, v in json.items()})
                    response = client.request(
                        method, url, data=form_data, files=files, timeout=req_timeout
                    )
                else:
                    response = client.request(method, url, json=json, timeout=req_timeout)
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

    # ---- Sessions ----

    def get_sessions(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/sessions", timeout=QUICK_TIMEOUT)
        return data if isinstance(data, list) else data.get("sessions", [])

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/sessions/{session_id}")  # type: ignore

    def create_session(self, title: str = "New session") -> dict[str, Any]:
        return self._request("POST", "/api/sessions", json={"title": title}, timeout=QUICK_TIMEOUT)  # type: ignore

    def update_session(self, session_id: str, *, title: str | None = None, favourite: bool | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if title is not None:
            payload["title"] = title
        if favourite is not None:
            payload["favourite"] = favourite
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
        filename: str = "images.zip",
    ) -> dict[str, Any]:
        files = {"file": (filename, zip_bytes, "application/zip")}
        data = self._request(
            "POST",
            "/api/entities",
            data={"name": name, "trigger_word": trigger_word},
            files=files,
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
