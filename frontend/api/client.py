"""HTTP client for backend API."""

from __future__ import annotations

from typing import Any

import httpx

from config import BACKEND_URL

DEFAULT_TIMEOUT = 90.0
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

    def get_sessions(self) -> list[dict[str, Any]]:
        """Fetch list of sessions."""
        data = self._request("GET", "/api/sessions", timeout=QUICK_TIMEOUT)
        return data if isinstance(data, list) else data.get("sessions", [])

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Fetch single session with images."""
        return self._request("GET", f"/api/sessions/{session_id}")  # type: ignore

    def create_session(self, title: str = "New session") -> dict[str, Any]:
        """Create a new session."""
        return self._request("POST", "/api/sessions", json={"title": title}, timeout=QUICK_TIMEOUT)  # type: ignore

    def get_entities(self) -> list[dict[str, Any]]:
        """Fetch list of entities."""
        data = self._request("GET", "/api/entities", timeout=QUICK_TIMEOUT)
        return data if isinstance(data, list) else data.get("entities", [])

    def upload_entity(
        self,
        name: str,
        trigger_word: str,
        zip_bytes: bytes,
        filename: str = "images.zip",
    ) -> dict[str, Any]:
        """Upload ZIP and start entity training."""
        files = {"file": (filename, zip_bytes, "application/zip")}
        return self._request(
            "POST",
            "/api/entities",
            data={"name": name, "trigger_word": trigger_word},
            files=files,
        )  # type: ignore

    def delete_entity(self, entity_id: str) -> None:
        """Delete an entity."""
        self._request("DELETE", f"/api/entities/{entity_id}")

    def generate(
        self,
        session_id: str,
        prompt: str,
        *,
        steps: int = 25,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
        entity_id: str | None = None,
        lora_strength: float = 0.8,
        style: str | None = None,
        lightning: str | None = None,
        color: str | None = None,
    ) -> dict[str, Any]:
        """Request image generation."""
        payload: dict[str, Any] = {
            "session_id": session_id,
            "prompt": prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed if seed >= 0 else None,
            "lora_strength": lora_strength,
        }
        if entity_id:
            payload["entity_id"] = entity_id
        if style:
            payload["style"] = style
        if lightning:
            payload["lightning"] = lightning
        if color:
            payload["color"] = color

        return self._request("POST", "/api/generate", json=payload)  # type: ignore

    def health(self) -> bool:
        """Check if backend is reachable."""
        try:
            self._request("GET", "/health", timeout=QUICK_TIMEOUT)
            return True
        except BackendError:
            return False
