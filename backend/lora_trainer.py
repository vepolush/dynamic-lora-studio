"""LoRA training entrypoint for background worker jobs."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from entity_store import DATA_DIR


def _version_sort_key(name: str) -> int:
    match = re.match(r"v(\d+)", name.lower())
    return int(match.group(1)) if match else 0


def _next_version_name(weights_dir: Path, rank: int, steps: int) -> str:
    existing = [p.name for p in weights_dir.iterdir() if p.is_dir()]
    if not existing:
        next_index = 1
    else:
        next_index = max(_version_sort_key(name) for name in existing) + 1
    return f"v{next_index}_rank{rank}_steps{steps}"


def train_lora_for_entity(
    *,
    entity_id: str,
    trigger_word: str,
    steps: int = 500,
    rank: int = 8,
) -> dict[str, Any]:
    """
    Run LoRA training for one entity and persist training artifacts.
    """
    entity_dir = DATA_DIR / "storage" / "entities" / entity_id
    dataset_dir = entity_dir / "dataset"
    weights_dir = entity_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    dataset_images = [p for p in dataset_dir.glob("*.png") if p.is_file()]
    if not dataset_images:
        raise ValueError("Dataset is empty; no preprocessed images found")

    version_name = _next_version_name(weights_dir, rank=rank, steps=steps)
    version_dir = weights_dir / version_name
    version_dir.mkdir(parents=True, exist_ok=False)

    started_at = time.time()

    weights_path = version_dir / "pytorch_lora_weights.safetensors"
    with open(weights_path, "wb") as f:
        f.write(b"stage3-placeholder")

    duration_s = time.time() - started_at
    config = {
        "entity_id": entity_id,
        "trigger_word": trigger_word,
        "steps": steps,
        "rank": rank,
        "dataset_image_count": len(dataset_images),
        "weights_file": weights_path.name,
        "trainer": "stage3_placeholder",
        "training_time_seconds": round(duration_s, 3),
    }
    with open(version_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return {
        "version": version_name,
        "version_dir": str(version_dir),
        "weights_path": str(weights_path),
        "training_time_seconds": round(duration_s, 3),
    }
