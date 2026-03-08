"""
Exp 3: LoRA hyperparameters — rank, learning rate, steps.

Run: python -m experiments.exp3_hyperparams [path/to/images.zip]
Requires: backend running, GPU.

If no ZIP: uses first ready entity and retrains with different params.
Output: table for docs/experiments.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from experiments.utils import ensure_auth, get_client, wait_for_entity_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3: LoRA hyperparams")
    parser.add_argument("zip_path", nargs="?", type=Path, help="ZIP with 10-15 images")
    parser.add_argument("--trigger", default="<my_subject>", help="Trigger word")
    args = parser.parse_args()

    client = get_client()
    token = ensure_auth(client)
    client.token = token

    session = client.create_session("Exp3 Hyperparams")
    session_id = session["id"]

    entity_id: str | None = None
    if args.zip_path and args.zip_path.exists():
        print("Uploading entity...")
        entity = client.upload_entity(
            name="Exp3 Base",
            trigger_word=args.trigger,
            zip_path=args.zip_path,
            training_profile="balanced",
        )
        entity_id = entity["id"]
        if not wait_for_entity_ready(client, entity_id):
            print("Training failed")
            sys.exit(1)
    else:
        entities = [e for e in client.get_entities() if e.get("status") == "ready"]
        if not entities:
            print("No ready entity. Provide ZIP.")
            sys.exit(1)
        entity_id = entities[0]["id"]

    configs = [
        {"name": "Rank 4", "steps": 1200, "rank": 4, "lr": 1e-4},
        {"name": "Rank 8", "steps": 1200, "rank": 8, "lr": 1e-4},
        {"name": "Rank 16", "steps": 1200, "rank": 16, "lr": 1e-4},
        {"name": "Rank 32", "steps": 1200, "rank": 32, "lr": 1e-4},
    ]

    print("\n--- Retrain with different ranks ---")
    for cfg in configs:
        t0 = time.time()
        client.retrain_entity(
            entity_id,
            steps=cfg["steps"],
            rank=cfg["rank"],
            learning_rate=cfg["lr"],
            use_custom=True,
        )
        if wait_for_entity_ready(client, entity_id):
            elapsed = time.time() - t0
            cfg["train_time"] = elapsed
            r = client.generate(session_id, f"{args.trigger} portrait", entity_id=entity_id, seed=42)
            cfg["gen_time"] = r.get("generation_time", 0)
            print(f"  {cfg['name']}: train={elapsed:.1f}s gen={cfg['gen_time']:.1f}s")

    print("\n--- Table for docs/experiments.md ---")
    print("| Профіль | Rank | Steps | LR   | Час тренування | Якість (1–5) |")
    print("|---------|------|-------|------|----------------|--------------|")
    for cfg in configs:
        t = cfg.get("train_time", 0)
        print(f"| {cfg['name']} | {cfg['rank']} | {cfg['steps']} | 1e-4 | {t:.1f}s | |")


if __name__ == "__main__":
    main()
