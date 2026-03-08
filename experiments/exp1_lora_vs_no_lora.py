"""
Exp 1: Without LoRA vs With LoRA — subjective likeness evaluation.

Run: python -m experiments.exp1_lora_vs_no_lora [path/to/images.zip]
Requires: backend running, GPU.

If no ZIP provided, uses existing entity. Output: table for docs/experiments.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.utils import BACKEND_URL, ensure_auth, get_client, wait_for_entity_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1: LoRA vs no LoRA")
    parser.add_argument("zip_path", nargs="?", type=Path, help="ZIP with 10-15 images")
    parser.add_argument("--seeds", default="42,123,456", help="Comma-separated seeds")
    parser.add_argument("--trigger", default="<my_subject>", help="Trigger word")
    args = parser.parse_args()

    client = get_client()
    token = ensure_auth(client)
    client.token = token

    session = client.create_session("Exp1 LoRA vs No LoRA")
    session_id = session["id"]

    entity_id: str | None = None
    if args.zip_path:
        if not args.zip_path.exists():
            print(f"Error: {args.zip_path} not found")
            sys.exit(1)
        print("Uploading entity...")
        entity = client.upload_entity(
            name="Exp1 Subject",
            trigger_word=args.trigger,
            zip_path=args.zip_path,
            training_profile="balanced",
        )
        entity_id = entity["id"]
        print(f"Entity {entity_id} queued. Waiting for training...")
        if not wait_for_entity_ready(client, entity_id):
            print("Training failed or timed out")
            sys.exit(1)
        print("Entity ready.")
    else:
        entities = client.get_entities()
        ready = [e for e in entities if e.get("status") == "ready"]
        if not ready:
            print("No ready entity. Provide ZIP: python -m experiments.exp1_lora_vs_no_lora path/to/images.zip")
            sys.exit(1)
        entity_id = ready[0]["id"]
        args.trigger = ready[0].get("trigger_word", args.trigger)
        print(f"Using entity {entity_id}")

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    prompt_base = "a photo of a dog sitting on grass"  # or customize

    print("\n--- Generation without LoRA ---")
    for seed in seeds:
        r = client.generate(session_id, prompt_base, seed=seed)
        t = r.get("generation_time", 0)
        print(f"  Seed {seed}: {t:.1f}s")

    print("\n--- Generation with LoRA ---")
    for seed in seeds:
        prompt_lora = f"{args.trigger} sitting on grass"
        r = client.generate(session_id, prompt_lora, entity_id=entity_id, seed=seed)
        t = r.get("generation_time", 0)
        print(f"  Seed {seed}: {t:.1f}s")

    print("\n--- Table for docs/experiments.md ---")
    print("| Seed | Без LoRA (1–5) | З LoRA (1–5) | Примітки |")
    print("|------|----------------|--------------|----------|")
    for seed in seeds:
        print(f"| {seed}   |                |              |          |")


if __name__ == "__main__":
    main()
