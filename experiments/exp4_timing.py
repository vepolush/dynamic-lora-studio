"""
Exp 4: Generation time / Training time.

Run: python -m experiments.exp4_timing [path/to/images.zip]
Requires: backend running, GPU.

Measures: avg generation time (5 images), training time (if ZIP provided).
Output: table for docs/experiments.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from experiments.utils import ensure_auth, get_client, wait_for_entity_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp4: Timing")
    parser.add_argument("zip_path", nargs="?", type=Path, help="ZIP for training time measurement")
    parser.add_argument("--n-gen", type=int, default=5, help="Number of images to generate")
    args = parser.parse_args()

    client = get_client()
    token = ensure_auth(client)
    client.token = token

    session = client.create_session("Exp4 Timing")
    session_id = session["id"]

    # Generation time
    print(f"--- Generation time ({args.n_gen} images, 512x512, Normal) ---")
    times: list[float] = []
    for i in range(args.n_gen):
        t0 = time.time()
        r = client.generate(session_id, "a red apple on white background", seed=42 + i)
        elapsed = time.time() - t0
        times.append(elapsed)
        api_time = r.get("generation_time", 0)
        print(f"  Image {i+1}: {elapsed:.1f}s (API: {api_time:.1f}s)")
    avg = sum(times) / len(times) if times else 0
    print(f"  Average: {avg:.1f}s")

    # Training time (if ZIP provided)
    if args.zip_path and args.zip_path.exists():
        print("\n--- Training time ---")
        t0 = time.time()
        entity = client.upload_entity(
            name="Exp4 Timing",
            trigger_word="<test>",
            zip_path=args.zip_path,
            training_profile="balanced",
        )
        entity_id = entity["id"]
        if wait_for_entity_ready(client, entity_id):
            train_time = time.time() - t0
            n_imgs = client.get_entity(entity_id) or {}
            n_imgs = n_imgs.get("image_count", "?")
            print(f"  Dataset: {n_imgs} images, profile: balanced")
            print(f"  Training time: {train_time:.1f}s")

    print("\n--- Table for docs/experiments.md ---")
    print("| Датасет (n img) | Профіль | Час тренування (с) | GPU |")
    print("|-----------------|---------|--------------------|-----|")
    print("| -               | -       | -                  | -   |")
    print(f"\nGeneration avg: ~{avg:.0f} s/image")


if __name__ == "__main__":
    main()
