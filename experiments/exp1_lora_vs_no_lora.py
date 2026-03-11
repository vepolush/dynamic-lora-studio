"""
Exp 1: Without LoRA vs With LoRA — two-subtask comparison.

Run: python -m experiments.exp1_lora_vs_no_lora [path/to/images.zip]
Requires: backend running, GPU.

Subtask A (practical): different identity prompts
- no-LoRA: detailed identity description (prompt_no_lora.txt)
- with-LoRA: short trigger-based prompt (prompt_lora.txt)

Subtask B (strict control): same prompt text in both modes
- no-LoRA: detailed prompt_no_lora.txt
- with-LoRA: same prompt_no_lora.txt + entity_id
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.utils import (
    OUTPUT_DIR,
    ensure_auth,
    get_client,
    load_prompt_lora,
    load_prompt_no_lora,
    save_images_from_response,
    wait_for_entity_ready,
)
from experiments.plot_utils import save_multi_line_chart


def _run_generation_block(
    client,
    *,
    session_id: str,
    seeds: list[int],
    prompt_no_lora: str,
    prompt_with_lora: str,
    entity_id: str,
    out_dir_no_lora: str,
    out_dir_with_lora: str,
) -> tuple[list[float], list[float]]:
    no_lora_times: list[float] = []
    with_lora_times: list[float] = []

    print(f"\n{'='*60}")
    print(f"--- Generation WITHOUT LoRA ({len(seeds)} seeds) ---")
    print(f"{'='*60}")
    for seed in seeds:
        r = client.generate(session_id, prompt_no_lora, seed=seed)
        t = float(r.get("generation_time", 0))
        no_lora_times.append(t)
        print(f"  Seed {seed}: {t:.1f}s")
        save_images_from_response(r, session_id, subdir=f"{out_dir_no_lora}/seed{seed}")

    print(f"\n{'='*60}")
    print(f"--- Generation WITH LoRA ({len(seeds)} seeds) ---")
    print(f"{'='*60}")
    for seed in seeds:
        r = client.generate(session_id, prompt_with_lora, entity_id=entity_id, seed=seed)
        t = float(r.get("generation_time", 0))
        with_lora_times.append(t)
        print(f"  Seed {seed}: {t:.1f}s")
        save_images_from_response(r, session_id, subdir=f"{out_dir_with_lora}/seed{seed}")

    return no_lora_times, with_lora_times


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1: LoRA vs no LoRA")
    parser.add_argument("zip_path", nargs="?", type=Path, help="ZIP with 10-15 images")
    parser.add_argument("--seeds", default="42,123,456,789,1024", help="Comma-separated seeds")
    parser.add_argument("--trigger", default="<my_subject>", help="Trigger word")
    args = parser.parse_args()

    client = get_client()
    token = ensure_auth(client)
    client.token = token

    session = client.create_session("Exp1 LoRA vs No LoRA")
    session_id = session["id"]

    entity_id: str | None = None
    train_time = 0.0
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
        ok, train_time = wait_for_entity_ready(client, entity_id)
        if not ok:
            print("Training failed or timed out")
            sys.exit(1)
        print(f"Entity ready. Training time: {train_time:.1f}s")
    else:
        entities = client.get_entities()
        ready = [e for e in entities if e.get("status") == "ready"]
        if not ready:
            print("No ready entity. Provide ZIP: python -m experiments.exp1_lora_vs_no_lora path/to/images.zip")
            sys.exit(1)
        entity_id = ready[0]["id"]
        args.trigger = ready[0].get("trigger_word", args.trigger)
        print(f"Using entity: {ready[0].get('name', '?')} ({entity_id})")

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    prompt_no_detailed = load_prompt_no_lora()
    prompt_lora_short = load_prompt_lora(args.trigger)

    print(f"\nPrompt A (no LoRA, detailed): {prompt_no_detailed[:100]}...")
    print(f"Prompt A (with LoRA, short):  {prompt_lora_short[:100]}...")

    print(f"\n{'#'*70}")
    print("Subtask A: practical comparison (detailed no-LoRA vs short LoRA prompt)")
    print(f"{'#'*70}")
    a_no_times, a_with_times = _run_generation_block(
        client,
        session_id=session_id,
        seeds=seeds,
        prompt_no_lora=prompt_no_detailed,
        prompt_with_lora=prompt_lora_short,
        entity_id=entity_id,
        out_dir_no_lora="exp1/task_a/no_lora",
        out_dir_with_lora="exp1/task_a/with_lora",
    )

    print(f"\n{'#'*70}")
    print("Subtask B: strict control (same detailed prompt in both modes)")
    print(f"{'#'*70}")
    print(f"Shared prompt B: {prompt_no_detailed[:100]}...")
    b_no_times, b_with_times = _run_generation_block(
        client,
        session_id=session_id,
        seeds=seeds,
        prompt_no_lora=prompt_no_detailed,
        prompt_with_lora=prompt_no_detailed,
        entity_id=entity_id,
        out_dir_no_lora="exp1/task_b/no_lora",
        out_dir_with_lora="exp1/task_b/with_lora",
    )

    print(f"\n{'='*60}")
    print("--- Table templates for docs/experiments.md ---")
    print(f"{'='*60}")
    print("Subtask A:")
    print("| Seed | Без LoRA (1–5) | З LoRA (1–5) | Примітки |")
    print("|------|----------------|--------------|----------|")
    for seed in seeds:
        print(f"| {seed}   |                |              |          |")
    print("\nSubtask B:")
    print("| Seed | Без LoRA (1–5) | З LoRA (1–5) | Примітки |")
    print("|------|----------------|--------------|----------|")
    for seed in seeds:
        print(f"| {seed}   |                |              |          |")

    plot_dir = OUTPUT_DIR / "plots" / "exp1"
    save_multi_line_chart(
        output_path=plot_dir / "task_a_generation_time_by_seed.png",
        title="Exp1 Task A: Generation Time by Seed (No LoRA vs With LoRA)",
        x_values=[float(s) for s in seeds],
        series=[
            ("No LoRA", a_no_times, "#4C78A8"),
            ("With LoRA", a_with_times, "#F58518"),
        ],
        x_label="Seed",
        y_label="Generation time (s)",
    )
    save_multi_line_chart(
        output_path=plot_dir / "task_b_generation_time_by_seed.png",
        title="Exp1 Task B: Generation Time by Seed (No LoRA vs With LoRA)",
        x_values=[float(s) for s in seeds],
        series=[
            ("No LoRA", b_no_times, "#4C78A8"),
            ("With LoRA", b_with_times, "#F58518"),
        ],
        x_label="Seed",
        y_label="Generation time (s)",
    )
    print(f"\nPlots saved to {plot_dir}")
    print("\nImages saved to experiments/output/exp1/task_a and experiments/output/exp1/task_b")


if __name__ == "__main__":
    main()
