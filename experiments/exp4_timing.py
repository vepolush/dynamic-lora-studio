"""
Exp 4: Generation time / Training time.

Run: python -m experiments.exp4_timing [path/to/images.zip]
Requires: backend running, GPU.

Measures:
- Generation time: N images without LoRA + N with LoRA
- Training time: all 3 profiles (fast/balanced/strong) if ZIP provided
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
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
from experiments.plot_utils import save_bar_chart, save_multi_line_chart


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp4: Timing")
    parser.add_argument("zip_path", nargs="?", type=Path, help="ZIP for training time measurement")
    parser.add_argument("--n-gen", type=int, default=5, help="Number of images to generate per mode")
    parser.add_argument("--trigger", default="<test>", help="Trigger word for training")
    args = parser.parse_args()

    client = get_client()
    token = ensure_auth(client)
    client.token = token

    session = client.create_session("Exp4 Timing")
    session_id = session["id"]

    prompt_no = load_prompt_no_lora()
    avg_lora = 0.0
    times_lora: list[float] = []

    # --- Generation time WITHOUT LoRA ---
    print(f"{'='*60}")
    print(f"--- Generation time WITHOUT LoRA ({args.n_gen} images, 512x512) ---")
    print(f"{'='*60}")
    times_no_lora: list[float] = []
    for i in range(args.n_gen):
        t0 = time.time()
        r = client.generate(session_id, prompt_no, seed=42 + i)
        elapsed = time.time() - t0
        times_no_lora.append(elapsed)
        api_time = r.get("generation_time", 0)
        print(f"  Image {i+1}: {elapsed:.1f}s (API: {api_time:.1f}s)")
        save_images_from_response(r, session_id, subdir=f"exp4/no_lora/img{i+1}")
    avg_no = sum(times_no_lora) / len(times_no_lora) if times_no_lora else 0
    med_no = statistics.median(times_no_lora) if times_no_lora else 0.0
    std_no = statistics.stdev(times_no_lora) if len(times_no_lora) > 1 else 0.0
    print(f"  Average (no LoRA): {avg_no:.1f}s")

    # --- Training time (all 3 profiles) ---
    profiles = ["fast", "balanced", "strong"]
    train_results: list[dict] = []

    if args.zip_path and args.zip_path.exists():
        print(f"\n{'='*60}")
        print("--- Training time (3 profiles) ---")
        print(f"{'='*60}")
        for profile in profiles:
            entity = client.upload_entity(
                name=f"Exp4 {profile}",
                trigger_word=args.trigger,
                zip_path=args.zip_path,
                training_profile=profile,
            )
            eid = entity["id"]
            ok, train_time = wait_for_entity_ready(client, eid)
            n_imgs = (client.get_entity(eid) or {}).get("image_count", "?")
            status = "ready" if ok else "FAILED"
            print(f"  {profile}: {status} — {train_time:.1f}s ({n_imgs} images)")
            train_results.append({"profile": profile, "time": train_time, "n_imgs": n_imgs, "entity_id": eid, "ok": ok})

        # --- Generation time WITH LoRA (use balanced entity) ---
        balanced_eid = None
        for tr in train_results:
            if tr["profile"] == "balanced" and tr["ok"]:
                balanced_eid = tr["entity_id"]
                break

        if balanced_eid:
            prompt_lora = load_prompt_lora(args.trigger)
            print(f"\n{'='*60}")
            print(f"--- Generation time WITH LoRA ({args.n_gen} images) ---")
            print(f"{'='*60}")
            for i in range(args.n_gen):
                t0 = time.time()
                r = client.generate(session_id, prompt_lora, entity_id=balanced_eid, seed=42 + i)
                elapsed = time.time() - t0
                times_lora.append(elapsed)
                api_time = r.get("generation_time", 0)
                print(f"  Image {i+1}: {elapsed:.1f}s (API: {api_time:.1f}s)")
                save_images_from_response(r, session_id, subdir=f"exp4/with_lora/img{i+1}")
            avg_lora = sum(times_lora) / len(times_lora) if times_lora else 0
            med_lora = statistics.median(times_lora) if times_lora else 0.0
            std_lora = statistics.stdev(times_lora) if len(times_lora) > 1 else 0.0
            print(f"  Average (with LoRA): {avg_lora:.1f}s")

    plot_dir = OUTPUT_DIR / "plots" / "exp4"
    x_values = list(range(1, len(times_no_lora) + 1))
    if times_no_lora:
        series = [("No LoRA", times_no_lora, "#4C78A8")]
        if times_lora:
            series.append(("With LoRA", times_lora, "#F58518"))
        save_multi_line_chart(
            output_path=plot_dir / "generation_time_comparison.png",
            title="Exp4: Generation Time per Image",
            x_values=x_values,
            series=series,
            x_label="Image index",
            y_label="Total generation time (s)",
        )
        save_bar_chart(
            output_path=plot_dir / "generation_avg_comparison.png",
            title="Exp4: Average Generation Time",
            labels=["No LoRA", "With LoRA"],
            values=[avg_no, avg_lora],
            x_label="Mode",
            y_label="Average generation time (s)",
            color="#54A24B",
        )
    if train_results:
        save_bar_chart(
            output_path=plot_dir / "training_time_profiles.png",
            title="Exp4: Training Time by Profile",
            labels=[str(r["profile"]) for r in train_results],
            values=[float(r.get("time", 0.0)) for r in train_results],
            x_label="Training profile",
            y_label="Training time (s)",
            color="#E45756",
        )

    print(f"\n{'='*60}")
    print("--- Summary ---")
    print(f"{'='*60}")
    print(f"Generation (no LoRA):   avg ~{avg_no:.1f} s/image")
    print(f"                        median ~{med_no:.1f} s/image, std ~{std_no:.2f}")
    if times_lora:
        print(f"Generation (with LoRA): avg ~{avg_lora:.1f} s/image")
        print(f"                        median ~{med_lora:.1f} s/image, std ~{std_lora:.2f}")
    if train_results:
        for tr in train_results:
            print(f"Training ({tr['profile']}): {tr['time']:.1f}s ({tr['n_imgs']} imgs)")
    print(f"\nPlots saved to {plot_dir}")
    print(f"\nImages saved to experiments/output/exp4/")


if __name__ == "__main__":
    main()
