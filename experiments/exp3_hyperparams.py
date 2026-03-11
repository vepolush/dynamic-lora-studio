"""
Exp 3: LoRA hyperparameters — rank, learning rate, steps, lr scheduler.

Run: python -m experiments.exp3_hyperparams [path/to/images.zip]
Requires: backend running, GPU.

Comprehensive grid: varies rank, LR, steps, and lr_scheduler.
Each config retrains the entity and generates images for comparison.
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
    save_images_from_response,
    wait_for_entity_ready,
)
from experiments.plot_utils import save_bar_chart

CONFIGS = [
    # --- 3a: Rank (LR=1e-4, steps=1200, polynomial) ---
    {"group": "Rank",     "label": "rank4",   "steps": 1200, "rank": 4,  "lr": 1e-4, "sched": "polynomial", "warmup": 0.06},
    {"group": "Rank",     "label": "rank8",   "steps": 1200, "rank": 8,  "lr": 1e-4, "sched": "polynomial", "warmup": 0.06},
    {"group": "Rank",     "label": "rank16",  "steps": 1200, "rank": 16, "lr": 1e-4, "sched": "polynomial", "warmup": 0.06},
    {"group": "Rank",     "label": "rank32",  "steps": 1200, "rank": 32, "lr": 1e-4, "sched": "polynomial", "warmup": 0.06},
    # --- 3b: Learning rate (rank=16, steps=1200, polynomial) ---
    {"group": "LR",       "label": "lr5e-5",  "steps": 1200, "rank": 16, "lr": 5e-5, "sched": "polynomial", "warmup": 0.06},
    {"group": "LR",       "label": "lr1e-4",  "steps": 1200, "rank": 16, "lr": 1e-4, "sched": "polynomial", "warmup": 0.06},
    {"group": "LR",       "label": "lr2e-4",  "steps": 1200, "rank": 16, "lr": 2e-4, "sched": "polynomial", "warmup": 0.06},
    {"group": "LR",       "label": "lr5e-4",  "steps": 1200, "rank": 16, "lr": 5e-4, "sched": "polynomial", "warmup": 0.06},
    # --- 3c: Steps (rank=16, LR=1e-4, polynomial) ---
    {"group": "Steps",    "label": "steps500", "steps": 500,  "rank": 16, "lr": 1e-4, "sched": "polynomial", "warmup": 0.06},
    {"group": "Steps",    "label": "steps700", "steps": 700,  "rank": 16, "lr": 1e-4, "sched": "polynomial", "warmup": 0.06},
    {"group": "Steps",    "label": "steps1200","steps": 1200, "rank": 16, "lr": 1e-4, "sched": "polynomial", "warmup": 0.06},
    {"group": "Steps",    "label": "steps1800","steps": 1800, "rank": 16, "lr": 1e-4, "sched": "polynomial", "warmup": 0.06},
    {"group": "Steps",    "label": "steps2500","steps": 2500, "rank": 16, "lr": 1e-4, "sched": "polynomial", "warmup": 0.06},
    # --- 3d: LR Scheduler (rank=16, LR=1e-4, steps=1200) ---
    {"group": "Scheduler","label": "constant", "steps": 1200, "rank": 16, "lr": 1e-4, "sched": "constant",   "warmup": 0.0},
    {"group": "Scheduler","label": "polynomial","steps":1200, "rank": 16, "lr": 1e-4, "sched": "polynomial", "warmup": 0.06},
    {"group": "Scheduler","label": "cosine",   "steps": 1200, "rank": 16, "lr": 1e-4, "sched": "cosine",     "warmup": 0.08},
    # --- 3e: Training profiles (presets) ---
    {"group": "Profile",  "label": "fast",     "steps": 700,  "rank": 8,  "lr": 1e-4, "sched": "constant",   "warmup": 0.03},
    {"group": "Profile",  "label": "balanced", "steps": 1200, "rank": 16, "lr": 1e-4, "sched": "polynomial", "warmup": 0.06},
    {"group": "Profile",  "label": "strong",   "steps": 1800, "rank": 16, "lr": 8e-5, "sched": "cosine",     "warmup": 0.08},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3: LoRA hyperparams (comprehensive)")
    parser.add_argument("zip_path", nargs="?", type=Path, help="ZIP with 10-15 images")
    parser.add_argument("--trigger", default="<my_subject>", help="Trigger word")
    parser.add_argument("--seeds", default="42,123", help="Seeds per config")
    parser.add_argument("--groups", default=None, help="Run only these groups (comma-separated: Rank,LR,Steps,...)")
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=3,
        help="How many repeated generation runs per seed (for mean/median/std)",
    )
    args = parser.parse_args()

    client = get_client()
    token = ensure_auth(client)
    client.token = token

    session = client.create_session("Exp3 Hyperparams")
    session_id = session["id"]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

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
        ok, t = wait_for_entity_ready(client, entity_id)
        if not ok:
            print("Training failed")
            sys.exit(1)
        print(f"Entity ready ({t:.1f}s)")
    else:
        entities = [e for e in client.get_entities() if e.get("status") == "ready"]
        if not entities:
            print("No ready entity. Provide ZIP.")
            sys.exit(1)
        entity_id = entities[0]["id"]
        args.trigger = entities[0].get("trigger_word", args.trigger)
        print(f"Using entity: {entities[0].get('name', '?')} ({entity_id})")

    prompt = load_prompt_lora(args.trigger)
    print(f"Prompt: {prompt[:80]}...")

    selected_groups = None
    if args.groups:
        selected_groups = {g.strip() for g in args.groups.split(",")}

    configs = CONFIGS
    if selected_groups:
        configs = [c for c in configs if c["group"] in selected_groups]
    print(f"\nRunning {len(configs)} configs...\n")

    results: list[dict] = []
    current_group = ""
    for i, cfg in enumerate(configs):
        if cfg["group"] != current_group:
            current_group = cfg["group"]
            print(f"\n{'='*60}")
            print(f"--- Group: {current_group} ---")
            print(f"{'='*60}")

        label = cfg["label"]
        print(f"\n  [{i+1}/{len(configs)}] {label}: rank={cfg['rank']} steps={cfg['steps']} lr={cfg['lr']:.0e} sched={cfg['sched']}")

        client.retrain_entity(
            entity_id,
            steps=cfg["steps"],
            rank=cfg["rank"],
            learning_rate=cfg["lr"],
            lr_scheduler=cfg["sched"],
            warmup_ratio=cfg["warmup"],
            use_custom=True,
        )
        ok, train_time = wait_for_entity_ready(client, entity_id)
        if not ok:
            print(f"    FAILED (after {train_time:.1f}s)")
            results.append({**cfg, "train_time": train_time, "status": "failed"})
            continue

        print(f"    Trained in {train_time:.1f}s")

        gen_times: list[float] = []
        for seed in seeds:
            repeat_times: list[float] = []
            for rep in range(args.timing_repeats):
                r = client.generate(session_id, prompt, entity_id=entity_id, seed=seed)
                gt = float(r.get("generation_time", 0))
                repeat_times.append(gt)
                gen_times.append(gt)
                save_images_from_response(
                    r,
                    session_id,
                    subdir=f"exp3/{current_group}/{label}/seed{seed}/rep{rep+1}",
                )
            mean_seed = statistics.mean(repeat_times) if repeat_times else 0.0
            med_seed = statistics.median(repeat_times) if repeat_times else 0.0
            std_seed = statistics.stdev(repeat_times) if len(repeat_times) > 1 else 0.0
            print(
                f"    Seed {seed}: mean={mean_seed:.2f}s median={med_seed:.2f}s std={std_seed:.2f}s "
                f"(n={len(repeat_times)})"
            )

        avg_gen = statistics.mean(gen_times) if gen_times else 0.0
        med_gen = statistics.median(gen_times) if gen_times else 0.0
        std_gen = statistics.stdev(gen_times) if len(gen_times) > 1 else 0.0
        results.append(
            {
                **cfg,
                "train_time": train_time,
                "avg_gen": avg_gen,
                "median_gen": med_gen,
                "std_gen": std_gen,
                "status": "ok",
            }
        )

    print(f"\n{'='*60}")
    print("--- Full results table ---")
    print(f"{'='*60}")
    print(
        f"| {'Group':<10} | {'Label':<12} | {'Rank':>4} | {'Steps':>5} | {'LR':>8} | "
        f"{'Sched':<12} | {'Train(s)':>9} | {'Gen mean':>8} | {'Gen med':>7} | {'Gen std':>7} | Score |"
    )
    print(f"|{'-'*12}|{'-'*14}|{'-'*6}|{'-'*7}|{'-'*10}|{'-'*14}|{'-'*11}|{'-'*10}|{'-'*9}|{'-'*9}|-------|")
    for r in results:
        st = r.get("status", "?")
        tt = r.get("train_time", 0)
        gt = r.get("avg_gen", 0)
        gm = r.get("median_gen", 0)
        gs = r.get("std_gen", 0)
        score = "" if st == "ok" else "FAIL"
        print(
            f"| {r['group']:<10} | {r['label']:<12} | {r['rank']:>4} | {r['steps']:>5} | "
            f"{r['lr']:>8.0e} | {r['sched']:<12} | {tt:>8.1f}s | {gt:>7.2f}s | {gm:>6.2f}s | {gs:>6.2f}s | {score:<5} |"
        )

    ok_results = [r for r in results if r.get("status") == "ok"]
    plot_dir = OUTPUT_DIR / "plots" / "exp3"
    if ok_results:
        save_bar_chart(
            output_path=plot_dir / "train_time_all_configs.png",
            title="Exp3: Training Time per Config",
            labels=[str(r["label"]) for r in ok_results],
            values=[float(r.get("train_time", 0.0)) for r in ok_results],
            x_label="Config",
            y_label="Training time (s)",
            color="#4C78A8",
            rotate_x=45,
        )
        save_bar_chart(
            output_path=plot_dir / "generation_time_all_configs.png",
            title="Exp3: Avg Generation Time per Config",
            labels=[str(r["label"]) for r in ok_results],
            values=[float(r.get("avg_gen", 0.0)) for r in ok_results],
            x_label="Config",
            y_label="Average generation time (s)",
            color="#F58518",
            rotate_x=45,
        )

        groups = sorted({str(r["group"]) for r in ok_results})
        for group in groups:
            g_rows = [r for r in ok_results if r["group"] == group]
            save_bar_chart(
                output_path=plot_dir / f"train_time_{group.lower()}.png",
                title=f"Exp3: Training Time ({group})",
                labels=[str(r["label"]) for r in g_rows],
                values=[float(r.get("train_time", 0.0)) for r in g_rows],
                x_label=f"{group} config",
                y_label="Training time (s)",
                color="#72B7B2",
                rotate_x=30,
            )
    print(f"\nPlots saved to {plot_dir}")
    print(f"\nImages saved to experiments/output/exp3/")


if __name__ == "__main__":
    main()
