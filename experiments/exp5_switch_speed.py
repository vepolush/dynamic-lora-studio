"""
Exp 5: Dynamic LoRA switching — how fast does the system react?

Run: python -m experiments.exp5_switch_speed
Requires: backend running, 2+ ready entities.

Measures: 
- time when switching between entities over multiple cycles.
- no-LoRA → LoRA transition and LoRA → no-LoRA.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from experiments.utils import (
    OUTPUT_DIR,
    ensure_auth,
    get_client,
    load_prompt_lora,
    load_prompt_no_lora,
    save_images_from_response,
)
from experiments.plot_utils import save_bar_chart, save_line_chart


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp5: LoRA switch speed")
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Number of A↔B cycles (2*cycles generations in switch phase)",
    )
    parser.add_argument(
        "--same-runs",
        type=int,
        default=5,
        help="Number of same-entity repeated runs for stable baseline",
    )
    args = parser.parse_args()

    client = get_client()
    token = ensure_auth(client)
    client.token = token

    entities = [e for e in client.get_entities() if e.get("status") == "ready"]
    if len(entities) < 2:
        print("Need at least 2 ready entities. Create them in the UI first.")
        sys.exit(1)

    session = client.create_session("Exp5 Switch")
    session_id = session["id"]

    e1, e2 = entities[0], entities[1]
    trigger1 = e1.get("trigger_word", "<a>")
    trigger2 = e2.get("trigger_word", "<b>")
    prompt1 = load_prompt_lora(trigger1)
    prompt2 = load_prompt_lora(trigger2)
    prompt_no = load_prompt_no_lora()

    print(f"Entity A: {e1['name']} ({e1['id']}, trigger={trigger1})")
    print(f"Entity B: {e2['name']} ({e2['id']}, trigger={trigger2})")

    # --- Phase 1: no-LoRA baseline ---
    print(f"\n{'='*60}")
    print("--- Phase 1: No LoRA baseline ---")
    print(f"{'='*60}")
    t0 = time.time()
    r = client.generate(session_id, prompt_no, seed=1)
    t_baseline = time.time() - t0
    print(f"  No LoRA: {t_baseline:.1f}s (gen: {r.get('generation_time', 0):.1f}s)")
    save_images_from_response(r, session_id, subdir="exp5/baseline")

    # --- Phase 2: no-LoRA → LoRA A (first load) ---
    print(f"\n{'='*60}")
    print("--- Phase 2: No LoRA → Entity A (first LoRA load) ---")
    print(f"{'='*60}")
    t0 = time.time()
    r = client.generate(session_id, prompt1, entity_id=e1["id"], seed=10)
    t_first_load = time.time() - t0
    print(f"  A (first load): {t_first_load:.1f}s (gen: {r.get('generation_time', 0):.1f}s)")
    save_images_from_response(r, session_id, subdir="exp5/first_load_A")

    # --- Phase 2.5: Same-entity repeated runs (for stable baseline) ---
    print(f"\n{'='*60}")
    print(f"--- Phase 2.5: Same-entity repeated runs (n={args.same_runs}) ---")
    print(f"{'='*60}")
    same_times: list[float] = []
    for i in range(args.same_runs):
        t0 = time.time()
        r_same = client.generate(session_id, prompt1, entity_id=e1["id"], seed=30 + i)
        elapsed_same = time.time() - t0
        same_times.append(elapsed_same)
        print(
            f"  A same #{i+1}: {elapsed_same:.2f}s (gen: {float(r_same.get('generation_time', 0)):.2f}s)"
        )
        save_images_from_response(r_same, session_id, subdir=f"exp5/same_entity_A/{i+1}")

    # --- Phase 3: Multiple A↔B switches ---
    print(f"\n{'='*60}")
    print(f"--- Phase 3: Switching A ↔ B ({args.cycles} cycles) ---")
    print(f"{'='*60}")
    switch_times: list[dict] = []
    sequence: list[tuple[str, str, str]] = []
    for _ in range(args.cycles):
        sequence.append(("A", e1["id"], prompt1))
        sequence.append(("B", e2["id"], prompt2))
    for i, (label, eid, prompt) in enumerate(sequence):
        t0 = time.time()
        r = client.generate(session_id, prompt, entity_id=eid, seed=100 + i)
        elapsed = time.time() - t0
        gen_t = r.get("generation_time", 0)
        is_switch = i > 0 and sequence[i - 1][1] != eid
        tag = " [SWITCH]" if is_switch else ""
        print(f"  {i+1}. {label}{tag}: {elapsed:.1f}s (gen: {gen_t:.1f}s)")
        save_images_from_response(r, session_id, subdir=f"exp5/cycle/{i+1}_{label}")
        switch_times.append({"step": i + 1, "entity": label, "is_switch": is_switch, "total": elapsed, "gen": gen_t})

    # --- Phase 4: LoRA → no-LoRA (unload) ---
    print(f"\n{'='*60}")
    print("--- Phase 4: LoRA → No LoRA (unload) ---")
    print(f"{'='*60}")
    t0 = time.time()
    r = client.generate(session_id, prompt_no, seed=200)
    t_unload = time.time() - t0
    print(f"  No LoRA after LoRA: {t_unload:.1f}s (gen: {r.get('generation_time', 0):.1f}s)")
    save_images_from_response(r, session_id, subdir="exp5/unload")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("--- Summary ---")
    print(f"{'='*60}")
    switch_only = [s["total"] for s in switch_times if s["is_switch"]]
    avg_same = sum(same_times) / len(same_times) if same_times else 0
    avg_switch = sum(switch_only) / len(switch_only) if switch_only else 0
    med_same = statistics.median(same_times) if same_times else 0.0
    med_switch = statistics.median(switch_only) if switch_only else 0.0
    std_same = statistics.stdev(same_times) if len(same_times) > 1 else 0.0
    std_switch = statistics.stdev(switch_only) if len(switch_only) > 1 else 0.0

    print(f"  No LoRA baseline:       {t_baseline:.1f}s")
    print(f"  First LoRA load:        {t_first_load:.1f}s")
    print(
        f"  Same entity:            mean={avg_same:.2f}s median={med_same:.2f}s std={std_same:.2f}s "
        f"(n={len(same_times)})"
    )
    print(
        f"  Switch entity:          mean={avg_switch:.2f}s median={med_switch:.2f}s std={std_switch:.2f}s "
        f"(n={len(switch_only)})"
    )
    print(f"  Switch overhead:        ~{max(0, avg_switch - avg_same):.1f}s")
    print(f"  LoRA → No LoRA:         {t_unload:.1f}s")
    plot_dir = OUTPUT_DIR / "plots" / "exp5"
    save_line_chart(
        output_path=plot_dir / "switch_sequence_time.png",
        title="Exp5: A/B Switching Sequence Time",
        x_values=[float(s["step"]) for s in switch_times],
        y_values=[float(s["total"]) for s in switch_times],
        x_label="Sequence step",
        y_label="Total generation time (s)",
        color="#4C78A8",
    )
    save_bar_chart(
        output_path=plot_dir / "switch_summary.png",
        title="Exp5: Switching Summary",
        labels=["Baseline", "First load", "Same avg", "Switch avg", "Unload"],
        values=[t_baseline, t_first_load, avg_same, avg_switch, t_unload],
        x_label="Metric",
        y_label="Time (s)",
        color="#F58518",
        rotate_x=20,
    )
    print(f"\nPlots saved to {plot_dir}")
    print(f"\nImages saved to experiments/output/exp5/")


if __name__ == "__main__":
    main()
