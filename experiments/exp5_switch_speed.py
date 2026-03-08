"""
Exp 5: Dynamic LoRA switching — how fast does the system react?

Run: python -m experiments.exp5_switch_speed
Requires: backend running, 2+ ready entities.

Measures: time from generate request to response when switching between entities.
Output: delay in seconds.
"""

from __future__ import annotations

import sys
import time

from experiments.utils import ensure_auth, get_client


def main() -> None:
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

    print(f"Entity A: {e1['name']} ({e1['id']})")
    print(f"Entity B: {e2['name']} ({e2['id']})")
    print("\n--- Switching A -> B -> A ---")

    # A
    t0 = time.time()
    r1 = client.generate(session_id, f"{trigger1} in a garden", entity_id=e1["id"], seed=1)
    t1 = time.time() - t0
    print(f"  A: {t1:.1f}s (gen_time: {r1.get('generation_time', 0):.1f}s)")

    # B (switch)
    t0 = time.time()
    r2 = client.generate(session_id, f"{trigger2} in a garden", entity_id=e2["id"], seed=2)
    t2 = time.time() - t0
    print(f"  B (switch): {t2:.1f}s (gen_time: {r2.get('generation_time', 0):.1f}s)")

    # A again
    t0 = time.time()
    r3 = client.generate(session_id, f"{trigger1} in a garden", entity_id=e1["id"], seed=3)
    t3 = time.time() - t0
    print(f"  A again: {t3:.1f}s (gen_time: {r3.get('generation_time', 0):.1f}s)")

    # Estimate switch overhead: compare B (after switch) vs A (same entity twice)
    overhead = t2 - (t1 + t3) / 2 if (t1 + t3) > 0 else 0
    print(f"\n--- Estimated switch overhead: ~{max(0, overhead):.1f}s ---")
    print("(Difference between first request to new entity vs same entity)")


if __name__ == "__main__":
    main()
