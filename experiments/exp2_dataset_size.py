"""
Exp 2: How many images for quality LoRA?

Run: python -m experiments.exp2_dataset_size path/to/full_dataset.zip
Requires: backend running, GPU. ZIP should have 30+ images.

Creates 4 entities with 5, 10, 20, 30 images. Output: table for docs/experiments.md.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.utils import ensure_auth, get_client, wait_for_entity_ready


def make_subset_zip(source_zip: Path, n: int, out_path: Path) -> None:
    """Extract first n images from source ZIP into new ZIP at out_path."""
    img_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    collected: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(source_zip, "r") as zf:
        for name in sorted(zf.namelist()):
            if Path(name).suffix.lower() not in img_ext:
                continue
            if ".." in name or name.startswith("/"):
                continue
            try:
                data = zf.read(name)
                collected.append((Path(name).name, data))
                if len(collected) >= n:
                    break
            except Exception:
                continue
    if len(collected) < n:
        raise ValueError(f"Only {len(collected)} images in ZIP, need {n}")
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in collected:
            zf.writestr(fname, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp2: Dataset size")
    parser.add_argument("zip_path", type=Path, help="ZIP with 30+ images")
    parser.add_argument("--sizes", default="5,10,20,30", help="Comma-separated image counts")
    parser.add_argument("--trigger", default="<my_subject>", help="Trigger word")
    args = parser.parse_args()

    if not args.zip_path.exists():
        print(f"Error: {args.zip_path} not found")
        sys.exit(1)

    client = get_client()
    token = ensure_auth(client)
    client.token = token

    session = client.create_session("Exp2 Dataset Size")
    session_id = session["id"]

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    max_n = max(sizes)
    entities: list[dict] = []

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for n in sizes:
            subset_zip = tmp_path / f"subset_{n}.zip"
            print(f"Creating subset of {n} images...")
            make_subset_zip(args.zip_path, n, subset_zip)
            entity = client.upload_entity(
                name=f"Exp2 n={n}",
                trigger_word=args.trigger,
                zip_path=subset_zip,
                training_profile="balanced",
            )
            entities.append({"n": n, "entity_id": entity["id"]})

        print("Waiting for all entities to train...")
        for item in entities:
            if wait_for_entity_ready(client, item["entity_id"]):
                print(f"  n={item['n']} ready")
            else:
                print(f"  n={item['n']} failed")

        print("\n--- Generate 3 images per entity ---")
        for item in entities:
            eid = item["entity_id"]
            prompt = f"{args.trigger} in a garden"
            for i in range(3):
                r = client.generate(session_id, prompt, entity_id=eid, seed=42 + i)
                t = r.get("generation_time", 0)
                print(f"  n={item['n']} img{i+1}: {t:.1f}s")

    print("\n--- Table for docs/experiments.md ---")
    print("| Кількість | Схожість (1–5) | Артефакти | Висновок |")
    print("|-----------|----------------|-----------|----------|")
    for n in sizes:
        print(f"| {n}         |                |           |          |")


if __name__ == "__main__":
    main()
