"""
Exp 2: How many images for quality LoRA?

Run: python -m experiments.exp2_dataset_size path/to/full_dataset.zip
Requires: backend running, GPU. ZIP should have 30+ images.

Creates entities with 5, 10, 20, 30 images.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.utils import (
    OUTPUT_DIR,
    ensure_auth,
    get_client,
    load_prompt_lora,
    save_images_from_response,
    wait_for_entity_ready,
)
from experiments.plot_utils import save_bar_chart


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
    parser.add_argument("--seeds", default="42,123,456", help="Comma-separated seeds")
    parser.add_argument(
        "--normalized-base-steps",
        type=int,
        default=1200,
        help="Base steps used to compute normalized steps-per-image",
    )
    parser.add_argument(
        "--normalized-base-size",
        type=int,
        default=30,
        help="Base dataset size used to compute normalized steps-per-image",
    )
    parser.add_argument(
        "--normalized-rank",
        type=int,
        default=16,
        help="Rank for normalized retraining",
    )
    parser.add_argument(
        "--normalized-lr",
        type=float,
        default=1e-4,
        help="Learning rate for normalized retraining",
    )
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
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    entities: list[dict] = []
    gen_avg_fixed_by_n: dict[int, float] = {}
    train_fixed_by_n: dict[int, float] = {}
    train_norm_by_n: dict[int, float] = {}
    gen_avg_norm_by_n: dict[int, float] = {}

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

        print("\nWaiting for all entities to train...")
        for item in entities:
            ok, elapsed = wait_for_entity_ready(client, item["entity_id"])
            item["train_time"] = elapsed
            train_fixed_by_n[item["n"]] = float(elapsed)
            status = "ready" if ok else "FAILED"
            print(f"  n={item['n']}: {status} (training: {elapsed:.1f}s)")

        prompt = load_prompt_lora(args.trigger)
        print(f"\nPrompt: {prompt[:80]}...")
        print(f"\n{'='*60}")
        print(f"--- Generate {len(seeds)} images per entity ---")
        print(f"{'='*60}")
        for item in entities:
            eid = item["entity_id"]
            gen_times: list[float] = []
            for i, seed in enumerate(seeds):
                r = client.generate(session_id, prompt, entity_id=eid, seed=seed)
                t = r.get("generation_time", 0)
                gen_times.append(float(t))
                print(f"  n={item['n']} seed={seed}: {t:.1f}s")
                save_images_from_response(r, session_id, subdir=f"exp2/n{item['n']}/seed{seed}")
            gen_avg_fixed_by_n[item["n"]] = sum(gen_times) / len(gen_times) if gen_times else 0.0

        # ------------------------------------------------------------------
        # Additional phase: normalized steps-per-image
        # ------------------------------------------------------------------
        steps_per_image = args.normalized_base_steps / float(args.normalized_base_size)
        print(f"\n{'='*60}")
        print("--- Additional phase: normalized steps-per-image ---")
        print(f"{'='*60}")
        print(
            f"Base: {args.normalized_base_steps} steps @ {args.normalized_base_size} images "
            f"-> {steps_per_image:.2f} steps/image"
        )
        for item in entities:
            n = int(item["n"])
            eid = item["entity_id"]
            norm_steps = max(100, int(round(steps_per_image * n)))
            print(f"\n  n={n}: retrain with normalized steps={norm_steps}")
            client.retrain_entity(
                eid,
                steps=norm_steps,
                rank=args.normalized_rank,
                learning_rate=args.normalized_lr,
                use_custom=True,
            )
            ok, elapsed = wait_for_entity_ready(client, eid)
            train_norm_by_n[n] = float(elapsed)
            status = "ready" if ok else "FAILED"
            print(f"    {status} (training: {elapsed:.1f}s)")
            if not ok:
                continue

            norm_gen_times: list[float] = []
            for seed in seeds:
                r = client.generate(session_id, prompt, entity_id=eid, seed=seed)
                t = float(r.get("generation_time", 0))
                norm_gen_times.append(t)
                print(f"    seed={seed}: {t:.1f}s")
                save_images_from_response(
                    r,
                    session_id,
                    subdir=f"exp2_normalized/n{n}/seed{seed}",
                )
            gen_avg_norm_by_n[n] = (
                sum(norm_gen_times) / len(norm_gen_times) if norm_gen_times else 0.0
            )

    print(f"\n{'='*60}")
    print("--- Table for docs/experiments.md ---")
    print(f"{'='*60}")
    print("| Кількість | Train fixed, c | Gen fixed avg, c | Train normalized, c | Gen normalized avg, c |")
    print("|-----------|----------------|------------------|---------------------|-----------------------|")
    for item in entities:
        n = int(item["n"])
        tf = train_fixed_by_n.get(n, 0.0)
        gf = gen_avg_fixed_by_n.get(n, 0.0)
        tn = train_norm_by_n.get(n, 0.0)
        gn = gen_avg_norm_by_n.get(n, 0.0)
        print(f"| {n:<9} | {tf:>10.1f}      | {gf:>14.2f}    | {tn:>15.1f}   | {gn:>17.2f}     |")

    fixed_train_values = [train_fixed_by_n.get(n, 0.0) for n in sizes]
    norm_train_values = [train_norm_by_n.get(n, 0.0) for n in sizes]
    fixed_gen_values = [gen_avg_fixed_by_n.get(n, 0.0) for n in sizes]
    norm_gen_values = [gen_avg_norm_by_n.get(n, 0.0) for n in sizes]

    def _stats(vals: list[float]) -> tuple[float, float, float]:
        clean = [v for v in vals if v > 0]
        if not clean:
            return 0.0, 0.0, 0.0
        mean_v = statistics.mean(clean)
        med_v = statistics.median(clean)
        std_v = statistics.stdev(clean) if len(clean) > 1 else 0.0
        return mean_v, med_v, std_v

    tfix_mean, tfix_med, tfix_std = _stats(fixed_train_values)
    tnorm_mean, tnorm_med, tnorm_std = _stats(norm_train_values)
    gfix_mean, gfix_med, gfix_std = _stats(fixed_gen_values)
    gnorm_mean, gnorm_med, gnorm_std = _stats(norm_gen_values)

    print("\nTiming stats (mean / median / std):")
    print(
        f"  Train fixed:      {tfix_mean:.1f} / {tfix_med:.1f} / {tfix_std:.1f} s"
    )
    print(
        f"  Train normalized: {tnorm_mean:.1f} / {tnorm_med:.1f} / {tnorm_std:.1f} s"
    )
    print(
        f"  Gen fixed avg:    {gfix_mean:.2f} / {gfix_med:.2f} / {gfix_std:.2f} s"
    )
    print(
        f"  Gen normalized:   {gnorm_mean:.2f} / {gnorm_med:.2f} / {gnorm_std:.2f} s"
    )

    plot_dir = OUTPUT_DIR / "plots" / "exp2"
    save_bar_chart(
        output_path=plot_dir / "training_time_vs_dataset_size.png",
        title="Exp2: Training Time vs Dataset Size (fixed profile)",
        labels=[str(n) for n in sizes],
        values=fixed_train_values,
        x_label="Number of images in dataset",
        y_label="Training time (s)",
        color="#4C78A8",
    )
    save_bar_chart(
        output_path=plot_dir / "training_time_vs_dataset_size_normalized.png",
        title="Exp2: Training Time vs Dataset Size (normalized steps/image)",
        labels=[str(n) for n in sizes],
        values=norm_train_values,
        x_label="Number of images in dataset",
        y_label="Training time (s)",
        color="#72B7B2",
    )
    save_bar_chart(
        output_path=plot_dir / "generation_time_vs_dataset_size.png",
        title="Exp2: Avg Generation Time vs Dataset Size (fixed profile)",
        labels=[str(n) for n in sizes],
        values=fixed_gen_values,
        x_label="Number of images in dataset",
        y_label="Average generation time (s)",
        color="#F58518",
    )
    save_bar_chart(
        output_path=plot_dir / "generation_time_vs_dataset_size_normalized.png",
        title="Exp2: Avg Generation Time vs Dataset Size (normalized steps/image)",
        labels=[str(n) for n in sizes],
        values=norm_gen_values,
        x_label="Number of images in dataset",
        y_label="Average generation time (s)",
        color="#54A24B",
    )
    print(f"\nPlots saved to {plot_dir}")
    print(f"\nImages saved to experiments/output/exp2/")


if __name__ == "__main__":
    main()
