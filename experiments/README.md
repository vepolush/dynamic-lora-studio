# Experiments

Scripts for diploma experiments.

## Prompt files

Two separate prompt files (use `{trigger}` placeholder):

- **`prompt_lora.txt`** — short prompt for generation WITH LoRA. Trigger word does the heavy lifting.
  Example: `a photo of a cat {trigger}, sitting on a windowsill, soft natural light`
- **`prompt_no_lora.txt`** — detailed prompt for generation WITHOUT LoRA. Describes the subject explicitly.
  Example: `high-quality photo of a tabby European shorthair cat {trigger}, green eyes, ...`

Edit these before running. Subject type (cat, dog, etc.) — write it yourself in the prompt.

## Output

Generated images are saved to `experiments/output/` (subdirs per experiment). Each path is printed.
Statistical charts are saved to `experiments/output/plots/`.

## Prerequisites

- Backend running (`uvicorn main:app --host 0.0.0.0 --port 8000`)
- GPU (for generation and training)
- `pip install httpx matplotlib`

## Usage

Run from project root:

```bash
# Exp 1: LoRA vs no LoRA (two subtasks: practical + strict control)
python -m experiments.exp1_lora_vs_no_lora path/to/images.zip --trigger "<stefan_cat>"
python -m experiments.exp1_lora_vs_no_lora   # use existing entity

# Exp 2: Dataset size (ZIP with 30+ images, shows training time per entity)
python -m experiments.exp2_dataset_size path/to/full_dataset.zip --trigger "<stefan_cat>"
# Exp 2 + normalized steps-per-image base (1200 steps for 30 images)
python -m experiments.exp2_dataset_size path/to/full_dataset.zip --normalized-base-steps 1200 --normalized-base-size 30

# Exp 3: Hyperparameters — comprehensive grid (rank, LR, steps, scheduler, profiles)
python -m experiments.exp3_hyperparams path/to/images.zip --trigger "<stefan_cat>"
python -m experiments.exp3_hyperparams --groups Rank,LR   # run only specific groups
python -m experiments.exp3_hyperparams --timing-repeats 5

# Exp 4: Timing — generation + training (3 profiles: fast/balanced/strong)
python -m experiments.exp4_timing path/to/images.zip
python -m experiments.exp4_timing   # generation only

# Exp 5: LoRA switch speed (requires 2+ ready entities, 5 A↔B cycles)
python -m experiments.exp5_switch_speed
python -m experiments.exp5_switch_speed --cycles 8 --same-runs 8
```

## Environment

- `BACKEND_URL` — default `http://localhost:8000`
