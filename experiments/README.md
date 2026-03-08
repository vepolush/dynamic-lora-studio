# Experiments

Scripts for diploma experiments.

## Prerequisites

- Backend running (`uvicorn main:app --host 0.0.0.0 --port 8000`)
- GPU (for generation and training)
- `pip install httpx` (or use project venv with backend deps)

## Usage

Run from project root:

```bash
# Exp 1: LoRA vs no LoRA
python -m experiments.exp1_lora_vs_no_lora path/to/images.zip
python -m experiments.exp1_lora_vs_no_lora   # use existing entity

# Exp 2: Dataset size (ZIP with 30+ images)
python -m experiments.exp2_dataset_size path/to/full_dataset.zip

# Exp 3: Hyperparameters (rank, LR, steps)
python -m experiments.exp3_hyperparams path/to/images.zip
python -m experiments.exp3_hyperparams   # use existing entity

# Exp 4: Timing
python -m experiments.exp4_timing
python -m experiments.exp4_timing path/to/images.zip   # + training time

# Exp 5: LoRA switch speed (requires 2+ ready entities)
python -m experiments.exp5_switch_speed
```

## Environment

- `BACKEND_URL` — default `http://localhost:8000`

## Output

Each script prints tables for documentation. Fill in subjective scores (1–5) manually after reviewing generated images.
