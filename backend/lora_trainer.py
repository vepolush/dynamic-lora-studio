"""LoRA trainer used by the background worker queue."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers

from entity_store import DATA_DIR
from model_manager import MODEL_DIR, MODEL_ID

IMAGE_SIZE = 512
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR_SCHEDULER = "polynomial"
DEFAULT_WARMUP_RATIO = 0.06
DEFAULT_MAX_GRAD_NORM = 1.0


class EntityDataset(Dataset):
    """Dataset for preprocessed entity images and trigger prompt."""

    def __init__(self, image_dir: Path, tokenizer: CLIPTokenizer, trigger_word: str) -> None:
        self.image_paths = sorted([p for p in image_dir.glob("*.png") if p.is_file()])
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    IMAGE_SIZE,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        prompt = f"a photo of {trigger_word}"
        self.input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "input_ids": self.input_ids,
        }


def _version_sort_key(name: str) -> int:
    match = re.match(r"v(\d+)", name.lower())
    return int(match.group(1)) if match else 0


def _next_version_name(weights_dir: Path, rank: int, steps: int) -> str:
    existing = [p.name for p in weights_dir.iterdir() if p.is_dir()]
    if not existing:
        next_index = 1
    else:
        next_index = max(_version_sort_key(name) for name in existing) + 1
    return f"v{next_index}_rank{rank}_steps{steps}"


def train_lora_for_entity(
    *,
    entity_id: str,
    trigger_word: str,
    steps: int = 1200,
    rank: int = 16,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    lr_scheduler: str = DEFAULT_LR_SCHEDULER,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
) -> dict[str, Any]:
    """Run LoRA fine-tuning and persist versioned weights."""
    entity_dir = DATA_DIR / "storage" / "entities" / entity_id
    dataset_dir = entity_dir / "dataset"
    weights_dir = entity_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    dataset_images = [p for p in dataset_dir.glob("*.png") if p.is_file()]
    if not dataset_images:
        raise ValueError("Dataset is empty; no preprocessed images found")

    version_name = _next_version_name(weights_dir, rank=rank, steps=steps)
    version_dir = weights_dir / version_name
    version_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_source = MODEL_DIR if Path(MODEL_DIR).exists() else MODEL_ID

    tokenizer = CLIPTokenizer.from_pretrained(model_source, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_source, subfolder="text_encoder").to(
        device=device,
        dtype=weight_dtype,
    )
    vae = AutoencoderKL.from_pretrained(model_source, subfolder="vae").to(
        device=device,
        dtype=weight_dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(model_source, subfolder="unet").to(
        device=device,
        dtype=weight_dtype,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(model_source, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    if weight_dtype == torch.float16:
        cast_training_params(unet, dtype=torch.float32)

    dataset = EntityDataset(dataset_dir, tokenizer, trigger_word)
    dataloader = DataLoader(dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    warmup_steps = max(1, int(steps * max(0.0, min(warmup_ratio, 0.25))))

    started_at = time.time()
    global_step = 0
    unet.train()
    while global_step < steps:
        for batch in dataloader:
            if global_step >= steps:
                break
            pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device=device)
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(input_ids)[0]
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_grad_norm,
                )
            lr_scale = _compute_lr_scale(
                step=global_step + 1,
                total_steps=steps,
                warmup_steps=warmup_steps,
                scheduler=lr_scheduler,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate * lr_scale
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

    unet_lora_state_dict = get_peft_model_state_dict(unet)
    unet_lora_state_dict = convert_state_dict_to_diffusers(unet_lora_state_dict)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(version_dir),
        unet_lora_layers=unet_lora_state_dict,
    )
    weights_path = version_dir / "pytorch_lora_weights.safetensors"

    duration_s = time.time() - started_at
    config = {
        "entity_id": entity_id,
        "trigger_word": trigger_word,
        "steps": steps,
        "rank": rank,
        "dataset_image_count": len(dataset_images),
        "weights_file": weights_path.name,
        "trainer": "native_diffusers_lora",
        "training_time_seconds": round(duration_s, 3),
        "learning_rate": learning_rate,
        "lr_scheduler": lr_scheduler,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "max_grad_norm": max_grad_norm,
        "batch_size": DEFAULT_BATCH_SIZE,
        "device": str(device),
    }
    with open(version_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return {
        "version": version_name,
        "version_dir": str(version_dir),
        "weights_path": str(weights_path),
        "training_time_seconds": round(duration_s, 3),
    }


def _compute_lr_scale(*, step: int, total_steps: int, warmup_steps: int, scheduler: str) -> float:
    if step <= warmup_steps:
        return max(1e-3, float(step) / float(max(1, warmup_steps)))

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    name = scheduler.lower().strip()

    if name == "constant":
        return 1.0
    if name == "cosine":
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))
    if name == "polynomial":
        return max(1e-3, (1.0 - progress) ** 1.0)
    return 1.0
