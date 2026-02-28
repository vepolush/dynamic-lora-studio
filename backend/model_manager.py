"""Model loading and image generation for SD 1.5."""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    PNDMScheduler,
)
from huggingface_hub import snapshot_download

MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_DIR = "/workspace/models/sd-1-5"

SCHEDULERS: dict[str, type] = {
    "dpm++_2m_karras": DPMSolverMultistepScheduler,
    "dpm++_2m_sde_karras": DPMSolverMultistepScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "ddim": DDIMScheduler,
    "pndm": PNDMScheduler,
}


class ModelManager:
    def __init__(self) -> None:
        self.pipe: StableDiffusionPipeline | None = None

    def load_model(self) -> None:
        print("Checking base model...")
        if not os.path.exists(os.path.join(MODEL_DIR, "model_index.json")):
            print(f"Downloading {MODEL_ID}...")
            snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR)
            print("Download complete.")
        else:
            print("Model found locally.")

        print("Loading model to GPU...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()
        print("Model ready.")

    def _set_scheduler(self, name: str) -> None:
        if self.pipe is None:
            return
        sched_cls = SCHEDULERS.get(name, DPMSolverMultistepScheduler)
        kwargs: dict[str, Any] = {}
        if "karras" in name:
            kwargs["use_karras_sigmas"] = True
        if "sde" in name and sched_cls is DPMSolverMultistepScheduler:
            kwargs["algorithm_type"] = "sde-dpmsolver++"
        self.pipe.scheduler = sched_cls.from_config(self.pipe.scheduler.config, **kwargs)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        *,
        steps: int = 25,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
        num_images: int = 1,
        scheduler: str = "dpm++_2m_karras",
    ) -> list[dict[str, Any]]:
        """Generate images and return list of {base64, seed} dicts."""
        if self.pipe is None:
            raise RuntimeError("Model not loaded")

        self._set_scheduler(scheduler)

        generator = None
        seeds: list[int] = []
        if seed is not None and seed >= 0:
            seeds = [seed + i for i in range(num_images)]
            generator = [torch.Generator("cuda").manual_seed(s) for s in seeds]
        else:
            import random
            seeds = [random.randint(0, 2**32 - 1) for _ in range(num_images)]
            generator = [torch.Generator("cuda").manual_seed(s) for s in seeds]

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_images_per_prompt=num_images,
            generator=generator,
        )

        images_out: list[dict[str, Any]] = []
        for i, img in enumerate(result.images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            images_out.append({
                "base64": b64,
                "seed": seeds[i] if i < len(seeds) else -1,
                "png_bytes": buf.getvalue(),
            })
        return images_out


ml_manager = ModelManager()
