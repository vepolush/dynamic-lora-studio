import os
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download

MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_DIR = "/workspace/models/sd-1-5"

class ModelManager:
    def __init__(self):
        self.pipe = None

    def load_model(self):
        """Перевіряє наявність моделі, завантажує за потреби та ініціалізує пайплайн."""
        print("⏳ Перевірка базової моделі...")
        
        if not os.path.exists(os.path.join(MODEL_DIR, "model_index.json")):
            print(f"📥 Модель не знайдена локально. Завантаження {MODEL_ID}...")
            snapshot_download(repo_id=MODEL_ID, local_dir=MODEL_DIR)
            print("✅ Завантаження завершено!")
        else:
            print("✅ Модель знайдена локально.")

        print("🚀 Завантаження моделі у VRAM (GPU)...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        self.pipe.to("cuda") # Переносимо на відеокарту
        print("✅ Модель готова до генерації!")

    def generate(self, prompt: str, steps: int, guidance_scale: float):
        """Базова функція генерації (поки без LoRA)"""
        if self.pipe is None:
            raise RuntimeError("Модель не завантажена!")
        
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        )
        return result.images[0]

ml_manager = ModelManager()