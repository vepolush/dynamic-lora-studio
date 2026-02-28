"""SD 1.5 prompt engineering — style, lighting, color, quality mappings."""

from __future__ import annotations

STYLE_PROMPTS: dict[str, str] = {
    "Realistic": "realistic, photorealistic, detailed, 8k uhd, high quality",
    "Photographic": "professional photography, DSLR, 50mm lens, f/1.8, bokeh, sharp focus, film grain",
    "3D Render": "3D render, octane render, unreal engine 5, highly detailed, volumetric",
    "Anime": "anime style, cel shading, anime key visual, vibrant, clean lines",
    "Digital Art": "digital art, trending on artstation, highly detailed, illustration",
    "Oil Painting": "oil painting on canvas, classical art style, rich texture, brush strokes visible",
    "Watercolor": "watercolor painting, soft edges, fluid colors, wet-on-wet technique",
    "Pixel Art": "pixel art, 16-bit, retro game style, crisp pixels",
    "Comic Book": "comic book style, bold outlines, halftone dots, vibrant flat colors",
    "Cinematic": "cinematic still, movie scene, anamorphic lens, cinematic color grading, film grain",
    "Fantasy Art": "fantasy art, magical, ethereal, epic composition, concept art",
    "Line Art": "line art, ink drawing, pen strokes, minimal, black and white",
    "Isometric": "isometric view, 3D isometric, clean geometric, detailed miniature",
    "Low Poly": "low poly 3D, geometric, stylized, clean background, minimal",
    "Neon Punk": "neon punk, cyberpunk aesthetic, neon lights, dark background, glitch art",
    "Origami": "origami, paper craft, folded paper art, minimal, studio lighting",
}

LIGHTING_PROMPTS: dict[str, str] = {
    "Backlight": "backlit, rim light, silhouette edges",
    "Glowing": "glowing, luminescent, ethereal soft light",
    "Direct Sunlight": "direct sunlight, harsh shadows, bright daylight",
    "Neon Light": "neon lighting, colorful neon glow, electric atmosphere",
    "Studio": "studio lighting, three-point lighting, professional setup, softbox",
    "Soft Light": "soft diffused lighting, gentle shadows, overcast",
    "Hard Light": "hard directional lighting, sharp shadows, high contrast",
    "Rim Light": "rim lighting, edge light, glowing outline",
    "Volumetric": "volumetric lighting, god rays, atmospheric haze, light shafts",
    "Golden Hour": "golden hour, warm sunset light, long soft shadows",
    "Blue Hour": "blue hour, twilight, cool blue ambient tones",
    "Moonlight": "moonlight, night scene, cool silver illumination",
    "Candlelight": "candlelight, warm orange glow, intimate, flickering shadows",
    "Dramatic": "dramatic lighting, chiaroscuro, deep shadows, high contrast",
    "Ambient Occlusion": "ambient occlusion, soft global illumination, contact shadows",
}

COLOR_PROMPTS: dict[str, str] = {
    "Warm": "warm color palette, warm tones, amber and gold hues",
    "Cool": "cool color palette, cool tones, blue and teal hues",
    "Vibrant": "vibrant saturated colors, vivid, bold palette",
    "Muted": "muted colors, desaturated, subdued earthy palette",
    "Monochrome": "monochrome, black and white, grayscale",
    "Pastel": "pastel colors, soft light palette, gentle tones",
    "Neon": "neon colors, fluorescent, bright vivid glowing colors",
    "Sepia": "sepia tone, vintage warm brown, aged photograph look",
    "High Contrast": "high contrast, deep blacks, bright whites, punchy",
    "Desaturated": "desaturated, washed out colors, faded",
    "Cyberpunk Palette": "cyberpunk palette, magenta, cyan, neon pink and blue",
    "Earthy Tones": "earthy tones, brown, olive green, natural organic palette",
    "Sunset Gradient": "sunset gradient, warm orange to cool purple transition",
}

QUALITY_PRESETS: dict[str, dict] = {
    "Draft": {
        "steps": 15,
        "scheduler": "euler",
        "prompt_suffix": "",
        "negative_extra": "",
    },
    "Normal": {
        "steps": 25,
        "scheduler": "dpm++_2m_karras",
        "prompt_suffix": "high quality, detailed",
        "negative_extra": "",
    },
    "High": {
        "steps": 35,
        "scheduler": "dpm++_2m_karras",
        "prompt_suffix": "masterpiece, best quality, highly detailed, 8k",
        "negative_extra": "lowres, jpeg artifacts",
    },
    "Ultra": {
        "steps": 50,
        "scheduler": "dpm++_2m_sde_karras",
        "prompt_suffix": "masterpiece, best quality, ultra detailed, 8k uhd, professional, award winning",
        "negative_extra": "lowres, jpeg artifacts, compression artifacts",
    },
}

BASE_NEGATIVE = (
    "(worst quality, low quality:1.4), blurry, watermark, text, signature, "
    "username, artist name, cropped, out of frame"
)

REALISTIC_NEGATIVE_EXTRA = (
    "(semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), "
    "(bad anatomy, bad hands, extra fingers, missing fingers:1.3), "
    "deformed, disfigured, mutation, ugly"
)


def build_enhanced_prompt(
    user_prompt: str,
    *,
    style: str | None = None,
    lighting: str | None = None,
    color: str | None = None,
    quality: str = "Normal",
) -> str:
    """Combine user prompt with style/lighting/color/quality tokens for SD 1.5."""
    parts: list[str] = []

    if user_prompt.strip():
        parts.append(user_prompt.strip())

    if style and style in STYLE_PROMPTS:
        parts.append(STYLE_PROMPTS[style])

    if lighting and lighting in LIGHTING_PROMPTS:
        parts.append(LIGHTING_PROMPTS[lighting])

    if color and color in COLOR_PROMPTS:
        parts.append(COLOR_PROMPTS[color])

    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["Normal"])
    if preset["prompt_suffix"]:
        parts.append(preset["prompt_suffix"])

    return ", ".join(parts)


def build_negative_prompt(
    user_negative: str = "",
    *,
    style: str | None = None,
    quality: str = "Normal",
) -> str:
    """Build negative prompt with base tokens + quality extras + style-specific tokens."""
    parts: list[str] = [BASE_NEGATIVE]

    if style in ("Realistic", "Photographic"):
        parts.append(REALISTIC_NEGATIVE_EXTRA)

    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["Normal"])
    if preset["negative_extra"]:
        parts.append(preset["negative_extra"])

    if user_negative.strip():
        parts.append(user_negative.strip())

    return ", ".join(parts)


def get_quality_steps(quality: str, user_steps: int | None = None) -> int:
    """Return inference steps — user override or quality preset default."""
    if user_steps and user_steps > 0:
        return user_steps
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["Normal"])
    return preset["steps"]


def get_quality_scheduler(quality: str, user_scheduler: str | None = None) -> str:
    """Return scheduler name — user override or quality preset default."""
    if user_scheduler:
        return user_scheduler
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["Normal"])
    return preset["scheduler"]
