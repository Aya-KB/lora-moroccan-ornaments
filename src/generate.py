"""Chargement du pipeline Stable Diffusion 1.5 (+ LoRA) et génération reproductible."""
from __future__ import annotations

import argparse
from pathlib import Path

LORA_FILE = "pytorch_lora_weights.safetensors"


def load_pipeline(base: str, lora_dir: str | Path | None = None, device: str = "cuda", dtype=None):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
    pipe = StableDiffusionPipeline.from_pretrained(
        base, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe._lora_active = False
    if lora_dir is not None:
        set_lora(pipe, lora_dir)
    return pipe


def set_lora(pipe, lora_dir: str | Path | None) -> None:
    """Remplace la LoRA active (None => modèle de base)."""
    if getattr(pipe, "_lora_active", False):
        pipe.unload_lora_weights()
        pipe._lora_active = False
    if lora_dir is not None:
        pipe.load_lora_weights(str(lora_dir), weight_name=LORA_FILE)
        pipe._lora_active = True


def generate_image(pipe, prompt: str, seed: int = 0, steps: int = 25, guidance: float = 7.5,
                   scale: float = 1.0, negative_prompt: str | None = None, size: int = 512):
    """Génère une image PIL. `scale` = intensité de la LoRA (0 = modèle de base)."""
    import torch

    gen = torch.Generator(device=pipe.device).manual_seed(int(seed))
    kwargs = dict(prompt=prompt, negative_prompt=negative_prompt or None, num_inference_steps=int(steps),
                  guidance_scale=float(guidance), generator=gen, height=size, width=size)
    if getattr(pipe, "_lora_active", False):
        kwargs["cross_attention_kwargs"] = {"scale": float(scale)}
    return pipe(**kwargs).images[0]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Génère une image avec ou sans LoRA.")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--lora", default=None, help="dossier contenant pytorch_lora_weights.safetensors")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--out", default="results/sample.png")
    a = ap.parse_args(argv)
    pipe = load_pipeline(a.base, a.lora)
    img = generate_image(pipe, a.prompt, a.seed, a.steps, a.guidance, a.scale)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(a.out)
    print("image enregistrée :", a.out)


if __name__ == "__main__":
    main()
