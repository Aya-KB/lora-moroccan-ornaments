"""Démo interactive Gradio : prompt + intensité de la LoRA + comparaison avant/après.

    python demo/app.py --lora outputs/r16_s800/checkpoint-800 --share
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402

from src.generate import generate_image, load_pipeline  # noqa: E402

PREFIX = "a photo of moroccan zellige tilework, "


def build_demo(pipe):
    cache: dict = {}

    def before_image(prompt, neg, seed, steps, guidance):
        key = (prompt, neg, int(seed), int(steps), float(guidance))
        if key not in cache:                      # LoRA à l'intensité 0 = modèle de base
            cache[key] = generate_image(pipe, prompt, seed, steps, guidance, 0.0, neg)
        return cache[key]

    def compare(prompt, neg, scale, seed, steps, guidance, add_prefix):
        full = (PREFIX + prompt) if add_prefix else prompt
        before = before_image(full, neg, seed, steps, guidance)
        after = generate_image(pipe, full, seed, steps, guidance, scale, neg)
        return before, after

    def sweep(prompt, neg, seed, steps, guidance, add_prefix):
        full = (PREFIX + prompt) if add_prefix else prompt
        scales = [0.0, 0.5, 1.0, 1.5]
        return [(generate_image(pipe, full, seed, steps, guidance, s, neg), f"intensité {s}") for s in scales]

    with gr.Blocks(title="LoRA zellige marocain") as demo:
        gr.Markdown("# Stable Diffusion 1.5 + LoRA « zellige marocain »\n"
                    "Même graine, même prompt : à gauche le modèle de base, à droite avec la LoRA.")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value="a courtyard with a central fountain and tiled walls", lines=2)
            neg = gr.Textbox(label="Prompt négatif", value="blurry, low quality, text, watermark", lines=2)
        add_prefix = gr.Checkbox(label=f"Ajouter le préfixe d'entraînement « {PREFIX.strip()} »", value=True)
        with gr.Row():
            scale = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Intensité de la LoRA")
            seed = gr.Number(value=0, precision=0, label="Graine")
            steps = gr.Slider(10, 50, value=25, step=1, label="Étapes de diffusion")
            guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance (CFG)")
        with gr.Tab("Avant / après"):
            btn = gr.Button("Générer", variant="primary")
            with gr.Row():
                out_b = gr.Image(label="Avant : modèle de base", type="pil")
                out_a = gr.Image(label="Après : avec LoRA", type="pil")
            btn.click(compare, [prompt, neg, scale, seed, steps, guidance, add_prefix], [out_b, out_a])
        with gr.Tab("Balayage d'intensité"):
            btn2 = gr.Button("Comparer 4 intensités")
            gal = gr.Gallery(label="0.0 / 0.5 / 1.0 / 1.5", columns=4)
            btn2.click(sweep, [prompt, neg, seed, steps, guidance, add_prefix], gal)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--lora", required=True, help="dossier contenant pytorch_lora_weights.safetensors")
    ap.add_argument("--share", action="store_true", help="lien public (utile sur Colab/Kaggle)")
    a = ap.parse_args()
    pipe = load_pipeline(a.base, a.lora)
    build_demo(pipe).queue().launch(share=a.share)


if __name__ == "__main__":
    main()
