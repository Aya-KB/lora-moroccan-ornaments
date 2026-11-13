"""Fonctions utilitaires partagées (chemins, YAML, graine aléatoire, planches de vignettes)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    """Chemin absolu : un chemin relatif est interprété par rapport à la racine du dépôt."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_yaml(path: str | Path) -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path: str | Path) -> None:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path):
    with open(resolve(path), encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def contact_sheet(images, labels=None, cols=4, thumb=256, pad=6, label_h=22):
    """Assemble une liste d'images PIL en une planche (avec légendes optionnelles)."""
    from PIL import Image, ImageDraw

    if not images:
        raise ValueError("Aucune image à afficher.")
    rows = (len(images) + cols - 1) // cols
    h_cell = thumb + (label_h if labels else 0)
    sheet = Image.new("RGB", (cols * (thumb + pad) + pad, rows * (h_cell + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x, y = pad + c * (thumb + pad), pad + r * (h_cell + pad)
        sheet.paste(img.convert("RGB").resize((thumb, thumb)), (x, y))
        if labels:
            draw.text((x + 2, y + thumb + 4), str(labels[i])[: thumb // 6], fill="black")
    return sheet
