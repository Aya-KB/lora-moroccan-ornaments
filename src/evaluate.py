"""Évaluation quantitative et visuelle : baseline vs LoRA (plusieurs rangs / nombres de pas).

Mesures (toutes via CLIP ViT-B/32, similarité cosinus x 100) :
  * clip_t        : respect du texte (image générée vs prompt). Plus haut = mieux.
  * style_test    : similarité moyenne aux images de TEST (jamais vues) = adoption du domaine.
  * train_max     : similarité max à une image d'ENTRAÎNEMENT. Valeur très haute = copie/mémorisation.
  * copy_rate     : part des images dont train_max dépasse le seuil (surapprentissage).
  * ood_clip_t    : respect du texte hors domaine (une baisse = oubli / sur-spécialisation).
  * coût          : temps d'entraînement, pic VRAM, paramètres entraînables (lus dans run_info.json).

    python -m src.evaluate --config configs/eval.yaml
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .data_utils import load_meta, load_split
from .generate import LORA_FILE, generate_image, load_pipeline, set_lora
from .utils import load_json, load_yaml, resolve


class ClipScorer:
    def __init__(self, name: str, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch, self.device = torch, device
        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(name)

    def text(self, texts: list[str]):
        torch = self.torch
        with torch.no_grad():
            inp = self.proc.tokenizer(texts, padding=True, truncation=True, max_length=77,
                                      return_tensors="pt").to(self.device)
            out = self.model.text_model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
            e = self.model.text_projection(out.pooler_output)
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu()

    def images(self, imgs: list[Image.Image]):
        torch = self.torch
        with torch.no_grad():
            px = self.proc.image_processor(imgs, return_tensors="pt")["pixel_values"].to(self.device)
            e = self.model.visual_projection(self.model.vision_model(pixel_values=px).pooler_output)
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu()


def build_prompts(prompts_cfg: dict, prefix: str, test_items) -> list[dict]:
    rows = []
    for group in ("in_domain", "novel"):
        for i, p in enumerate(prompts_cfg[group]):
            rows.append({"group": group, "pid": f"{group}{i}", "prompt": prefix + p, "ref_id": ""})
    for i, p in enumerate(prompts_cfg["out_of_domain"]):
        rows.append({"group": "out_of_domain", "pid": f"ood{i}", "prompt": p, "ref_id": ""})
    for _, cap, img_id in test_items:       # prompts = légendes des images de test
        rows.append({"group": "test_caption", "pid": f"test_{img_id}", "prompt": cap, "ref_id": img_id})
    return rows


def discover_variants(outputs_dir: Path) -> list[dict]:
    """Retourne la baseline + tous les checkpoints trouvés dans outputs/<run>/checkpoint-<pas>."""
    variants = [{"name": "baseline", "run": "baseline", "rank": 0, "steps": 0, "path": None,
                 "train_time_s": 0.0, "peak_vram_gb": np.nan, "params": 0}]
    for run_dir in sorted(outputs_dir.glob("*/")):
        info_p = run_dir / "run_info.json"
        if not info_p.exists():
            continue
        info = load_json(info_p)
        ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        for c in ckpts:
            if not (c / LORA_FILE).exists():
                continue
            step = int(c.name.split("-")[1])
            variants.append({
                "name": f"{info['run_name']}@{step}", "run": info["run_name"], "rank": info["rank"],
                "steps": step, "path": c,
                "train_time_s": info["ckpt_time_s"].get(str(step), info["train_time_s"]),
                "peak_vram_gb": info["peak_vram_gb"], "params": info["trainable_params"],
            })
    return variants


def labeled_grid(columns: dict[str, list[Image.Image]], row_labels: list[str], thumb=256) -> Image.Image:
    """Grille : une colonne par modèle, une ligne par prompt."""
    names = list(columns)
    head, left, pad = 24, 150, 4
    n_rows = len(row_labels)
    W = left + len(names) * (thumb + pad)
    H = head + n_rows * (thumb + pad)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for j, n in enumerate(names):
        d.text((left + j * (thumb + pad) + 4, 6), n, fill="black")
        for i in range(n_rows):
            sheet.paste(columns[n][i].resize((thumb, thumb)), (left + j * (thumb + pad), head + i * (thumb + pad)))
    for i, lab in enumerate(row_labels):
        words, line, y = lab.split(), "", head + i * (thumb + pad) + 4
        for w in words:
            if len(line) + len(w) > 24:
                d.text((4, y), line, fill="black"); y += 12; line = ""
            line += w + " "
        d.text((4, y), line, fill="black")
    return sheet


def run_evaluation(cfg: dict) -> pd.DataFrame:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = resolve(cfg["results_dir"])
    (res / "figures").mkdir(parents=True, exist_ok=True)
    gen_dir = res / "generated"
    prefix = load_yaml("configs/data.yaml")["caption_prefix"]
    train_items, test_items = load_split("train"), load_split("test")
    prompts = build_prompts(load_yaml("configs/prompts.yaml"), prefix, test_items)
    variants = discover_variants(resolve(cfg["outputs_dir"]))
    if len(variants) == 1:
        print("Attention : aucun checkpoint LoRA trouvé, seule la baseline sera évaluée.")

    clip = ClipScorer(cfg["clip_model"], device)
    train_emb = clip.images([Image.open(p) for p, _, _ in train_items])
    test_emb = clip.images([Image.open(p) for p, _, _ in test_items])
    test_by_id = {i: test_emb[k] for k, (_, _, i) in enumerate(test_items)}

    pipe = load_pipeline(cfg["pretrained_model"], None, device)
    rows, grid_imgs = [], {}
    for v in variants:
        set_lora(pipe, v["path"])
        print(f"== génération : {v['name']}")
        for pr in prompts:
            for seed in cfg["seeds"]:
                out = gen_dir / v["name"].replace("@", "_") / f"{pr['pid']}_s{seed}.png"
                if out.exists():
                    img = Image.open(out).convert("RGB")
                else:
                    img = generate_image(pipe, pr["prompt"], seed, cfg["num_inference_steps"],
                                         cfg["guidance_scale"], cfg["lora_scale"], cfg["negative_prompt"])
                    out.parent.mkdir(parents=True, exist_ok=True)
                    img.save(out)
                ie = clip.images([img])[0]
                te = clip.text([pr["prompt"]])[0]
                sims_train = (train_emb @ ie).numpy() * 100
                rows.append({
                    "variant": v["name"], "run": v["run"], "rank": v["rank"], "steps": v["steps"],
                    "group": pr["group"], "pid": pr["pid"], "seed": seed, "prompt": pr["prompt"],
                    "clip_t": float(te @ ie) * 100,
                    "style_test": float((test_emb @ ie).mean()) * 100,
                    "train_max": float(sims_train.max()),
                    "ref_sim": float(test_by_id[pr["ref_id"]] @ ie) * 100 if pr["ref_id"] else np.nan,
                })
                if seed == cfg["grid_seed"]:
                    grid_imgs.setdefault(v["name"], {})[pr["pid"]] = img
    df = pd.DataFrame(rows)
    df["copy"] = (df["train_max"] / 100 > cfg["memorization_threshold"]).astype(float)
    df.to_csv(res / "metrics_per_image.csv", index=False)

    # ---- résumé par variante (coût + moyennes par groupe)
    summ = pd.DataFrame(variants).drop(columns=["path"])
    piv = lambda col, grp: df[df.group == grp].groupby("variant")[col].mean()
    nov = df[df.group.isin(["in_domain", "novel", "test_caption"])]
    summ = summ.set_index("name")
    summ["clip_t_in_domain"] = piv("clip_t", "in_domain")
    summ["clip_t_novel"] = piv("clip_t", "novel")
    summ["clip_t_test_caption"] = piv("clip_t", "test_caption")
    summ["style_test_novel"] = piv("style_test", "novel")
    summ["ref_sim_test"] = piv("ref_sim", "test_caption")
    summ["clip_t_ood"] = piv("clip_t", "out_of_domain")
    summ["style_test_ood"] = piv("style_test", "out_of_domain")
    summ["train_max_novel"] = piv("train_max", "novel")
    summ["copy_rate_domain"] = nov.groupby("variant")["copy"].mean()
    summ = summ.reset_index().rename(columns={"name": "variant"})
    summ.to_csv(res / "metrics_summary.csv", index=False, float_format="%.3f")

    # ---- grilles visuelles (une par groupe de prompts)
    for group in ("in_domain", "novel", "out_of_domain", "test_caption"):
        pids = [p["pid"] for p in prompts if p["group"] == group]
        cols = {name: [imgs[p] for p in pids] for name, imgs in grid_imgs.items()}
        labels = [next(p["prompt"] for p in prompts if p["pid"] == pid)[:80] for pid in pids]
        labeled_grid(cols, labels).save(res / "figures" / f"grid_{group}.png")
    make_figures(summ, df, res / "figures", cfg)
    print(summ.round(2).to_string())
    return summ


def make_figures(summ: pd.DataFrame, df: pd.DataFrame, fig_dir: Path, cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lora = summ[summ["rank"] > 0]
    base = summ[summ["rank"] == 0]
    panels = [
        ("clip_t_novel", "Respect du texte (prompts nouveaux)\nCLIP-T ↑"),
        ("style_test_novel", "Adoption du style (vs images de test)\nCLIP-I ↑"),
        ("clip_t_ood", "Respect du texte hors domaine\n(chute = oubli) ↑"),
        ("copy_rate_domain", "Taux de copie des images d'entraînement\n(surapprentissage) ↓"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for ax, (col, title) in zip(axes, panels):
        for r, g in lora.groupby("rank"):
            g = g.sort_values("steps")
            ax.plot(g["steps"], g[col], marker="o", label=f"rang {int(r)}")
        if len(base):
            ax.axhline(float(base[col].iloc[0]), color="gray", ls="--", label="baseline")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("pas d'entraînement")
        ax.grid(alpha=.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "metrics_vs_steps.png", dpi=150)
    plt.close(fig)

    # courbes de perte
    fig, ax = plt.subplots(figsize=(6, 4))
    for run in lora["run"].unique():
        p = resolve(cfg["outputs_dir"]) / run / "run_info.json"
        if p.exists():
            loss = np.array(load_json(p)["loss"])
            k = min(50, max(1, len(loss) // 10))
            ax.plot(np.convolve(loss, np.ones(k) / k, mode="valid"), label=run)
    ax.set_xlabel("pas"); ax.set_ylabel("perte MSE (moyenne glissante)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "training_loss.png", dpi=150)
    plt.close(fig)

    # coût
    final = lora.sort_values("steps").groupby("run").tail(1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (col, t) in zip(axes, [("train_time_s", "Temps d'entraînement (s)"),
                                   ("peak_vram_gb", "Pic VRAM (Go)"),
                                   ("params", "Paramètres entraînables")]):
        ax.bar(final["run"], final[col])
        ax.set_title(t, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    axes[1].axhline(12, color="red", ls="--")
    fig.tight_layout()
    fig.savefig(fig_dir / "cost.png", dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    a = ap.parse_args(argv)
    run_evaluation(load_yaml(a.config))


if __name__ == "__main__":
    main()
