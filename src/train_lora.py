"""Entraînement d'une LoRA sur l'UNet de Stable Diffusion 1.5.

Choix pour tenir dans 12 Go de VRAM (mesuré et enregistré dans run_info.json) :
  * poids gelés en fp16, paramètres LoRA en fp32 + autocast fp16 + GradScaler ;
  * VAE et encodeur de texte utilisés une seule fois (latents et embeddings mis en cache) puis libérés ;
  * gradient checkpointing, batch de 2, résolution 512.

Usage :
    python -m src.train_lora --config configs/train_base.yaml --rank 16 --steps 800
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .data_utils import load_split
from .utils import load_yaml, resolve, save_json, set_seed


def pil_to_tensor(img: Image.Image, size: int):
    import torch

    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def save_lora(unet, out_dir: Path) -> None:
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    out_dir.mkdir(parents=True, exist_ok=True)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(out_dir), unet_lora_layers=state, safe_serialization=True
    )


def train(cfg: dict) -> Path:
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Un GPU CUDA est requis (Colab : Exécution > Modifier le type d'exécution > T4).")
    device = "cuda"
    set_seed(cfg["seed"])
    rank = int(cfg["rank"])
    alpha = int(cfg["alpha"]) if cfg.get("alpha") else rank
    steps = int(cfg["steps"])
    bs, accum = int(cfg["batch_size"]), int(cfg["grad_accum"])
    run_name = cfg.get("run_name") or f"r{rank}_s{steps}"
    out_root = resolve(cfg["output_root"]) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    items = load_split("train")
    if len(items) == 0:
        raise RuntimeError("Aucune image d'entraînement : exécutez `python -m src.data_utils prepare`.")
    print(f"[{run_name}] {len(items)} images d'entraînement, rang={rank}, alpha={alpha}, pas={steps}")

    torch.cuda.reset_peak_memory_stats()
    base = cfg["pretrained_model"]
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=torch.float16).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")

    # ---- mise en cache des latents (avec version retournée) et des embeddings de texte
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    latents_cache, embeds_cache = [], []
    with torch.no_grad():
        for path, caption, _ in items:
            x = pil_to_tensor(Image.open(path), cfg["resolution"]).unsqueeze(0).to(device, torch.float16)
            variants = [x] + ([torch.flip(x, dims=[3])] if cfg.get("hflip", True) else [])
            lats = [vae.encode(v).latent_dist.sample(gen) * vae.config.scaling_factor for v in variants]
            ids = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors="pt").input_ids.to(device)
            emb = text_encoder(ids)[0]
            latents_cache.append(torch.cat(lats, dim=0))          # (1 ou 2, 4, 64, 64)
            embeds_cache.append(emb)                              # (1, 77, 768)
    del vae, text_encoder
    torch.cuda.empty_cache()

    # ---- UNet gelé + adaptateur LoRA
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=torch.float16).to(device)
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=rank, lora_alpha=alpha, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()           # LoRA en fp32 (stabilité avec GradScaler)
    n_params = sum(p.numel() for p in params)
    if cfg.get("grad_checkpointing", True):
        unet.enable_gradient_checkpointing()
    unet.train()

    optimizer = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    lr_sched = get_scheduler(cfg["lr_scheduler"], optimizer=optimizer,
                             num_warmup_steps=int(cfg["warmup_steps"]), num_training_steps=steps)
    scaler = torch.amp.GradScaler("cuda")
    pred_type = noise_scheduler.config.prediction_type
    save_steps = {s for s in range(int(cfg["save_every"]), steps + 1, int(cfg["save_every"]))} | {steps}

    rng = random.Random(cfg["seed"])
    losses, ckpt_time, window = [], {}, []
    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(1, steps + 1):
        step_loss = 0.0
        for _ in range(accum):
            idx = [rng.randrange(len(items)) for _ in range(bs)]
            lat = torch.cat([latents_cache[i][rng.randrange(latents_cache[i].shape[0])][None] for i in idx])
            enc = torch.cat([embeds_cache[i] for i in idx])
            noise = torch.randn_like(lat)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
            noisy = noise_scheduler.add_noise(lat, noise, t)
            target = noise if pred_type == "epsilon" else noise_scheduler.get_velocity(lat, noise, t)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = unet(noisy, t, encoder_hidden_states=enc).sample
            loss = F.mse_loss(pred.float(), target.float()) / accum
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
        scaler.step(optimizer)
        scaler.update()
        lr_sched.step()
        optimizer.zero_grad(set_to_none=True)
        if not math.isfinite(step_loss):
            raise RuntimeError(f"Perte non finie au pas {step} : baissez lr ou vérifiez les données.")
        losses.append(step_loss)
        window.append(step_loss)
        if step % 25 == 0 or step == 1:
            print(f"  pas {step:5d}/{steps}  perte(moy 25) = {np.mean(window[-25:]):.4f}  "
                  f"lr = {lr_sched.get_last_lr()[0]:.2e}")
        if step in save_steps:
            torch.cuda.synchronize()
            ckpt_time[step] = time.time() - t0
            save_lora(unet, out_root / f"checkpoint-{step}")
            print(f"  -> checkpoint-{step} sauvegardé ({ckpt_time[step]:.0f} s)")
    torch.cuda.synchronize()
    total = time.time() - t0

    import diffusers, peft, transformers
    info = {
        "run_name": run_name, "rank": rank, "alpha": alpha, "steps": steps,
        "batch_size": bs, "grad_accum": accum, "lr": float(cfg["lr"]),
        "n_train_images": len(items),
        "epochs_equivalent": round(steps * bs * accum / len(items), 1),
        "trainable_params": n_params,
        "train_time_s": round(total, 1),
        "steps_per_s": round(steps / total, 3),
        "ckpt_time_s": {str(k): round(v, 1) for k, v in ckpt_time.items()},
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                     "peft": peft.__version__, "transformers": transformers.__version__},
        "config": cfg, "loss": [round(x, 5) for x in losses],
    }
    save_json(info, out_root / "run_info.json")
    print(f"[{run_name}] terminé en {total/60:.1f} min, pic VRAM {info['peak_vram_gb']} Go, "
          f"{n_params/1e6:.2f} M paramètres entraînables.")
    del unet, optimizer
    torch.cuda.empty_cache()
    return out_root


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train_base.yaml")
    for k, typ in [("rank", int), ("steps", int), ("alpha", int), ("batch_size", int),
                   ("lr", float), ("save_every", int), ("seed", int), ("run_name", str)]:
        ap.add_argument(f"--{k}", type=typ, default=None)
    a = ap.parse_args(argv)
    cfg = load_yaml(a.config)
    for k, v in vars(a).items():
        if k != "config" and v is not None:
            cfg[k] = v
    train(cfg)


if __name__ == "__main__":
    main()
