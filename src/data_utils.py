"""Constitution du mini-jeu de données : recherche/téléchargement (Wikimedia Commons),
préparation (recadrage 512x512, doublons, split train/test) et contrôle qualité.

Commandes (depuis la racine du dépôt) :
    python -m src.data_utils download            # cherche + télécharge des images autorisées
    python -m src.data_utils fetch               # re-télécharge les images listées dans metadata.csv
    python -m src.data_utils add-own --dir MES_PHOTOS --author "Prénom Nom" --license "CC BY 4.0"
    python -m src.data_utils prepare             # 512x512, doublons, split train/test
    python -m src.data_utils check               # vérifie licences, légendes, effectifs
    python -m src.data_utils stats
"""
from __future__ import annotations

import argparse
import html
import random
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .utils import load_yaml, resolve

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "LoRA-ZellijeStudentProject/1.0 (projet pedagogique; voir README du depot)"
}
META_PATH = "data/metadata.csv"
RAW_DIR, OWN_DIR, PROC_DIR = "data/raw", "data/own", "data/processed"
COLUMNS = [
    "id", "title", "author", "license", "source_url", "file_url",
    "caption", "reviewed", "status", "split",
]
ALLOWED_LICENSE = re.compile(r"^(cc[ -]?by|cc0|cc[ -]?zero|public domain|pd)", re.I)
FORBIDDEN_LICENSE = re.compile(r"(-nc|-nd|\bnc\b|\bnd\b)", re.I)


# --------------------------------------------------------------------------- métadonnées
def load_meta() -> pd.DataFrame:
    p = resolve(META_PATH)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[COLUMNS]


def save_meta(df: pd.DataFrame) -> None:
    p = resolve(META_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(p, index=False, encoding="utf-8")


def next_ids(existing: set[str], n: int) -> list[str]:
    out, i = [], 1
    while len(out) < n:
        cand = f"img_{i:03d}"
        if cand not in existing:
            out.append(cand)
            existing = existing | {cand}
        i += 1
    return out


def find_raw(img_id: str) -> Path | None:
    for d in (RAW_DIR, OWN_DIR):
        hits = sorted(resolve(d).glob(f"{img_id}.*"))
        if hits:
            return hits[0]
    return None


# --------------------------------------------------------------------------- textes
def strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def license_ok(name: str) -> bool:
    name = (name or "").strip()
    return bool(ALLOWED_LICENSE.match(name)) and not FORBIDDEN_LICENSE.search(name)


def clean_title(title: str) -> str:
    t = re.sub(r"^(File|Fichier):", "", title or "", flags=re.I)
    t = re.sub(r"\.(jpe?g|png|tiff?|webp)$", "", t, flags=re.I)
    t = re.sub(r"[_\-]+", " ", t)
    t = re.sub(r"\b(IMG|DSC|DSCN|PXL|CIMG|P)\s?\d{3,}\b", " ", t, flags=re.I)
    t = re.sub(r"\(\d+\)", " ", t)
    t = re.sub(r"\b\d{3,}\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,.-").lower()
    return t


def draft_caption(title: str, prefix: str) -> str:
    c = clean_title(title)
    return (prefix + c) if c else prefix.rstrip(", ")


# --------------------------------------------------------------------------- Commons
def _query(params: dict) -> dict:
    import requests

    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def iter_commons_files(source: dict, pause: float = 1.0):
    """Génère (titre, imageinfo) pour une catégorie ou une recherche Commons."""
    base = {
        "action": "query", "format": "json", "formatversion": 2,
        "prop": "imageinfo", "iiprop": "url|extmetadata|mime|size", "iiurlwidth": 1024,
    }
    if source["type"] == "category":
        base.update(generator="categorymembers", gcmtitle=source["name"],
                    gcmtype="file", gcmlimit=50)
    elif source["type"] == "search":
        base.update(generator="search", gsrsearch=f'{source["name"]} filetype:bitmap',
                    gsrnamespace=6, gsrlimit=50)
    else:
        raise ValueError(f"type de source inconnu : {source['type']}")
    cont: dict = {}
    while True:
        data = _query({**base, **cont})
        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [None])[0]
            if info:
                yield page["title"], info
        if "continue" not in data:
            return
        cont = data["continue"]
        time.sleep(pause)


def _download_file(url: str, dest: Path) -> None:
    import requests

    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)


def cmd_download(cfg: dict) -> None:
    meta = load_meta()
    known = set(meta["source_url"])
    existing_ids = set(meta["id"])
    new_rows: list[dict] = []
    bad_words = [w.lower() for w in cfg.get("exclude_title_words", [])]
    for src in cfg["sources"]:
        if len(meta) + len(new_rows) >= cfg["max_images"]:
            break
        seen = kept = 0
        try:
            for title, info in iter_commons_files(src, cfg["request_pause_s"]):
                if len(meta) + len(new_rows) >= cfg["max_images"]:
                    break
                seen += 1
                ext = info.get("extmetadata", {})
                lic = strip_html(ext.get("LicenseShortName", {}).get("value", ""))
                page_url = info.get("descriptionurl", "")
                if not license_ok(lic) or page_url in known:
                    continue
                if info.get("mime") not in ("image/jpeg", "image/png"):
                    continue
                if min(info.get("width", 0), info.get("height", 0)) < cfg["min_side"]:
                    continue
                if any(w in title.lower() for w in bad_words):
                    continue
                url = info.get("thumburl") or info["url"]
                new_id = next_ids(existing_ids | {r["id"] for r in new_rows}, 1)[0]
                suffix = ".png" if info["mime"] == "image/png" else ".jpg"
                try:
                    _download_file(url, resolve(RAW_DIR) / f"{new_id}{suffix}")
                except Exception as e:  # réseau capricieux : on passe à l'image suivante
                    print(f"  ! échec téléchargement {title}: {e}")
                    continue
                new_rows.append({
                    "id": new_id,
                    "title": title.replace("File:", ""),
                    "author": strip_html(ext.get("Artist", {}).get("value", "")) or "inconnu",
                    "license": lic,
                    "source_url": page_url,
                    "file_url": url,
                    "caption": draft_caption(title, cfg["caption_prefix"]),
                    "reviewed": "0",
                    "status": "",
                    "split": "",
                })
                known.add(page_url)
                kept += 1
                time.sleep(cfg["request_pause_s"])
        except Exception as e:
            print(f"! source {src} ignorée ({e})")
        print(f"source {src['name']!r}: {seen} fichiers vus, {kept} retenus")
    if new_rows:
        meta = pd.concat([meta, pd.DataFrame(new_rows)], ignore_index=True)
        save_meta(meta)
    print(f"{len(new_rows)} nouvelles images ; total = {len(meta)}. "
          f"Relisez et corrigez les légendes dans {META_PATH} puis mettez reviewed=1.")


def cmd_fetch(cfg: dict) -> None:
    meta = load_meta()
    n = 0
    for _, row in meta.iterrows():
        if find_raw(row["id"]) is None and row["file_url"]:
            suffix = ".png" if row["file_url"].lower().endswith(".png") else ".jpg"
            _download_file(row["file_url"], resolve(RAW_DIR) / f"{row['id']}{suffix}")
            n += 1
            time.sleep(cfg["request_pause_s"])
    print(f"{n} images re-téléchargées.")


def cmd_add_own(cfg: dict, directory: str, author: str, license_: str) -> None:
    meta = load_meta()
    files = sorted(p for p in Path(directory).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    ids = next_ids(set(meta["id"]), len(files))
    rows = []
    for p, new_id in zip(files, ids):
        img = Image.open(p).convert("RGB")
        img.thumbnail((1024, 1024))
        dest = resolve(OWN_DIR) / f"{new_id}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, quality=92)
        rows.append({
            "id": new_id, "title": p.stem, "author": author, "license": license_,
            "source_url": f"photo originale ({p.name})", "file_url": "",
            "caption": draft_caption(p.stem, cfg["caption_prefix"]),
            "reviewed": "0", "status": "", "split": "",
        })
    if rows:
        save_meta(pd.concat([meta, pd.DataFrame(rows)], ignore_index=True))
    print(f"{len(rows)} photos ajoutées dans {OWN_DIR}/ (à commiter : elles ne sont pas re-téléchargeables).")


# --------------------------------------------------------------------------- préparation
def dhash(img: Image.Image, size: int = 8) -> np.ndarray:
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def cmd_prepare(cfg: dict) -> None:
    meta = load_meta()
    size = int(cfg["image_size"])
    proc = resolve(PROC_DIR)
    if proc.exists():
        shutil.rmtree(proc)
    (proc / "train").mkdir(parents=True)
    (proc / "test").mkdir(parents=True)

    accepted: list[tuple[str, np.ndarray]] = []
    tmp: dict[str, Image.Image] = {}
    for i, row in meta.iterrows():
        if row["status"] == "excluded":      # exclusion manuelle (filigrane, hors sujet...)
            meta.at[i, "split"] = ""
            continue
        path = find_raw(row["id"])
        if path is None:
            meta.at[i, "status"], meta.at[i, "split"] = "missing", ""
            continue
        img = Image.open(path).convert("RGB")
        if min(img.size) < cfg["min_side"]:
            meta.at[i, "status"], meta.at[i, "split"] = "too_small", ""
            continue
        img = center_square(img).resize((size, size), Image.LANCZOS)
        h = dhash(img)
        dup = next((o for o, oh in accepted if np.count_nonzero(h != oh) <= cfg["dedup_hamming"]), None)
        if dup:
            print(f"doublon : {row['id']} ~ {dup}")
            meta.at[i, "status"], meta.at[i, "split"] = "duplicate", ""
            continue
        accepted.append((row["id"], h))
        tmp[row["id"]] = img
        meta.at[i, "status"] = "ok"

    ok_ids = sorted(tmp)
    n_test = min(int(cfg["n_test"]), max(len(ok_ids) - 1, 0))
    test_ids = set(random.Random(cfg["seed"]).sample(ok_ids, n_test)) if n_test else set()
    for i, row in meta.iterrows():
        if row["id"] in tmp:
            split = "test" if row["id"] in test_ids else "train"
            meta.at[i, "split"] = split
            tmp[row["id"]].save(proc / split / f"{row['id']}.png")
    save_meta(meta)
    print(f"{len(ok_ids) - n_test} images train, {n_test} test dans {PROC_DIR}/")
    print(meta["status"].value_counts().to_string())


# --------------------------------------------------------------------------- contrôle
def cmd_check(cfg: dict) -> int:
    meta = load_meta()
    ok = meta[meta["status"] == "ok"]
    errors, warnings = [], []
    n_train, n_test = (ok["split"] == "train").sum(), (ok["split"] == "test").sum()
    if len(ok) == 0:
        errors.append("aucune image valide : lancez `prepare`.")
    if not 20 <= len(ok) <= 50:
        warnings.append(f"{len(ok)} images valides (objectif : 20 à 50).")
    if not 5 <= n_test <= 10:
        warnings.append(f"{n_test} images de test (objectif : 5 à 10).")
    for _, r in ok.iterrows():
        tag = r["id"]
        if not license_ok(r["license"]):
            errors.append(f"{tag}: licence non vérifiée/non compatible ({r['license']!r}).")
        if not r["author"]:
            errors.append(f"{tag}: auteur manquant (attribution obligatoire).")
        n_words = len(r["caption"].split())
        if n_words < 4:
            errors.append(f"{tag}: légende trop courte ({n_words} mots).")
        if n_words > 40:
            warnings.append(f"{tag}: légende longue ({n_words} mots ; CLIP tronque à 77 tokens).")
        if r["reviewed"] != "1":
            errors.append(f"{tag}: légende non relue (mettez reviewed=1 après correction).")
        if not (resolve(PROC_DIR) / r["split"] / f"{tag}.png").exists():
            errors.append(f"{tag}: image préparée absente (relancez `prepare`).")
    for w in warnings:
        print("AVERTISSEMENT :", w)
    for e in errors:
        print("ERREUR :", e)
    print(f"\n{len(ok)} images valides ({n_train} train / {n_test} test) — "
          f"{len(errors)} erreur(s), {len(warnings)} avertissement(s).")
    return 1 if errors else 0


def cmd_stats() -> None:
    meta = load_meta()
    print(meta["status"].replace("", "(non préparé)").value_counts().to_string())
    print()
    print(meta[meta["status"] == "ok"]["license"].value_counts().to_string())


# --------------------------------------------------------------------------- accès pour l'entraînement
def load_split(split: str) -> list[tuple[Path, str, str]]:
    """Retourne [(chemin_image, légende, id)] pour 'train' ou 'test'."""
    meta = load_meta()
    sel = meta[(meta["status"] == "ok") & (meta["split"] == split)]
    return [(resolve(PROC_DIR) / split / f"{r['id']}.png", r["caption"], r["id"])
            for _, r in sel.iterrows()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("download", "fetch", "prepare", "check", "stats"):
        sub.add_parser(name)
    own = sub.add_parser("add-own")
    own.add_argument("--dir", required=True)
    own.add_argument("--author", required=True)
    own.add_argument("--license", required=True)
    a = ap.parse_args(argv)
    cfg = load_yaml(a.config)
    if a.cmd == "download":
        cmd_download(cfg)
    elif a.cmd == "fetch":
        cmd_fetch(cfg)
    elif a.cmd == "prepare":
        cmd_prepare(cfg)
    elif a.cmd == "check":
        return cmd_check(cfg)
    elif a.cmd == "stats":
        cmd_stats()
    elif a.cmd == "add-own":
        cmd_add_own(cfg, a.dir, a.author, a.license)
    return 0


if __name__ == "__main__":
    sys.exit(main())
