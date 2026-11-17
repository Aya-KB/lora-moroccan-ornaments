"""Relecture interactive des légendes, image par image.

    python scripts/review_captions.py

Pour chaque image non relue : l'image s'ouvre, la légende brouillon s'affiche. Tapez :
  <Entrée>      garder la légende telle quelle
  du texte      nouvelle description (le préfixe « a photo of moroccan zellige tilework, » est ajouté seul)
  x             exclure l'image (hors sujet, filigrane, portrait...)
  s             passer (à revoir plus tard)
  q             sauvegarder et quitter
La sauvegarde se fait après chaque image : vous pouvez arrêter et reprendre quand vous voulez.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image  # noqa: E402

from src import data_utils as du  # noqa: E402
from src.utils import load_yaml  # noqa: E402


def show(path: Path) -> None:
    if hasattr(os, "startfile"):          # Windows : visionneuse par défaut
        os.startfile(str(path))           # type: ignore[attr-defined]
    else:
        Image.open(path).show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-open", action="store_true", help="n'ouvre pas les images (tests)")
    a = ap.parse_args()
    prefix = load_yaml("configs/data.yaml")["caption_prefix"]
    meta = du.load_meta()
    todo = [i for i, r in meta.iterrows() if r["reviewed"] != "1" and r["status"] != "excluded"]
    print(f"{len(todo)} images à relire. (Entrée = garder, x = exclure, s = passer, q = quitter)\n")
    for n, i in enumerate(todo, 1):
        r = meta.loc[i]
        path = du.find_raw(r["id"])
        if path is None:
            print(f"{r['id']}: fichier introuvable, ignoré")
            continue
        if not a.no_open:
            show(path)
        print(f"[{n}/{len(todo)}] {r['id']} — {r['title']}  ({r['license']})")
        print(f"   légende : {r['caption']}")
        ans = input("   > ").strip()
        if ans.lower() == "q":
            break
        if ans.lower() == "s":
            continue
        if ans.lower() == "x":
            meta.at[i, "status"] = "excluded"
            meta.at[i, "reviewed"] = "0"
        else:
            if ans:
                meta.at[i, "caption"] = ans if ans.lower().startswith("a photo of") else prefix + ans
            meta.at[i, "reviewed"] = "1"
        du.save_meta(meta)
        print()
    du.save_meta(meta)
    rev = (meta["reviewed"] == "1").sum()
    exc = (meta["status"] == "excluded").sum()
    print(f"\nSauvegardé : {rev} relues, {exc} exclues, {len(meta) - rev - exc} restantes.")
    print("Ensuite : python -m src.data_utils prepare && python -m src.data_utils check")


if __name__ == "__main__":
    main()
