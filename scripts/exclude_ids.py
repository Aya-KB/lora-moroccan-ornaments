"""Exclut (ou ré-ouvre) des images par identifiant, sans toucher aux autres.

    python scripts/exclude_ids.py img_001 img_002 img_003        # exclure
    python scripts/exclude_ids.py --keep img_010 img_011         # annuler une exclusion
    python scripts/exclude_ids.py --reset-caption img_001        # légende brouillon, reviewed=0
    python scripts/exclude_ids.py --list                         # liste id, statut, titre, légende
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_utils as du  # noqa: E402
from src.utils import load_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--keep", action="store_true", help="annule l'exclusion des ids donnés")
    ap.add_argument("--reset-caption", action="store_true", help="légende brouillon + reviewed=0")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    meta = du.load_meta()
    if a.list:
        for _, r in meta.iterrows():
            flag = "X" if r["status"] == "excluded" else ("v" if r["reviewed"] == "1" else ".")
            print(f"{flag} {r['id']}  {r['title'][:55]:55}  | {r['caption'][-45:]}")
        print("\nX = exclue, v = relue, . = à relire")
        return
    unknown = [i for i in a.ids if i not in set(meta["id"])]
    if unknown:
        sys.exit(f"Identifiants inconnus : {unknown}")
    prefix = load_yaml("configs/data.yaml")["caption_prefix"]
    for i in a.ids:
        m = meta["id"] == i
        if a.reset_caption:
            title = meta.loc[m, "title"].iloc[0]
            meta.loc[m, "caption"] = du.draft_caption(title, prefix)
            meta.loc[m, "reviewed"] = "0"
        elif a.keep:
            meta.loc[m, "status"] = ""
        else:
            meta.loc[m, "status"] = "excluded"
            meta.loc[m, "reviewed"] = "0"
    du.save_meta(meta)
    print(f"{len(a.ids)} image(s) mises à jour. Exclues au total : {(meta['status'] == 'excluded').sum()}")


if __name__ == "__main__":
    main()