"""Planches de relecture : python scripts/review_sheet.py -> results/review_sheet_N.png"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src import data_utils as du
from src.utils import contact_sheet, resolve

meta = du.load_meta()
items = []
for _, r in meta.iterrows():
    p = du.find_raw(r["id"])
    if p:
        items.append((Image.open(p).convert("RGB"), f'{r["id"]} {r["title"][:30]}'))
for k in range(0, len(items), 12):
    chunk = items[k:k + 12]
    sheet = contact_sheet([i for i, _ in chunk], [l for _, l in chunk], cols=4, thumb=300)
    out = resolve("results") / f"review_sheet_{k // 12 + 1}.png"
    sheet.save(out)
    print(out)