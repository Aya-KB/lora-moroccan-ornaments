"""Applique les légendes proposées et les exclusions : python scripts/apply_captions.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data_utils as du  # noqa: E402
from src.utils import load_yaml  # noqa: E402

CAPTIONS = {
    "img_009": "black, white, green and golden-brown star rosette pattern with a central cross motif",
    "img_010": "blue and turquoise geometric tiles along the edge of a fountain basin with carved white stone",
    "img_014": "white tiles with dark green and maroon star motifs below a green and red zigzag border band",
    "img_015": "white and dark green tile panel with a repeating small diamond and star pattern inside a thin border",
    "img_017": "fountain wall with a central black and white diamond pattern panel flanked by green tiled benches",
    "img_018": "multicolored geometric floor tiles and a blue and white tile border beneath a yellow carved wooden window",
    "img_020": "carved stone arched niche with a blue and brown tiled fountain panel and a stone basin",
    "img_022": "stone wall with carved arches and a blue tiled fountain niche above a white basin",
    "img_024": "blue tiled fountain wall with three arched niches and fine geometric patterns",
    "img_028": "green, black, blue and white geometric tile band at the base of a carved stucco column",
    "img_029": "terracotta orange, cream, dark blue and black interlaced star pattern with a central green star",
    "img_031": "large circular rosette pattern in blue, turquoise, white and brown with concentric star rings",
    "img_032": "multicolored horizontal bands of orange, purple, yellow and white tiles with star rosettes and diamond borders",
    "img_035": "orange, dark blue, black, green and white interlocking repeating multi-pointed star pattern",
    "img_038": "dense mosaic panel with red, blue and yellow rosettes on a dark green and black background next to a blue and white border",
    "img_039": "large blue and white radiating star at the center surrounded by turquoise, brown and black star tiles",
    "img_040": "tall panels of blue, black and cream tilework with circular rosettes beside a wooden studded door",
    "img_042": "blue tilework with a fine geometric pattern framing a large arched brass door",
    "img_043": "teal and green tile lattice pattern with repeating motifs on a beige minaret wall above small arched windows",
    "img_044": "white interlaced star network with mustard yellow, teal, blue and black tiles",
}
EXCLUDE = ["img_008", "img_011", "img_021", "img_025", "img_030",
           "img_034", "img_036", "img_037", "img_041", "img_045"]

prefix = load_yaml("configs/data.yaml")["caption_prefix"]
meta = du.load_meta()
ids = set(meta["id"])
bad = [i for i in list(CAPTIONS) + EXCLUDE if i not in ids]
if bad:
    sys.exit(f"Identifiants inconnus : {bad}")
for i, text in CAPTIONS.items():
    n = len(text.split())
    assert 4 <= n <= 30, (i, n)
    m = meta["id"] == i
    meta.loc[m, "caption"] = prefix + text
    meta.loc[m, "reviewed"] = "1"
    meta.loc[m, "status"] = ""
for i in EXCLUDE:
    m = meta["id"] == i
    meta.loc[m, "status"] = "excluded"
    meta.loc[m, "reviewed"] = "0"
du.save_meta(meta)
print(f"{len(CAPTIONS)} légendes appliquées, {len(EXCLUDE)} images exclues.")
print("Ensuite : python -m src.data_utils prepare && python -m src.data_utils check")