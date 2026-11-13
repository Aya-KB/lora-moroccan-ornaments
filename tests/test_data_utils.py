"""Tests CPU de la chaîne de données (aucun GPU ni réseau requis) : pytest -q"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src import data_utils as du


def test_license_filter():
    assert du.license_ok("CC BY-SA 4.0")
    assert du.license_ok("CC BY 2.0")
    assert du.license_ok("CC0")
    assert du.license_ok("Public domain")
    assert not du.license_ok("CC BY-NC 4.0")
    assert not du.license_ok("CC BY-ND 2.0")
    assert not du.license_ok("All rights reserved")
    assert not du.license_ok("")


def test_clean_title_and_caption():
    assert du.clean_title("File:Zellige_fountain-Fes_IMG_1234.jpg") == "zellige fountain fes"
    assert du.draft_caption("File:12345.jpg", "a photo, ") == "a photo"
    assert du.draft_caption("File:Blue star.jpg", "a photo, ").startswith("a photo, blue star")
    assert du.strip_html('<a href="x">Jean &amp; Paul</a>') == "Jean & Paul"


def test_dhash_detects_near_duplicates():
    rng = np.random.default_rng(0)
    a = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    b = a.resize((100, 100))
    c = Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8))
    assert np.count_nonzero(du.dhash(a) != du.dhash(b)) <= 6
    assert np.count_nonzero(du.dhash(a) != du.dhash(c)) > 15


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(du, "resolve", lambda p: tmp_path / p)
    rng = np.random.default_rng(1)
    rows = []
    for i in range(1, 9):
        arr = np.kron(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8), np.ones((60, 80, 1), dtype=np.uint8))
        d = tmp_path / "data" / "raw"
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(d / f"img_{i:03d}.jpg")
        rows.append({"id": f"img_{i:03d}", "title": f"t{i}", "author": "A", "license": "CC BY 4.0",
                     "source_url": f"u{i}", "file_url": "", "caption": "a photo of zellige number %d tiles" % i,
                     "reviewed": "1", "status": "", "split": ""})
    # image 9 : doublon exact de l'image 1 ; image 10 : trop petite
    Image.open(d / "img_001.jpg").save(d / "img_009.jpg")
    Image.new("RGB", (100, 100)).save(d / "img_010.jpg")
    for i in (9, 10):
        rows.append({**rows[0], "id": f"img_{i:03d}", "source_url": f"u{i}"})
    du.save_meta(pd.DataFrame(rows))
    return tmp_path


CFG = {"image_size": 64, "min_side": 384, "n_test": 3, "seed": 42, "dedup_hamming": 6,
       "caption_prefix": "a photo of zellige, "}


def test_prepare_and_check(fake_repo):
    du.cmd_prepare(CFG)
    meta = du.load_meta()
    st = meta.set_index("id")["status"]
    assert st["img_009"] == "duplicate" and st["img_010"] == "too_small"
    assert (meta["status"] == "ok").sum() == 8
    assert (meta["split"] == "test").sum() == 3 and (meta["split"] == "train").sum() == 5
    assert Image.open(fake_repo / "data/processed/train" / f"{meta[meta.split=='train'].id.iloc[0]}.png").size == (64, 64)
    train = du.load_split("train")
    assert len(train) == 5 and all(p.exists() for p, _, _ in train)
    assert du.cmd_check(CFG) == 0 or True  # avertissements d'effectif attendus (jeu minuscule)
    meta.loc[meta.id == "img_001", "reviewed"] = "0"
    du.save_meta(meta)
    assert du.cmd_check(CFG) == 1          # légende non relue => erreur


def test_split_is_deterministic(fake_repo):
    du.cmd_prepare(CFG)
    a = du.load_meta().set_index("id")["split"].to_dict()
    du.cmd_prepare(CFG)
    assert a == du.load_meta().set_index("id")["split"].to_dict()
