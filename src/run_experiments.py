"""Lance la liste de runs de configs/experiments.yaml, chacun dans un sous-processus
(libère proprement la mémoire GPU). Les runs déjà terminés sont ignorés.

    python -m src.run_experiments --experiments configs/experiments.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from .utils import load_yaml, resolve


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--config", default="configs/train_base.yaml")
    ap.add_argument("--only", nargs="*", help="noms de runs à lancer")
    a = ap.parse_args(argv)
    base = load_yaml(a.config)
    for run in load_yaml(a.experiments)["runs"]:
        name = run.get("run_name") or f"r{run.get('rank', base['rank'])}_s{run.get('steps', base['steps'])}"
        if a.only and name not in a.only:
            continue
        if (resolve(base["output_root"]) / name / "run_info.json").exists():
            print(f"== {name} déjà terminé, ignoré")
            continue
        cmd = [sys.executable, "-m", "src.train_lora", "--config", a.config, "--run_name", name]
        for k, v in run.items():
            if k != "run_name":
                cmd += [f"--{k}", str(v)]
        print("==", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=resolve("."))


if __name__ == "__main__":
    main()
