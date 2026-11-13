#!/usr/bin/env bash
# Initialisation locale puis publication (remplacez l'URL).
set -euo pipefail
git init -b main
git add .
git commit -m "chore: structure initiale du dépôt (src, configs, notebooks, docs)"
git remote add origin https://github.com/Aya-KB/lora-moroccan-ornaments.git
git push -u origin main
