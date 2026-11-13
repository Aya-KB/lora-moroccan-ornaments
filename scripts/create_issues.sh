#!/usr/bin/env bash
# Crée les issues de départ avec la CLI GitHub (gh auth login au préalable).
set -euo pipefail
mk() { gh issue create --title "$1" --body "$2" --label "$3"; }
gh label create data --color 0E8A16 2>/dev/null || true
gh label create training --color 1D76DB 2>/dev/null || true
gh label create evaluation --color D93F0B 2>/dev/null || true
gh label create demo --color 5319E7 2>/dev/null || true
gh label create report --color FBCA04 2>/dev/null || true
mk "Collecter 20-50 images autorisées (Commons / photos)" "Voir configs/data.yaml. Critère : licences vérifiées." data
mk "Relire et corriger toutes les légendes" "reviewed=1 dans data/metadata.csv." data
mk "Remplir docs/DATA_CARD.md" "Effectifs, biais, limites." data
mk "Baseline SD1.5 + première LoRA (r=8, 400 pas)" "Notebooks 01-02." training
mk "Grille rangs 4/16/64 et run long" "configs/experiments.yaml." training
mk "Évaluation : métriques + grilles" "Notebook 03, results/." evaluation
mk "Analyse du surapprentissage et choix du checkpoint" "copy_rate, clip_t_ood." evaluation
mk "Démo Gradio + vidéo courte" "demo/app.py." demo
mk "Rédiger le rapport (6-8 pages)" "report/rapport.md." report
mk "Préparer la soutenance (10 min)" "docs/SOUTENANCE.md." report
