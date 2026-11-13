# Fiche du modèle — LoRA zellige marocain

- **Modèle de base** : Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`), licence CreativeML OpenRAIL-M.
- **Adaptation** : LoRA sur `to_q, to_k, to_v, to_out.0` de l'UNet ; rang(s) 4 / 16 / 64, `alpha = rang`.
- **Données d'entraînement** : voir `DATA_CARD.md` (images sous licences libres, attribution dans `data/metadata.csv`).
- **Entraînement** : AdamW, lr 1e-4, batch 2, 512×512, fp16 + LoRA fp32, graine 42 ; détails et coûts mesurés dans `outputs/<run>/run_info.json` et `results/metrics_summary.csv`.
- **Usage prévu** : illustration d'ornements et d'architecture marocains (recherche, enseignement, inspiration graphique).
- **Hors périmètre** : imitation d'artistes vivants, désinformation, fabrication de « photos » présentées comme des documents réels.
- **Limites connues** : *à compléter avec les résultats* (compositions complexes, texte/calligraphie illisibles, oubli hors domaine à forte intensité, risque de copie des images d'entraînement à grand nombre de pas).
- **Résultats** : voir `results/` et `report/rapport.md`.
- **Poids** : non versionnés dans Git ; à publier en *release* ou sur Hugging Face Hub avec cette fiche.
