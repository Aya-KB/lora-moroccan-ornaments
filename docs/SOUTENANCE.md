# Soutenance — 10 minutes (≈ 9 diapositives)

| Min | Diapo | Contenu |
|---|---|---|
| 0:00–0:45 | 1. Problème | Pourquoi spécialiser un modèle ? (zellige : peu de données, 12 Go) |
| 0:45–1:45 | 2. LoRA | Idée : W + (α/r)·BA, rang r, ce qui est entraîné (≈ 1 % des poids) |
| 1:45–3:00 | 3. Données | 20–50 images, licences, légendes, doublons, split test |
| 3:00–4:00 | 4. Pipeline & contraintes | fp16, cache latents, checkpointing → pic VRAM mesuré |
| 4:00–5:00 | 5. Expériences | rangs 4/16/64, pas 200–800 (+ run long) |
| 5:00–7:00 | 6. Résultats chiffrés | `metrics_vs_steps.png`, tableau coût |
| 7:00–8:00 | 7. Exemples visuels | `grid_novel.png` + 1 échec commenté |
| 8:00–9:00 | 8. Démo | prompt + intensité + avant/après (en direct ou vidéo) |
| 9:00–10:00 | 9. Limites, éthique, conclusion | surapprentissage, biais, droits, perspectives |

Chaque membre parle ~2,5 min. Prévoir une vidéo de secours de la démo (`demo/`).
