---
title: "Adapter un modèle texte-image avec LoRA : le cas du zellige marocain"
author: "Noms des membres de l'équipe"
date: "2026"
---

> Gabarit de 6 à 8 pages. Remplacez tout texte entre *[crochets]* par vos résultats réels (`results/`). Ne rien affirmer qui ne soit pas mesuré.

## 1. Introduction (≈ 0,5 page)
Contexte (modèles texte-image, coût du réentraînement), problème (spécialiser sur un domaine avec peu de données et 12 Go de VRAM), objectifs et questions de recherche :
(Q1) quel rang LoRA ? (Q2) combien d'étapes avant le surapprentissage ? (Q3) quel coût ? (Q4) la LoRA respecte-t-elle encore le texte ?

## 2. Méthode (≈ 1,5 page)
**LoRA.** Pour un poids gelé $W$, on apprend $\Delta W = \frac{\alpha}{r} BA$ avec $A\in\mathbb{R}^{r\times k}$, $B\in\mathbb{R}^{d\times r}$ ; seul $BA$ est entraîné. Ici sur `to_q, to_k, to_v, to_out` de l'UNet de Stable Diffusion 1.5, avec $\alpha=r$.
**Objectif d'entraînement.** Débruitage : $\mathcal{L}=\mathbb{E}\,\|\epsilon-\epsilon_\theta(z_t,t,c)\|^2$.
**Contraintes mémoire.** fp16 gelé + LoRA fp32, cache des latents/embeddings, gradient checkpointing, batch 2. Pic VRAM mesuré : *[X Go]*.

## 3. Données (≈ 1 page)
Sources, licences (tableau récapitulatif), nombre d'images *[N train / M test]*, légendes, nettoyage (doublons, recadrage), limites et biais. *Planche de 12 images.*

## 4. Protocole expérimental (≈ 1 page)
Baseline ; première LoRA ; grille rangs {4,16,64} × pas {200,400,600,800} ; run long *[2000]* pour le surapprentissage ; graines fixes ; prompts (4 in-domain, 4 nouveaux, 4 hors domaine, légendes des images de test). Mesures : CLIP-T, CLIP-I (style, test), taux de copie (seuil 0,90), CLIP-T hors domaine, temps, VRAM, paramètres.

## 5. Résultats (≈ 2 pages)
*Tableau `metrics_summary.csv` ; figures `metrics_vs_steps.png`, `training_loss.png`, `cost.png` ; grilles `grid_*.png`.*
- Effet du rang : *[…]*
- Effet du nombre de pas, apparition du surapprentissage : *[…]*
- Respect du texte et oubli hors domaine : *[…]*
- Coût : *[…]*
- Exemples visuels commentés (réussites et échecs) : *[…]*

## 6. Démonstration (≈ 0,5 page)
Application Gradio : prompt, intensité de la LoRA, comparaison avant/après (capture d'écran).

## 7. Discussion, limites et éthique (≈ 1 page)
Limites des métriques CLIP et de l'échantillon ; recadrage centré ; biais géographique ; droits des images et licences (voir `docs/ETHICS.md`) ; mémorisation ; perspectives (DoRA, plus de données, SDXL).

## 8. Conclusion (≈ 0,25 page)
Réponse aux questions Q1–Q4 et recommandation de configuration.

## Références
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022.
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021.
- Documentation Hugging Face Diffusers et PEFT.
