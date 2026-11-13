# Fiche des données

| Champ | Valeur |
|---|---|
| Domaine | zellige marocain (un seul concept, un seul style) |
| Sources | Wikimedia Commons (catégories/recherches de `configs/data.yaml`), photographies de l'équipe (`data/own/`) |
| Licences admises | CC0, domaine public, CC BY, CC BY-SA (jamais NC ni ND) |
| Effectif final | *à compléter : N images valides (train / test)* — `python -m src.data_utils stats` |
| Préparation | recadrage carré centré, 512×512, suppression des doublons (dHash, distance ≤ 6), exclusion manuelle des filigranes |
| Légendes | courtes, descriptives, relues à la main ; préfixe commun « a photo of moroccan zellige tilework, » |
| Split | *n_test* images (5 à 10) jamais vues à l'entraînement, graine 42 |
| Attribution | `data/metadata.csv` : auteur, licence, URL de la page source de chaque image |
| Limites | petit échantillon ; recadrage centré qui coupe certains motifs ; photos majoritairement de villes impériales (Fès, Marrakech) → biais géographique ; *à compléter* |
| Artistes vivants | aucun nom d'artiste n'est utilisé dans les légendes ni les prompts |

Statistiques et exemples : *coller ici la sortie de `stats` et la planche du notebook 00*.
