Les images ne sont **pas** versionnées. Seul `metadata.csv` (auteur, licence, URL, légende, split) est commité.
- `raw/` : images téléchargées (ignoré) · `own/` : photos de l'équipe (à commiter, petites) · `processed/` : 512×512 train/test (ignoré)
- Recréer les données : `python -m src.data_utils fetch && python -m src.data_utils prepare`
