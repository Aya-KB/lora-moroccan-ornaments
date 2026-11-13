# Réflexion éthique

1. **Droits des images.** Seules des images sous licence libre compatible (CC0, domaine public, CC BY, CC BY-SA) ou des photos de l'équipe sont utilisées ; chaque image est tracée dans `data/metadata.csv`. L'attribution est conservée. L'effet juridique d'un entraînement sur des images CC BY-SA (le modèle est-il une œuvre dérivée ?) reste débattu : nous mentionnons la source des données avec les poids et nous ne prétendons pas fournir un avis juridique.
2. **Artistes.** Aucun nom d'artiste vivant dans les légendes ou prompts ; un seul style et un seul concept pour éviter l'imitation ciblée.
3. **Patrimoine culturel.** Le zellige est un artisanat vivant : le modèle produit des motifs « dans l'esprit » du zellige, sans valeur d'authenticité ni de remplacement du travail des maâlems. Les motifs générés peuvent être géométriquement incorrects.
4. **Biais.** Un petit corpus issu de Wikimedia sur-représente certaines villes, couleurs et monuments célèbres ; les sorties reflètent ce biais. À documenter avec des exemples.
5. **Mémorisation.** Un petit jeu + beaucoup de pas = risque de recracher des images d'entraînement. Nous mesurons ce risque (taux de copie) et choisissons un checkpoint en dessous du seuil.
6. **Sécurité.** Le vérificateur de sécurité de Stable Diffusion est désactivé dans le pipeline (domaine inoffensif, contrôlé) ; à réactiver pour toute mise en ligne publique de la démo.
7. **Coût environnemental.** Temps GPU et énergie sont mesurés (`run_info.json`) ; nous limitons les runs à la grille nécessaire.
8. **Transparence.** Les images générées doivent être présentées comme générées par IA.
