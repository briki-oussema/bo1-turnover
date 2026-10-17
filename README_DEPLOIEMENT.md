# BO1 – Réduire le turnover : application de déploiement

Contenu : `app.py` (application Streamlit) · `bo1_utils.py` (préparation + entraînement) · `model_bo1.joblib` (modèle final + seuil) · `data/` · `requirements.txt`

## Lancer en local
    pip install -r requirements.txt
    streamlit run app.py
Ouvrir http://localhost:8501

## Mettre en ligne gratuitement (Streamlit Community Cloud)
1. Créer un dépôt GitHub public et y déposer **tout le contenu de ce dossier**.
2. Aller sur https://share.streamlit.io → « New app » → choisir le dépôt, branche `main`, fichier `app.py`.
3. Cliquer sur « Deploy » : un lien public est fourni (à mettre dans la présentation).

Si le modèle sauvegardé est incompatible avec la version de scikit-learn du serveur, l'application ré-entraîne automatiquement le modèle (environ 1 minute au premier démarrage). Pour ré-entraîner à la main : `python bo1_utils.py`.

## Ce que fait l'application
- **Employé existant** : score de risque d'un employé du jeu de données.
- **Nouveau profil** : formulaire (12 variables principales + les autres en option).
- **Lot (CSV)** : scores pour un fichier entier + téléchargement (essayer `data/exemple_lot_60_employes.csv`).
- **À propos** : performances validées, limites, variables importantes. Le seuil de décision est réglable.

## Performances (validation croisée répétée sur 1 470 employés)
Recall 0,71 ± 0,07 · Precision 0,40 ± 0,04 · F1 0,51 ± 0,04 (seuil choisi sans utiliser de jeu de test).
