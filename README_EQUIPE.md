# Projet RH & marché du travail : 6 BO, une seule application

Chaque membre travaille **sur son PC**, dans **ses propres fichiers**. L'application affiche un onglet par BO ;
un BO sans fichier affiche « en cours », et une erreur dans un module n'affecte jamais les autres onglets.

| BO | Membre | Fichier à créer | Technique | Métriques à afficher |
|----|--------|-----------------|-----------|----------------------|
| 1 | Briki Oussema | modules/bo1.py (déjà fait) | Classification (Random Forest) | recall, precision, F1, AUC |
| 2 | Mahmoud Mabrouk | modules/bo2.py | Détection d'anomalies | part d'anomalies, lien avec OverTime / Attrition |
| 3 | Loua Cherif | modules/bo3.py | Analyse de survie | C-index, courbes de Kaplan-Meier |
| 4 | Ahmed Ghabri | modules/bo4.py | Clustering | silhouette, inertie, profil des groupes |
| 5 | Sarra Ghedir | modules/bo5.py | Règles d'association | support, confiance, lift |
| 6 | Med Tayeb El Elmi | modules/bo6.py | Régression | MAE, RMSE, R² |

## Étape 0 : une seule fois (par Briki, propriétaire du dépôt)
1. Dépôt GitHub → **Settings** → **Collaborators** → **Add people** → saisir le pseudo GitHub de chaque ami.
2. Chaque ami **accepte l'invitation** reçue par e-mail.

## Étape 1 : préparer son PC (chaque membre)
1. Ouvrir le dépôt → bouton vert **Code** → **Download ZIP** → extraire le dossier.
2. Ouvrir un terminal dans ce dossier et lancer : `pip install -r requirements.txt`
3. Lancer l'application pour voir le résultat : `streamlit run app.py` (s'ouvre sur http://localhost:8501).

## Étape 2 : créer son module
1. Copier `modules/_modele_bo.py` en `modules/boN.py` (N = son numéro de BO), puis changer `N = ...` dans le fichier.
2. Développer dans son notebook, puis **sauvegarder le modèle** : `joblib.dump(modele, 'models/model_boN.joblib')`.
3. Mettre ses données dans `data/boN_nom.csv` (préfixe boN_ pour ne pas écraser un autre fichier).
4. Compléter `render()` : démo, métriques, limites. Tester avec `streamlit run app.py` (sauvegarder le fichier suffit à rafraîchir).

## Étape 3 : envoyer son travail sur GitHub (sans toucher aux fichiers des autres)
1. Dans le dépôt, **ouvrir le dossier `modules`** → **Add file** → **Upload files** → glisser **son** `boN.py` → **Commit changes**.
2. Idem dans le dossier `models` (son `model_boN.joblib`) et dans `data` (son `boN_nom.csv`).
3. Message de commit conseillé : `BO4 - Ahmed`. L'application en ligne se met à jour toute seule en 1 à 2 minutes ; vérifier son onglet.

## Règles à respecter
- Ne **jamais** modifier `app.py`, `requirements.txt`, `bo1_utils.py` ni les fichiers des autres.
- Pas de `st.set_page_config()` ni de `st.sidebar` dans son module. Préfixer chaque `key=` de widget par `boN_`.
- Chemins relatifs depuis la racine du projet (`data/...`, `models/...`).
- Fichier > 25 Mo : impossible à importer par le navigateur ; réduire (échantillon des données, modèle plus petit).
- Bibliothèque manquante (hors pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, imbalanced-learn, lifelines, mlxtend) : le dire à Briki.

## Si quelque chose ne marche pas
- Son onglet affiche une erreur rouge avec le détail : copier ce texte pour demander de l'aide. Les autres onglets restent intacts.
- Page blanche après l'envoi : menu de l'application → **Reboot app**.
