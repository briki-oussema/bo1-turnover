"""
MODÈLE À COPIER : copie ce fichier en modules/boN.py (N = ton numéro de BO : bo2.py, bo3.py, ...).
Garde la fonction render() : c'est elle que l'application appelle pour afficher ton onglet.

RÈGLES (pour ne pas gêner les autres) :
  - ne touche qu'à TES fichiers : modules/boN.py, models/model_boN.joblib, data/boN_*.csv
  - n'appelle JAMAIS st.set_page_config() et n'écris pas dans st.sidebar
  - mets le préfixe boN_ devant chaque key= de widget (ex. key='bo2_seuil')
  - chemins relatifs depuis la racine du projet (ex. 'data/bo2_employees.csv')
  - si ton modèle ne se charge pas (autre version de scikit-learn), ré-entraîne-le dans le except
"""
import joblib, pandas as pd, streamlit as st

N = 2                                   # <- ton numéro de BO
TITLE = f"BO{N} · Titre de ton objectif métier"


@st.cache_data
def load_data():
    return pd.read_csv(f'data/bo{N}_donnees.csv')           # <- ton fichier de données


@st.cache_resource
def load_model():
    try:
        return joblib.load(f'models/model_bo{N}.joblib')    # <- ton modèle sauvegardé avec joblib.dump(...)
    except Exception:
        return None                                         # ou ré-entraîne ici ton modèle


def render():
    st.header(TITLE)
    st.caption("Objectif métier (BO) → objectif Data Science (DSO) · technique utilisée")

    t1, t2, t3 = st.tabs(["🔮 Démo", "📊 Métriques", "ℹ️ À propos"])
    with t1:
        st.write("Formulaire ou sélection, puis résultat du modèle (score, groupe, règles, prévision...).")
        # exemple : x = st.number_input("Variable", key=f"bo{N}_x"); st.write(modele.predict(...))
    with t2:
        st.write("Les métriques adaptées à ta technique (voir le tableau du README_EQUIPE).")
        # st.metric("Silhouette", "0.31"); st.dataframe(tableau_metriques)
    with t3:
        st.markdown("**Modèle** : ... · **Hyperparamètres** : ... · **Limites** : ...")
