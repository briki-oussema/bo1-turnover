import sys
from pathlib import Path

# Le dossier de ce fichier (permet de lancer l'application depuis n'importe où, ex. depuis un sous-dossier BO2/)
DOSSIER = Path(__file__).resolve().parent
sys.path.insert(0, str(DOSSIER))

import joblib
import pandas as pd
import streamlit as st
from burnout_utils import scorer, creer_features_brutes

st.set_page_config(page_title="Prévention du burn-out", page_icon="🧠", layout="centered")


@st.cache_resource
def charger():
    return joblib.load(DOSSIER / 'burnout_model.joblib')


bundle = charger()
d = bundle['defaults']
cat = bundle['categories']
TRAVEL = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]

st.title("🧠 Prévention du burn-out")
st.write("Outil d'aide à la décision pour les RH : deux modèles de détection d'anomalies "
         "(**Isolation Forest** et **LOF**) repèrent les profils de travail atypiques "
         "à partir du jeu Employee Attrition.")

with st.form("formulaire"):
    st.subheader("Profil de l'employé")
    c1, c2 = st.columns(2)
    with c1:
        overtime = st.radio("Heures supplémentaires", ["Non", "Oui"], horizontal=True)
        travel = st.selectbox("Déplacements professionnels", TRAVEL, index=1)
        wlb = st.slider("Équilibre vie pro / perso (1 = mauvais, 4 = excellent)", 1, 4, int(round(d['WorkLifeBalance'])))
        env = st.slider("Satisfaction de l'environnement (1 à 4)", 1, 4, int(round(d['EnvironmentSatisfaction'])))
        job = st.slider("Satisfaction au travail (1 à 4)", 1, 4, int(round(d['JobSatisfaction'])))
        rel = st.slider("Satisfaction des relations (1 à 4)", 1, 4, int(round(d['RelationshipSatisfaction'])))
        role = st.selectbox("Poste", cat['JobRole'], index=0)
        marital = st.selectbox("Situation familiale", cat['MaritalStatus'], index=0)
        gender = st.radio("Genre", ["Femme", "Homme"], horizontal=True)
    with c2:
        age = st.number_input("Âge", 18, 65, int(round(d['Age'])))
        income = st.number_input("Salaire mensuel", 1000, 20000, int(round(d['MonthlyIncome'])), step=100)
        dist = st.number_input("Distance domicile-travail (km)", 1, 50, int(round(d['DistanceFromHome'])))
        years_co = st.number_input("Années dans l'entreprise", 0, 40, int(round(d['YearsAtCompany'])))
        years_promo = st.number_input("Années depuis la dernière promotion", 0, 20, int(round(d['YearsSinceLastPromotion'])))
        total_years = st.number_input("Années d'expérience totales", 0, 45, int(round(d['TotalWorkingYears'])))
        nb_comp = st.number_input("Nombre d'employeurs précédents", 0, 15, int(round(d['NumCompaniesWorked'])))
    with st.expander("Autres informations (valeurs moyennes par défaut)"):
        involvement = st.slider("Implication dans le travail (1 à 4)", 1, 4, int(round(d['JobInvolvement'])))
        stock = st.slider("Niveau de stock-options (0 à 3)", 0, 3, int(round(d['StockOptionLevel'])))
        hike = st.number_input("Augmentation de salaire (%)", 0, 40, int(round(d['PercentSalaryHike'])))
        training = st.number_input("Formations l'an dernier", 0, 10, int(round(d['TrainingTimesLastYear'])))
    envoyer = st.form_submit_button("Analyser le profil")

if envoyer:
    row = pd.DataFrame([bundle['defaults']])
    row['OverTime'] = 1 if overtime == "Oui" else 0
    row['BusinessTravel'] = TRAVEL.index(travel)
    row['WorkLifeBalance'], row['EnvironmentSatisfaction'] = wlb, env
    row['JobSatisfaction'], row['RelationshipSatisfaction'] = job, rel
    row['Age'], row['MonthlyIncome'], row['DistanceFromHome'] = age, income, dist
    row['YearsAtCompany'], row['YearsSinceLastPromotion'] = years_co, years_promo
    row['TotalWorkingYears'], row['NumCompaniesWorked'] = total_years, nb_comp
    row['JobInvolvement'], row['StockOptionLevel'] = involvement, stock
    row['PercentSalaryHike'], row['TrainingTimesLastYear'] = hike, training
    row['Gender'] = 1 if gender == "Homme" else 0

    # Variables nominales : one-hot (la 1re catégorie est la référence, toutes les colonnes à 0)
    dept = bundle['role_to_dept'][role]
    for col, choix in [('JobRole', role), ('Department', dept), ('MaritalStatus', marital)]:
        for c in [c for c in row.columns if c.startswith(col + '_')]:
            row[c] = 0
        if f'{col}_{choix}' in row.columns:
            row[f'{col}_{choix}'] = 1

    s_if, s_lof = [float(x[0]) for x in scorer(row, bundle)]
    pct_if = float((bundle['scores_ref_if'] < s_if).mean() * 100)
    pct_lof = float((bundle['scores_ref_lof'] < s_lof).mean() * 100)
    flag_if, flag_lof = s_if > bundle['seuil_if'], s_lof > bundle['seuil_lof']
    brut = creer_features_brutes(row).iloc[0]

    st.subheader("Résultat")
    if flag_if and flag_lof:
        st.error("🔴 **Priorité haute** : profil atypique pour les deux modèles. À examiner par les RH.")
    elif flag_if or flag_lof:
        st.warning("🟠 **À surveiller** : profil atypique pour un seul modèle ("
                   + ("Isolation Forest" if flag_if else "LOF") + ").")
    else:
        st.success("🟢 Profil dans la norme du jeu de données.")

    a, b = st.columns(2)
    with a:
        st.markdown("**Isolation Forest**")
        st.metric("Score d'anomalie", f"{s_if:.3f}", help="Plus il est élevé, plus le profil est atypique.")
        st.progress(min(pct_if / 100, 1.0), text=f"Plus atypique que {pct_if:.0f} % des employés")
        st.caption("Signalé" if flag_if else "Non signalé")
    with b:
        st.markdown("**LOF (Local Outlier Factor)**")
        st.metric("Score d'anomalie", f"{s_lof:.3f}", help="Environ 1 = normal ; plus il est élevé, plus le profil est isolé de ses voisins.")
        st.progress(min(pct_lof / 100, 1.0), text=f"Plus atypique que {pct_lof:.0f} % des employés")
        st.caption("Signalé" if flag_lof else "Non signalé")

    m3, m4 = st.columns(2)
    m3.metric("Indice de surcharge (1 à 8)", f"{brut['OverloadIndex']:.0f}",
              help="Heures sup. x2 + déplacements + (5 - équilibre vie pro/perso).")
    m4.metric("Bien-être (1 à 4)", f"{brut['WellBeingScore']:.2f}", help="Moyenne des 4 satisfactions.")

    st.caption("Outil d'aide à la décision : un profil signalé n'est pas un diagnostic de burn-out. "
               "Les modèles repèrent des profils atypiques, à examiner avec les RH.")
