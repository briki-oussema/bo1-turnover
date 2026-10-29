"""BO2 · Prévenir le burn-out : détection d'anomalies non supervisée, Isolation Forest + LOF (Mahmoud Mabrouk)."""
import joblib, numpy as np, pandas as pd, streamlit as st
import burnout_utils as U

MODEL = 'models/model_bo2.joblib'
TRAVEL = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL)


def render():
    try:
        bundle = load_model()
    except Exception as ex:
        st.error(f"❌ BO2 : impossible de charger `{MODEL}` (version de scikit-learn différente ?). Détail : {ex}")
        return

    d, cat = bundle['defaults'], bundle['categories']

    st.header("🔥 BO2 · Prévenir le burn-out")
    st.caption("Détecter les employés au profil atypique de surcharge (DSO) · Isolation Forest + LOF · Employee Attrition")

    t1, t2, t3 = st.tabs(["🔮 Démo", "📊 Métriques", "ℹ️ À propos du modèle"])

    # ------------------------------------------------------------------ Démo
    with t1:
        with st.form("bo2_formulaire"):
            st.subheader("Profil de l'employé")
            c1, c2 = st.columns(2)
            with c1:
                overtime = st.radio("Heures supplémentaires", ["Non", "Oui"], horizontal=True, key="bo2_overtime")
                travel = st.selectbox("Déplacements professionnels", TRAVEL, index=1, key="bo2_travel")
                wlb = st.slider("Équilibre vie pro / perso (1 = mauvais, 4 = excellent)", 1, 4,
                                int(round(d['WorkLifeBalance'])), key="bo2_wlb")
                env = st.slider("Satisfaction de l'environnement (1 à 4)", 1, 4,
                                int(round(d['EnvironmentSatisfaction'])), key="bo2_env")
                job = st.slider("Satisfaction au travail (1 à 4)", 1, 4,
                                int(round(d['JobSatisfaction'])), key="bo2_job")
                rel = st.slider("Satisfaction des relations (1 à 4)", 1, 4,
                                int(round(d['RelationshipSatisfaction'])), key="bo2_rel")
                role = st.selectbox("Poste", cat['JobRole'], index=0, key="bo2_role")
                marital = st.selectbox("Situation familiale", cat['MaritalStatus'], index=0, key="bo2_marital")
                gender = st.radio("Genre", ["Femme", "Homme"], horizontal=True, key="bo2_gender")
            with c2:
                age = st.number_input("Âge", 18, 65, int(round(d['Age'])), key="bo2_age")
                income = st.number_input("Salaire mensuel", 1000, 20000, int(round(d['MonthlyIncome'])),
                                         step=100, key="bo2_income")
                dist = st.number_input("Distance domicile-travail (km)", 1, 50,
                                       int(round(d['DistanceFromHome'])), key="bo2_dist")
                years_co = st.number_input("Années dans l'entreprise", 0, 40,
                                           int(round(d['YearsAtCompany'])), key="bo2_years_co")
                years_promo = st.number_input("Années depuis la dernière promotion", 0, 20,
                                              int(round(d['YearsSinceLastPromotion'])), key="bo2_years_promo")
                total_years = st.number_input("Années d'expérience totales", 0, 45,
                                              int(round(d['TotalWorkingYears'])), key="bo2_total_years")
                nb_comp = st.number_input("Nombre d'employeurs précédents", 0, 15,
                                          int(round(d['NumCompaniesWorked'])), key="bo2_nb_comp")
            with st.expander("Autres informations (valeurs moyennes par défaut)"):
                involvement = st.slider("Implication dans le travail (1 à 4)", 1, 4,
                                        int(round(d['JobInvolvement'])), key="bo2_involvement")
                stock = st.slider("Niveau de stock-options (0 à 3)", 0, 3,
                                  int(round(d['StockOptionLevel'])), key="bo2_stock")
                hike = st.number_input("Augmentation de salaire (%)", 0, 40,
                                       int(round(d['PercentSalaryHike'])), key="bo2_hike")
                training = st.number_input("Formations l'an dernier", 0, 10,
                                           int(round(d['TrainingTimesLastYear'])), key="bo2_training")
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

            s_if, s_lof = [float(x[0]) for x in U.scorer(row, bundle)]
            pct_if = float((bundle['scores_ref_if'] < s_if).mean() * 100)
            pct_lof = float((bundle['scores_ref_lof'] < s_lof).mean() * 100)
            flag_if, flag_lof = s_if > bundle['seuil_if'], s_lof > bundle['seuil_lof']
            brut = U.creer_features_brutes(row).iloc[0]

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
                st.metric("Score d'anomalie", f"{s_lof:.3f}",
                          help="Environ 1 = normal ; plus il est élevé, plus le profil est isolé de ses voisins.")
                st.progress(min(pct_lof / 100, 1.0), text=f"Plus atypique que {pct_lof:.0f} % des employés")
                st.caption("Signalé" if flag_lof else "Non signalé")

            m3, m4 = st.columns(2)
            m3.metric("Indice de surcharge (1 à 8)", f"{brut['OverloadIndex']:.0f}",
                      help="Heures sup. x2 + déplacements + (5 - équilibre vie pro/perso).")
            m4.metric("Bien-être (1 à 4)", f"{brut['WellBeingScore']:.2f}", help="Moyenne des 4 satisfactions.")

            st.caption("Outil d'aide à la décision : un profil signalé n'est pas un diagnostic de burn-out. "
                       "Les modèles repèrent des profils atypiques, à examiner avec les RH.")

    # -------------------------------------------------------------- Métriques
    with t2:
        st.write("Il n'y a pas de colonne « burn-out » dans les données : precision, recall et F1 sont impossibles. "
                 "On évalue donc les modèles sur trois niveaux (validité métier, fiabilité, proxy Attrition).")
        try:
            ref_if = np.asarray(bundle['scores_ref_if']); ref_lof = np.asarray(bundle['scores_ref_lof'])
            f_if, f_lof = ref_if > bundle['seuil_if'], ref_lof > bundle['seuil_lof']
            both, either = (f_if & f_lof).sum(), (f_if | f_lof).sum()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signalés Isolation Forest", f"{int(f_if.sum())} / {len(ref_if)}")
            c2.metric("Signalés LOF", f"{int(f_lof.sum())} / {len(ref_lof)}")
            c3.metric("Signalés par les deux", f"{int(both)}")
            c4.metric("Accord (Jaccard)", f"{both / either:.2f}" if either else "n/a",
                      help="Part des employés signalés par les deux modèles parmi ceux signalés par au moins un.")
            st.caption("Calculé sur les employés du jeu d'entraînement (seuil = les 5 % les plus atypiques).")
        except Exception:
            st.info("Statistiques de comparaison indisponibles pour ce modèle.")
        st.markdown("""
**Grille d'évaluation (notebook, cellule 4.3)**
1. **Validité métier** : % de signalés en heures supplémentaires, OverloadIndex moyen, WellBeingScore moyen.
2. **Fiabilité** : stabilité (Jaccard selon les graines / valeurs de k), temps d'exécution, interprétabilité.
3. **Proxy Attrition** : taux d'attrition des signalés, lift, ROC-AUC (un ROC-AUC modeste est normal, Attrition n'est pas le burn-out).
""")

    # ------------------------------------------------------------------ À propos
    with t3:
        st.markdown(f"""
**Modèle 1 : Isolation Forest** (principal). Des coupures aléatoires isolent vite un employé atypique. Entrée : variables
préparées sans one-hot ni `Gender` (attribut sensible). Non supervisé, rapide, score explicable.

**Modèle 2 : LOF** (comparaison). Compare la densité locale d'un employé à celle de ses voisins, sur la projection PCA
(90 % de variance). Détecte les anomalies locales et valide Isolation Forest.

**Verdict** : 🔴 priorité haute = signalé par les deux modèles · 🟠 à surveiller = un seul · 🟢 dans la norme = aucun.

**Limites** : pas d'étiquette burn-out (Attrition n'est qu'un proxy) ; un profil signalé n'est pas un diagnostic ;
les variables non saisies prennent leur valeur médiane ; données d'exemple (jeu pédagogique IBM).
Modèle entraîné avec scikit-learn {bundle.get('sklearn_version', '?')}.""")
