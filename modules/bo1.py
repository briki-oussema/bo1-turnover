"""BO1 · Réduire le turnover : classification du risque de départ (Briki Oussema)."""
import joblib, pandas as pd, streamlit as st
import bo1_utils as U

DATA = 'data/Employee_Attrition_non_prepare.csv'
MAIN = ['OverTime', 'JobRole', 'BusinessTravel', 'MonthlyIncome', 'Age', 'DistanceFromHome', 'YearsAtCompany',
        'YearsSinceLastPromotion', 'JobSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction', 'MaritalStatus']

@st.cache_resource
def load_model():
    try:
        return joblib.load('models/model_bo1.joblib')
    except Exception:          # versions de bibliothèques différentes : on ré-entraîne (≈ 1 min)
        return U.train(DATA)

@st.cache_data
def load_data():
    return pd.read_csv(DATA, encoding='utf-8-sig').drop_duplicates().reset_index(drop=True)

def render():
    art, raw = load_model(), load_data()
    base = U.clean(raw)
    FEATS = [c for c in raw.columns if c not in ['Attrition', 'EmployeeNumber'] + U.CONST]
    st.header("🚪 BO1 · Réduire le turnover")
    st.caption("Classifier chaque employé selon son risque de départ (DSO) · Random Forest + SMOTE · Employee Attrition")
    thr0 = art['threshold']
    thr = st.slider("Seuil de décision (BO1)", 0.05, 0.80, float(round(thr0, 2)), 0.01, key='bo1_seuil',
                    help="Plus il est bas, plus on détecte de départs (recall ↑) mais plus il y a de fausses alertes (precision ↓).")
    st.caption(f"Seuil recommandé : {thr0:.2f} (recall visé ≥ 0,70). Outil d'aide à la décision RH : un score n'est pas une certitude.")

    def verdict(p):
        if p >= thr: st.error(f"⚠️ **Employé à risque** : probabilité estimée de départ **{p:.0%}**")
        else: st.success(f"✅ **Risque faible** : probabilité estimée de départ **{p:.0%}**")
        st.progress(min(float(p), 1.0), text=f"Score {p:.2f} · seuil {thr:.2f}")

    t1, t2, t3, t4 = st.tabs(["👤 Employé existant", "🆕 Nouveau profil", "📄 Lot (CSV)", "ℹ️ À propos du modèle"])
    with t1:
        num = st.selectbox("Numéro d'employé", sorted(raw['EmployeeNumber'].unique()), key='bo1_emp')
        row = raw[raw['EmployeeNumber'] == num].head(1)
        p, _ = U.predict(art, row); verdict(p[0])
        st.dataframe(row.drop(columns=['EmployeeNumber'] + U.CONST).T.rename(columns={row.index[0]: 'valeur'}).astype(str), height=300)
        if 'Attrition' in row: st.caption(f"Valeur réelle dans les données : Attrition = {row['Attrition'].iloc[0]}")
    with t2:
        st.write("Renseignez le profil (les autres variables prennent la valeur typique).")
        vals = {}
        def widget(c, container):
            s = raw[c]
            if s.dtype == object or str(s.dtype) == 'str':
                opts = sorted(base[c].dropna().unique()); d = art['prep']['fills'].get(c, opts[0])
                vals[c] = container.selectbox(c, opts, index=opts.index(d) if d in opts else 0, key='bo1_' + c)
            else:
                lo, hi = float(base[c].min()), float(base[c].max()); d = float(art['prep']['fills'].get(c, base[c].median()))
                vals[c] = container.number_input(c, lo, hi, d, step=1.0, key='bo1_' + c)
        cols = st.columns(3)
        for i, c in enumerate([c for c in MAIN if c in FEATS]): widget(c, cols[i % 3])
        with st.expander("Autres variables"):
            cols2 = st.columns(3)
            for i, c in enumerate([c for c in FEATS if c not in MAIN]): widget(c, cols2[i % 3])
        if st.button("Calculer le risque", type="primary", key='bo1_calc'):
            p, _ = U.predict(art, pd.DataFrame([vals])); verdict(p[0])
    with t3:
        st.write("Déposez un CSV avec les mêmes colonnes que le fichier Employee Attrition (colonne `Attrition` facultative).")
        up = st.file_uploader("Fichier CSV", type="csv", key='bo1_up')
        if up is not None:
            df = pd.read_csv(up, encoding='utf-8-sig')
            miss = [c for c in FEATS if c not in df.columns]
            if miss: st.error(f"Colonnes manquantes : {miss}")
            else:
                p, _ = U.predict(art, df); out = df.copy(); out['risque_depart'] = p.round(3); out['a_risque'] = (p >= thr).astype(int)
                st.metric("Employés à risque", f"{int(out['a_risque'].sum())} / {len(out)}")
                st.dataframe(out.sort_values('risque_depart', ascending=False).head(50))
                st.download_button("Télécharger les scores", out.to_csv(index=False).encode('utf-8'), "scores_turnover.csv", key='bo1_dl')
    with t4:
        r, pr, f = U.CV_METRICS['recall'], U.CV_METRICS['precision'], U.CV_METRICS['f1']
        c1, c2, c3 = st.columns(3)
        c1.metric("Recall (validation croisée)", f"{r[0]:.2f} ± {r[1]:.2f}"); c2.metric("Precision", f"{pr[0]:.2f} ± {pr[1]:.2f}"); c3.metric("F1", f"{f[0]:.2f} ± {f[1]:.2f}")
        st.markdown(f"""
**Modèle** : Random Forest (300 arbres) précédé de SMOTE, entraîné sur {art['n_train']} employés (16 % de départs), {len(art['prep']['columns'])} variables après encodage.
**Seuil** : choisi par validation croisée pour garantir un recall ≥ 0,70, sans utiliser de jeu de test.

**Limites** : à ce niveau de recall, environ 6 employés signalés sur 10 ne partiraient pas (precision ≈ 0,40) ; données d'exemple (jeu pédagogique IBM) ; le score aide à prioriser des entretiens de rétention, il ne doit pas servir à sanctionner.""")
        st.subheader("Variables les plus importantes")
        st.bar_chart(art['importances'].sort_values())
