"""BO5 · Garantir l'équité salariale (Sarra Ghedir).
Module autonome : app.py appelle render(). Dépendances : statsmodels, mlxtend, scipy (+ pandas, numpy, streamlit)."""
import html
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from mlxtend.frequent_patterns import fpgrowth


# ------------------------------------------------------------------ données fictives
def demo_rh(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    sexe = rng.choice(["F", "H"], n, p=[0.45, 0.55])
    dep = rng.choice(["Ventes", "Technique", "Finance", "Support"], n, p=[0.3, 0.3, 0.2, 0.2])
    niveau = np.where(sexe == "F",
                      rng.choice(["Junior", "Senior", "Manager"], n, p=[0.50, 0.35, 0.15]),
                      rng.choice(["Junior", "Senior", "Manager"], n, p=[0.30, 0.40, 0.30]))
    anciennete = rng.integers(0, 21, n)
    diplome = rng.choice(["Licence", "Master", "Doctorat"], n, p=[0.5, 0.4, 0.1])
    base = pd.Series(niveau).map({"Junior": 30000, "Senior": 45000, "Manager": 65000}).values
    malus = np.where(sexe == "F", 0.95, 1.0)
    malus = np.where((sexe == "F") & (dep == "Ventes"), 0.88, malus)
    salaire = base * (1 + 0.01 * anciennete) * malus * rng.lognormal(0, 0.08, n)
    df = pd.DataFrame({"id": np.arange(1, n + 1), "sexe": sexe, "departement": dep, "niveau": niveau,
                       "anciennete": anciennete, "diplome": diplome, "salaire": salaire.round(0)})
    df.loc[rng.choice(n, 60, replace=False), "salaire"] = np.nan
    df.loc[rng.choice(n, 90, replace=False), "departement"] = np.nan
    df.loc[rng.choice(n, 8, replace=False), "anciennete"] = 45
    return pd.concat([df, df.sample(15, random_state=2)], ignore_index=True)


# ------------------------------------------------------------------ nettoyage
def nettoyer(df, col_sexe, vf, vh, col_res, mode, controles, col_id=None, seuil_rare=0.02):
    rap = {"lignes_initiales": len(df)}
    d = df.copy()
    for c in d.columns:
        if d[c].dtype == object or str(d[c].dtype) in ("str", "string"):
            s = d[c].astype("string").str.strip()
            s = s.mask(s.isin(["", "?", "nan", "NaN", "None"]))
            d[c] = s.astype(object).where(s.notna(), np.nan)

    if col_id and col_id in d.columns:
        avant = len(d)
        d = d.drop_duplicates(subset=[col_id])
        rap["doublons_id_supprimes"] = avant - len(d)
        rap["lignes_identiques_conservees"] = 0
    else:
        rap["doublons_id_supprimes"] = 0
        rap["lignes_identiques_conservees"] = int(d.duplicated().sum())

    avant = len(d)
    d = d[d[col_sexe].astype(str).isin([vf, vh])].copy()
    d[col_sexe] = d[col_sexe].astype(str)
    rap["autres_genres_exclus"] = avant - len(d)

    avant = len(d)
    if mode == "continu":
        if d[col_res].dtype == object:
            d[col_res] = d[col_res].astype(str).str.replace(r"\s", "", regex=True).str.replace(",", ".", regex=False)
        d[col_res] = pd.to_numeric(d[col_res], errors="coerce")
        d = d[d[col_res] > 0]
        y = np.log(d[col_res])
        q1, q3 = y.quantile([0.25, 0.75]); i = q3 - q1
        rap["salaires_extremes_signales"] = int(((y < q1 - 3 * i) | (y > q3 + 3 * i)).sum())
    else:
        d = d[d[col_res].notna()]
    rap["resultat_invalide_exclu"] = avant - len(d)

    cat, num, constantes = [], [], []
    for c in controles:
        if pd.api.types.is_numeric_dtype(d[c]) and d[c].nunique() > 10:
            num.append(c)
        else:
            cat.append(c)
    rap["manquants"] = {c: int(d[c].isna().sum()) for c in controles if d[c].isna().sum() > 0}
    rap["regroupees"], rap["plafonnees"] = {}, {}
    for c in cat:
        d[c] = d[c].where(d[c].notna(), "Inconnu").astype(str)
        part = d[c].value_counts(normalize=True)
        rares = list(part[part < seuil_rare].index)
        if rares:
            d[c] = d[c].where(~d[c].isin(rares), "Autre")
            rap["regroupees"][c] = rares
        if d[c].nunique() < 2:
            constantes.append(c)
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d[c] = d[c].fillna(d[c].median())
        q1, q3 = d[c].quantile([0.25, 0.75]); i = q3 - q1
        lo, hi = q1 - 1.5 * i, q3 + 1.5 * i
        n_mod = int(((d[c] < lo) | (d[c] > hi)).sum())
        if n_mod:
            d[c] = d[c].clip(lo, hi)
            rap["plafonnees"][c] = n_mod
        if d[c].std() == 0:
            constantes.append(c)
    cat = [c for c in cat if c not in constantes]
    num = [c for c in num if c not in constantes]
    rap["constantes_ignorees"] = constantes
    rap["lignes_finales"] = len(d)
    return d.reset_index(drop=True), rap, cat, num


def resume_nettoyage(rap):
    lignes = [("Lignes importées", rap["lignes_initiales"])]
    if rap["doublons_id_supprimes"]:
        lignes.append(("Doublons d'identifiant supprimés", rap["doublons_id_supprimes"]))
    elif rap["lignes_identiques_conservees"]:
        lignes.append(("Lignes identiques détectées (conservées, pas d'identifiant)", rap["lignes_identiques_conservees"]))
    if rap["autres_genres_exclus"]:
        lignes.append(("Lignes hors des deux groupes comparés (exclues)", rap["autres_genres_exclus"]))
    if rap["resultat_invalide_exclu"]:
        lignes.append(("Salaire / résultat manquant ou invalide (exclu, jamais imputé)", rap["resultat_invalide_exclu"]))
    for c, n in rap["manquants"].items():
        lignes.append((f"Valeurs manquantes dans « {c} » (remplacées)", n))
    for c, n in rap["plafonnees"].items():
        lignes.append((f"Valeurs extrêmes plafonnées dans « {c} »", n))
    for c, r in rap["regroupees"].items():
        lignes.append((f"Modalités rares regroupées en « Autre » dans « {c} »", ", ".join(map(str, r))))
    if rap.get("salaires_extremes_signales"):
        lignes.append(("Salaires très extrêmes signalés (non modifiés)", rap["salaires_extremes_signales"]))
    if rap["constantes_ignorees"]:
        lignes.append(("Variables sans variation (ignorées)", ", ".join(rap["constantes_ignorees"])))
    lignes.append(("Lignes analysées", rap["lignes_finales"]))
    return [(a, str(b)) for a, b in lignes]


# ------------------------------------------------------------------ analyses
def _design(d, col_sexe, vf, cat, num):
    parts = [(d[col_sexe] == vf).astype(float).rename("femme")]
    refs = {}
    if cat:
        X = pd.get_dummies(d[cat].astype(str), prefix_sep="=")
        keep = []
        for c in cat:
            cols = [k for k in X.columns if k.startswith(c + "=")]
            ref = X[cols].mean().idxmax()
            refs[c] = ref.split("=", 1)[1]
            keep += [k for k in cols if k != ref]
        parts.append(X[keep].astype(float))
    if num:
        Z = d[num].astype(float)
        parts.append((Z - Z.mean()) / Z.std().replace(0, 1))
    return pd.concat(parts, axis=1), refs


def analyser(d, col_sexe, vf, col_res, mode, fav, cat, num):
    X, refs = _design(d, col_sexe, vf, cat, num)
    Xc = sm.add_constant(X, has_constant="add")
    f = X["femme"].values == 1
    res = {"mode": mode, "n_f": int(f.sum()), "n_h": int((~f).sum()), "references": refs}
    if mode == "continu":
        y = np.log(d[col_res].astype(float)).values
        m = sm.OLS(y, Xc).fit(cov_type="HC3")
        b = m.params["femme"]; lo, hi = m.conf_int().loc["femme"]
        s = d[col_res].astype(float).values
        delta = y[~f].mean() - y[f].mean()
        expl = (Xc[~f].mean() - Xc[f].mean()).drop(["const", "femme"]) @ m.params.drop(["const", "femme"])
        res.update(brut_moyenne=(1 - s[f].mean() / s[~f].mean()) * 100,
                   brut_mediane=(1 - np.median(s[f]) / np.median(s[~f])) * 100,
                   ajuste=-(np.exp(b) - 1) * 100, ic=(-(np.exp(hi) - 1) * 100, -(np.exp(lo) - 1) * 100),
                   p=float(m.pvalues["femme"]),
                   part_expliquee=(float(expl / delta) if delta > 1e-9 else None))
        ci = m.conf_int()
        coef = pd.DataFrame({"effet_%": (np.exp(m.params) - 1) * 100, "IC_bas_%": (np.exp(ci[0]) - 1) * 100,
                             "IC_haut_%": (np.exp(ci[1]) - 1) * 100, "p": m.pvalues}).drop("const")
    else:
        y = (d[col_res].astype(str) == str(fav)).astype(int).values
        m = sm.Logit(y, Xc).fit(disp=0)
        b = m.params["femme"]; ci = np.exp(m.conf_int())
        expl = (Xc[~f].mean() - Xc[f].mean()).drop(["const", "femme"]) @ m.params.drop(["const", "femme"])
        tot = expl - b
        Xa = Xc.copy(); Xa.loc[f, "femme"] = 0.0
        res.update(taux_f=float(y[f].mean()), taux_h=float(y[~f].mean()),
                   ratio=float(y[f].mean() / y[~f].mean()) if y[~f].mean() > 0 else None,
                   odds_ratio=float(np.exp(b)), ic=(float(ci.loc["femme", 0]), float(ci.loc["femme", 1])),
                   p=float(m.pvalues["femme"]), part_expliquee=(float(expl / tot) if tot > 1e-9 else None),
                   proba_f=float(m.predict(Xc)[f].mean()), proba_f_contrefactuel=float(m.predict(Xa)[f].mean()))
        coef = pd.DataFrame({"odds_ratio": np.exp(m.params), "IC_bas": ci[0], "IC_haut": ci[1],
                             "p": m.pvalues}).drop("const")
    res["coefficients"] = coef.drop(index="femme", errors="ignore")
    return res


def verdict(res, seuil=5.0):
    ph = []
    if res["mode"] == "continu":
        g, (lo, hi), p = res["ajuste"], res["ic"], res["p"]
        sig = lo > 0
        ph.append(f"Les femmes gagnent en moyenne {res['brut_moyenne']:.1f} % de moins que les hommes (écart brut ; médiane : {res['brut_mediane']:.1f} %).")
        ph.append(f"À profil comparable, l'écart est de {g:.1f} % (intervalle de confiance à 95 % : {lo:.1f} % à {hi:.1f} %, p = {p:.3g}).")
        if res["part_expliquee"] is not None:
            ph.append(f"Environ {res['part_expliquee']:.0%} de l'écart brut s'explique par les différences de profil ; {1 - res['part_expliquee']:.0%} reste inexpliqué.")
        if g >= seuil and sig:
            c, t = "rouge", f"Écart ajusté de {g:.1f} %, supérieur au seuil de {seuil:g} %"
        elif sig or g >= seuil:
            c, t = "orange", f"Écart ajusté de {g:.1f} % : à surveiller" + ("" if sig else " (non significatif)")
        else:
            c, t = "vert", "Aucun écart ajusté significatif détecté"
    else:
        (lo, hi), p, r = res["ic"], res["p"], res["ratio"]
        ph.append(f"Le résultat favorable concerne {res['taux_f']:.1%} des femmes et {res['taux_h']:.1%} des hommes (ratio femmes/hommes : {r:.2f}).")
        ph.append(f"À profil comparable, les cotes des femmes sont multipliées par {res['odds_ratio']:.2f} (intervalle de confiance à 95 % : {lo:.2f} à {hi:.2f}, p = {p:.3g}).")
        if res["part_expliquee"] is not None:
            ph.append(f"Environ {res['part_expliquee']:.0%} de l'écart s'explique par les différences de profil ; {1 - res['part_expliquee']:.0%} reste inexpliqué.")
            ph.append(f"Traitées comme des hommes, les femmes passeraient de {res['proba_f']:.1%} à {res['proba_f_contrefactuel']:.1%}.")
        sig = hi < 1
        if sig and r is not None and r < 0.8:
            c, t = "rouge", f"Ratio femmes/hommes de {r:.2f} (< 0,80) et écart significatif à profil égal"
        elif sig or (r is not None and r < 0.8):
            c, t = "orange", "Écart à surveiller (brut ou ajusté)"
        else:
            c, t = "vert", "Aucun écart ajusté significatif détecté"
    return {"couleur": c, "titre": t, "phrases": ph}


def zones(d, col_sexe, vf, col_res, mode, fav, cat, num, min_support=0.05, taille_min=30, max_len=2):
    T = pd.DataFrame({c: d[c].astype(str) for c in cat})
    for c in num:
        T[c] = pd.qcut(d[c], 4, duplicates="drop").astype(str)
    if T.shape[1] == 0:
        return pd.DataFrame()
    items = pd.get_dummies(T, prefix_sep="=").astype(bool)
    freq = fpgrowth(items, min_support=min_support, use_colnames=True, max_len=max_len)
    f = (d[col_sexe] == vf).values
    y = np.log(d[col_res].astype(float)).values if mode == "continu" else (d[col_res].astype(str) == str(fav)).astype(int).values
    raw = d[col_res].astype(float).values if mode == "continu" else None
    P, idx, rows = items.values, {c: i for i, c in enumerate(items.columns)}, []
    for s in freq["itemsets"]:
        m = P[:, [idx[i] for i in s]].all(axis=1)
        nf, nh = int((m & f).sum()), int((m & ~f).sum())
        if nf < taille_min or nh < taille_min:
            continue
        if mode == "continu":
            a, b = y[m & f], y[m & ~f]
            p = stats.ttest_ind(a, b, equal_var=False).pvalue
            rows.append({"Profil": " + ".join(sorted(s)), "Effectif F": nf, "Effectif H": nh,
                         "Indicateur F": raw[m & f].mean(), "Indicateur H": raw[m & ~f].mean(),
                         "Écart": (1 - np.exp(a.mean() - b.mean())) * 100, "p": p})
        else:
            pf, ph_ = y[m & f].mean(), y[m & ~f].mean()
            pp = (y[m & f].sum() + y[m & ~f].sum()) / (nf + nh)
            se = np.sqrt(pp * (1 - pp) * (1 / nf + 1 / nh))
            z = (ph_ - pf) / se if se > 0 else 0.0
            rows.append({"Profil": " + ".join(sorted(s)), "Effectif F": nf, "Effectif H": nh,
                         "Indicateur F": pf * 100, "Indicateur H": ph_ * 100,
                         "Écart": (ph_ - pf) * 100, "p": 2 * (1 - stats.norm.cdf(abs(z)))})
    G = pd.DataFrame(rows)
    if G.empty:
        return G
    G = G.sort_values("p").reset_index(drop=True)
    n = len(G)
    G["q"] = np.minimum.accumulate((G["p"] * n / (np.arange(n) + 1))[::-1])[::-1].clip(upper=1)
    G["Significatif"] = G["q"] < 0.05
    return G.sort_values("Écart", ascending=False).reset_index(drop=True)


def par_categorie(d, col_cat, col_sexe, vf, col_res, mode, fav, cat, num, seuil, taille_min):
    rows = []
    for v, g in d.groupby(col_cat):
        nf, nh = int((g[col_sexe] == vf).sum()), int((g[col_sexe] != vf).sum())
        if nf < taille_min or nh < taille_min:
            continue
        cats = [c for c in cat if c != col_cat and g[c].nunique() > 1]
        nums = [c for c in num if g[c].std() > 0]
        try:
            r = analyser(g.reset_index(drop=True), col_sexe, vf, col_res, mode, fav, cats, nums)
            vd = verdict(r, seuil)
        except Exception:
            continue
        ico = {"rouge": "🔴", "orange": "🟠", "vert": "🟢"}[vd["couleur"]]
        if mode == "continu":
            rows.append({"Catégorie": v, "Effectif F": nf, "Effectif H": nh, "Écart brut (%)": round(r["brut_moyenne"], 1),
                         "Écart ajusté (%)": round(r["ajuste"], 1), "p": round(r["p"], 4), "Verdict": ico})
        else:
            rows.append({"Catégorie": v, "Effectif F": nf, "Effectif H": nh, "Ratio F/H": round(r["ratio"], 2),
                         "Odds ratio": round(r["odds_ratio"], 2), "p": round(r["p"], 4), "Verdict": ico})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ rapport
def rapport_html(vd, tableaux, avertissements):
    couleur = {"rouge": "#c0392b", "orange": "#d68910", "vert": "#1e8449"}[vd["couleur"]]
    h = ["<html><head><meta charset='utf-8'><title>Audit d'équité salariale</title><style>",
         "body{font-family:Arial,sans-serif;max-width:900px;margin:30px auto;color:#222}",
         "table{border-collapse:collapse;width:100%;margin:10px 0}td,th{border:1px solid #ccc;padding:5px 8px;font-size:13px}",
         f".v{{border-left:8px solid {couleur};padding:10px 15px;background:#f7f7f7}}</style></head><body>",
         "<h1>Audit d'équité salariale</h1>", f"<div class='v'><h2>{html.escape(vd['titre'])}</h2><ul>"]
    h += [f"<li>{html.escape(p)}</li>" for p in vd["phrases"]] + ["</ul></div>"]
    for titre, t in tableaux.items():
        h += [f"<h3>{html.escape(titre)}</h3>", t.to_html(index=False, float_format=lambda x: f"{x:.3g}")]
    h += ["<h3>Précautions</h3><ul>"] + [f"<li>{html.escape(a)}</li>" for a in avertissements] + ["</ul></body></html>"]
    return "".join(h)


# ------------------------------------------------------------------ interface Streamlit
ADULT_COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
              "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
SRC = ["Démonstration : RH fictif (salaire chiffré)", "Démonstration : Adult (revenu > 50K)", "Importer mon fichier"]
PRESETS = {
    SRC[0]: dict(sexe="sexe", f="F", h="H", res="salaire", mode="continu", fav=None, id="id",
                 controles=["departement", "niveau", "anciennete", "diplome"]),
    SRC[1]: dict(sexe="sex", f="Female", h="Male", res="income", mode="binaire", fav=">50K", id=None,
                 controles=["age", "race", "education", "marital-status", "occupation", "native-country", "hours-per-week"]),
    SRC[2]: {}}


@st.cache_data(show_spinner=False)
def _charger(nom):
    if nom == SRC[0]:
        return demo_rh()
    df = None
    for p in sorted((Path(__file__).resolve().parent.parent / "data").glob("*[aA]dult*")):
        try:
            t = pd.read_csv(p, skipinitialspace=True, na_values="?")
            if not {"sex", "income"} <= set(t.columns):
                t = pd.read_csv(p, names=ADULT_COLS, skipinitialspace=True, na_values="?")
            if {"sex", "income"} <= set(t.columns):
                df = t
                break
        except Exception:
            continue
    if df is None:
        df = pd.read_csv(ADULT_URL, names=ADULT_COLS, skipinitialspace=True, na_values="?")
    df.columns = df.columns.str.replace(".", "-", regex=False)
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    return df


@st.cache_data(show_spinner="Calcul de l'écart ajusté…")
def _calcul(d, col_sexe, vf, col_res, mode, fav, cat, num, seuil):
    res = analyser(d, col_sexe, vf, col_res, mode, fav, cat, num)
    return res, verdict(res, seuil)


@st.cache_data(show_spinner="Recherche des zones d'écart…")
def _calcul_zones(d, col_sexe, vf, col_res, mode, fav, cat, num, sup, taille_min, lg):
    return zones(d, col_sexe, vf, col_res, mode, fav, cat, num, sup, taille_min, lg)


def _idx(options, valeur, defaut=0):
    return options.index(valeur) if valeur in options else defaut


def render():
    st.subheader("⚖️ BO5 · Audit d'équité salariale")
    st.caption("Mesurer, expliquer et localiser les écarts de rémunération entre femmes et hommes · Régression ajustée + "
               "règles d'association (FP-Growth) · Audit de groupe, jamais de décision individuelle")
    source = st.radio("Source des données", SRC, horizontal=True, key="bo5_source")
    with st.expander("⚙️ Paramètres de l'audit"):
        p1, p2 = st.columns(2)
        seuil = p1.slider("Seuil d'alerte de l'écart ajusté (%)", 1.0, 10.0, 5.0, 0.5, key="bo5_seuil",
                          help="Au-delà de ce seuil (et si l'écart est significatif), le verdict passe au rouge. 5 % correspond au seuil de la directive européenne sur la transparence salariale, à valider avec un juriste.")
        taille_min = p2.slider("Effectif minimal par groupe", 10, 200, 30, 5, key="bo5_taille_min",
                               help="Les groupes plus petits ne sont pas analysés : résultats instables et risque de ré-identification.")
        st.caption("Outil d'audit statistique de groupe : un écart inexpliqué est un signal à examiner, pas un verdict juridique. "
                   "Les données importées sont traitées en mémoire pendant la session.")

    p, k = PRESETS[source], SRC.index(source)
    t1, t2, t3, t4, t5, t6 = st.tabs(["1 · Données", "2 · Qualité", "3 · Résultats", "4 · Zones d'écart", "5 · Rapport", "ℹ️ À propos"])

    with t1:
        if source == SRC[2]:
            up = st.file_uploader("Fichier des salariés (CSV)", type=["csv"], key="bo5_up")
            if up is None:
                st.info("Importez un fichier CSV : une ligne par salarié, avec au minimum le sexe et le salaire.")
                return
            df = pd.read_csv(up, sep=None, engine="python", encoding="utf-8-sig")
        else:
            try:
                df = _charger(source)
            except Exception as e:
                st.error(f"Impossible de charger ces données de démonstration ({type(e).__name__}). "
                         "Utilisez « RH fictif » ou importez votre fichier.")
                return
            st.caption("Données de démonstration : " + ("salariés fictifs avec un écart planté, quelques valeurs manquantes et des doublons."
                       if source == SRC[0] else "recensement américain de 1994, revenu au-dessus ou en dessous de 50K."))
        st.dataframe(df.head(8))
        st.markdown("**Correspondance des colonnes**")
        cols = list(df.columns)
        c1, c2, c3 = st.columns(3)
        col_sexe = c1.selectbox("Colonne du sexe / genre", cols, index=_idx(cols, p.get("sexe")), key=f"bo5_sexe{k}")
        valeurs = sorted(df[col_sexe].dropna().astype(str).unique())
        if len(valeurs) > 20:
            st.warning("Cette colonne a plus de 20 valeurs : vérifiez qu'il s'agit bien du sexe / genre.")
        vf = c2.selectbox("Valeur « femmes »", valeurs, index=_idx(valeurs, p.get("f")), key=f"bo5_vf{k}")
        vh = c3.selectbox("Valeur « hommes » (référence)", valeurs, index=_idx(valeurs, p.get("h"), min(1, len(valeurs) - 1)), key=f"bo5_vh{k}")
        mode_lbl = st.radio("Type de résultat à auditer", ["Salaire chiffré", "Résultat oui/non (ex. salaire élevé)"],
                            index=0 if p.get("mode", "continu") == "continu" else 1, horizontal=True, key=f"bo5_mode{k}")
        mode = "continu" if mode_lbl.startswith("Salaire") else "binaire"
        c4, c5 = st.columns(2)
        col_res = c4.selectbox("Colonne du salaire / résultat", cols, index=_idx(cols, p.get("res"), len(cols) - 1), key=f"bo5_res{k}")
        fav = None
        if mode == "binaire":
            vals_res = sorted(df[col_res].dropna().astype(str).unique())
            fav = c5.selectbox("Valeur « favorable »", vals_res, index=_idx(vals_res, p.get("fav")), key=f"bo5_fav{k}")
        ident = st.selectbox("Colonne identifiant du salarié (facultatif, pour supprimer les doublons)", ["(aucune)"] + cols,
                             index=_idx(["(aucune)"] + cols, p.get("id")), key=f"bo5_id{k}")
        autres = [c for c in cols if c not in (col_sexe, col_res)]
        controles = st.multiselect("Critères de comparaison objectifs et neutres (poste, ancienneté, diplôme, temps de travail…)",
                                   autres, default=[c for c in p.get("controles", []) if c in autres], key=f"bo5_ctr{k}")
        if not controles:
            st.warning("Sans critère de comparaison, l'écart ajusté sera égal à l'écart brut.")

    try:
        d, rap, cat, num = nettoyer(df, col_sexe, vf, vh, col_res, mode, controles, None if ident == "(aucune)" else ident)
    except Exception as e:
        t1.error(f"Impossible de préparer les données avec cette correspondance de colonnes : {e}")
        return
    nf, nh = int((d[col_sexe] == vf).sum()), int((d[col_sexe] == vh).sum())
    if vf == vh or nf < taille_min or nh < taille_min:
        t1.error(f"Effectifs insuffisants après nettoyage (femmes : {nf}, hommes : {nh}, minimum : {taille_min}).")
        return
    try:
        res, vd = _calcul(d, col_sexe, vf, col_res, mode, fav, cat, num, seuil)
    except Exception as e:
        t1.error(f"Le modèle n'a pas pu être estimé ({type(e).__name__}). Vérifiez les colonnes choisies : une variable peut séparer parfaitement les groupes.")
        return

    with t2:
        st.markdown("**Rapport de nettoyage** : chaque étape est automatique et tracée. Le salaire n'est jamais imputé : un salaire manquant exclut la ligne.")
        st.table(pd.DataFrame(resume_nettoyage(rap), columns=["Étape", "Résultat"]).set_index("Étape"))

    with t3:
        box = {"rouge": st.error, "orange": st.warning, "vert": st.success}[vd["couleur"]]
        box(f"{ {'rouge': '🔴', 'orange': '🟠', 'vert': '🟢'}[vd['couleur']] } **{vd['titre']}**")
        for ph in vd["phrases"]:
            st.markdown(f"- {ph}")
        m1, m2, m3 = st.columns(3)
        pe = "n/a" if res["part_expliquee"] is None else f"{res['part_expliquee']:.0%}"
        if mode == "continu":
            m1.metric("Écart brut (moyenne)", f"{res['brut_moyenne']:.1f} %")
            m2.metric("Écart ajusté", f"{res['ajuste']:.1f} %", help="Femmes par rapport aux hommes, à critères de comparaison égaux")
            m3.metric("Part expliquée", pe)
            st.bar_chart(pd.DataFrame({"Écart (%)": [res["brut_moyenne"], res["ajuste"]]}, index=["Brut", "Ajusté"]))
        else:
            m1.metric("Ratio femmes / hommes", f"{res['ratio']:.2f}", help="Règle des 4/5 : signal d'alerte sous 0,80")
            m2.metric("Odds ratio ajusté", f"{res['odds_ratio']:.2f}", help="1 = aucun écart, inférieur à 1 = défavorable aux femmes")
            m3.metric("Part expliquée", pe)
            st.bar_chart(pd.DataFrame({"Taux (%)": [res["taux_f"] * 100, res["taux_h"] * 100]}, index=["Femmes", "Hommes"]))
        with st.expander("Effet de chaque critère de comparaison"):
            st.write("Modalités de référence : " + ", ".join(f"{c} = {v}" for c, v in res["references"].items()))
            st.dataframe(res["coefficients"].round(3))
        st.markdown("**Par catégorie de postes** : la directive européenne apprécie l'écart par catégorie de travailleurs.")
        dec = st.selectbox("Découper par", ["(aucune)"] + cat, key=f"bo5_dec{k}")
        if dec != "(aucune)":
            pc = par_categorie(d, dec, col_sexe, vf, col_res, mode, fav, cat, num, seuil, taille_min)
            if len(pc):
                st.dataframe(pc)
            else:
                st.info("Aucune catégorie n'a assez d'effectifs pour être analysée.")

    Z = pd.DataFrame()
    with t4:
        st.write("Où l'écart se concentre-t-il ? Chaque profil (1 ou 2 caractéristiques) compare femmes et hommes. "
                 "Les écarts sont **bruts** dans le profil (les autres facteurs ne sont pas contrôlés) et testés avec une correction pour tests multiples (Benjamini-Hochberg).")
        z1, z2 = st.columns(2)
        sup = z1.slider("Support minimal du profil", 0.02, 0.30, 0.05, 0.01, key=f"bo5_sup{k}")
        lg = z2.slider("Nombre maximal de caractéristiques", 1, 3, 2, key=f"bo5_lg{k}")
        Z = _calcul_zones(d, col_sexe, vf, col_res, mode, fav, cat, num, sup, taille_min, lg)
        if Z.empty:
            st.info("Aucun profil n'a assez d'effectifs : baissez le support minimal ou l'effectif minimal.")
        else:
            tout = st.checkbox("Afficher aussi les profils sans écart significatif", key=f"bo5_tout{k}")
            aff = Z if tout else Z[Z["Significatif"]]
            unite = "salaire moyen" if mode == "continu" else "% de résultat favorable"
            st.caption(f"{len(Z)} profils testés · {int(Z['Significatif'].sum())} avec un écart significatif · "
                       f"{int(((Z['Écart'] > 0) & Z['Significatif']).sum())} défavorables aux femmes · Indicateur : {unite} · Écart : positif = défavorable aux femmes")
            st.dataframe(aff.round(3))
            st.download_button("Télécharger les zones (CSV)", Z.to_csv(index=False).encode("utf-8"), "zones_ecart.csv", key=f"bo5_dlz{k}")

    with t5:
        st.write("Rapport autonome (HTML) : ouvrez-le dans le navigateur puis imprimez-le en PDF si besoin.")
        tab = {"Nettoyage des données": pd.DataFrame(resume_nettoyage(rap), columns=["Étape", "Résultat"])}
        if not Z.empty:
            tab["Zones d'écart (10 plus fortes)"] = Z[Z["Significatif"]].head(10).round(3)
        avert = ["Audit statistique de groupe : un écart inexpliqué est un signal à examiner, pas un verdict juridique.",
                 "« Expliqué » ne veut pas dire « justifié » : l'accès aux postes ou à l'ancienneté peut lui-même être inégal.",
                 "Les critères de comparaison doivent être objectifs et neutres au regard du genre.",
                 "Ne pas utiliser ce résultat pour une décision individuelle."]
        st.download_button("Télécharger le rapport (HTML)", rapport_html(vd, tab, avert).encode("utf-8"),
                           "rapport_equite_salariale.html", "text/html", key=f"bo5_dlr{k}")
        st.markdown("**Actions recommandées**")
        if vd["couleur"] == "vert":
            st.markdown("- Aucun écart ajusté significatif : conserver le suivi annuel et documenter la méthode.")
        else:
            st.markdown("- Examiner les zones d'écart de l'onglet 4, en commençant par les effectifs les plus importants.\n"
                        "- Vérifier si l'écart s'explique par des critères objectifs et neutres (documenter la justification).\n"
                        "- Corriger les écarts non justifiés et réévaluer à la prochaine campagne salariale.")

    with t6:
        st.markdown("""
**Méthode.** Le fichier est nettoyé automatiquement (doublons, valeurs manquantes, valeurs extrêmes, modalités rares), puis :
1. **Écart brut** : différence de salaire moyen (ou de taux de résultat favorable) entre femmes et hommes.
2. **Écart ajusté** : régression (log du salaire, ou logistique si résultat oui/non) qui compare à critères égaux, avec son intervalle de confiance et sa p-value.
3. **Part expliquée / inexpliquée** : décomposition de l'écart brut selon les différences de profil.
4. **Zones d'écart** : règles d'association (FP-Growth) pour repérer les profils fréquents, puis test femmes/hommes dans chacun, avec correction pour tests multiples.

**Limites.** Les modèles sont ré-estimés sur *vos* données : les résultats de la démonstration (recensement américain de 1994) ne se transposent pas à une entreprise.
Une association n'est pas une cause, et la qualité du résultat dépend des critères de comparaison choisis.
""")
