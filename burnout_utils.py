"""Fonctions partagées entre le notebook (export) et l'application Streamlit.
Elles reproduisent EXACTEMENT la Data Preparation du notebook : standardisation, Feature Engineering, PCA."""
import pandas as pd

NEW_SCALED = ['WellBeingScore', 'OverloadIndex', 'PromotionStagnation', 'JobHoppingRate']


def creer_features_brutes(raw):
    """Les 5 variables du Feature Engineering, calculées sur les valeurs brutes (non standardisées)."""
    out = pd.DataFrame(index=raw.index)
    out['WellBeingScore'] = raw[['WorkLifeBalance', 'EnvironmentSatisfaction',
                                 'JobSatisfaction', 'RelationshipSatisfaction']].mean(axis=1)
    out['OverloadIndex'] = raw['OverTime'] * 2 + raw['BusinessTravel'] + (5 - raw['WorkLifeBalance'])
    out['PromotionStagnation'] = raw['YearsSinceLastPromotion'] / (raw['YearsAtCompany'] + 1)
    out['JobHoppingRate'] = raw['NumCompaniesWorked'] / (raw['TotalWorkingYears'] + 1)
    out['LongCommute'] = (raw['DistanceFromHome'] > 20).astype(int)
    return out


def preparer_fe(raw, bundle):
    """raw : DataFrame de valeurs brutes (mêmes colonnes que X dans le notebook).
    Retourne le jeu préparé `fe` (mêmes colonnes et même ordre qu'au notebook)."""
    fe_new = raw.copy().astype(float)
    fe_new[bundle['to_scale']] = bundle['scaler_base'].transform(raw[bundle['to_scale']])
    nouvelles = creer_features_brutes(raw).astype(float)
    nouvelles[NEW_SCALED] = bundle['scaler_new'].transform(nouvelles[NEW_SCALED])
    for c in nouvelles.columns:
        fe_new[c] = nouvelles[c]
    return fe_new[bundle['cols_fe']]


def entree_if(fe_new, bundle):
    """Entrée du Modèle 1 (Isolation Forest) = config B."""
    return fe_new[bundle['cols_B']]


def entree_lof(fe_new, bundle):
    """Entrée du Modèle 2 (LOF) = projection PCA (90 % de variance)."""
    return pd.DataFrame(bundle['pca'].transform(fe_new), columns=bundle['cols_pca'], index=fe_new.index)


def scorer(raw, bundle):
    """Retourne (score Isolation Forest, score LOF) : plus haut = plus atypique."""
    fe_new = preparer_fe(raw, bundle)
    s_if = -bundle['model_if'].score_samples(entree_if(fe_new, bundle))
    s_lof = -bundle['model_lof'].score_samples(entree_lof(fe_new, bundle))
    return s_if, s_lof
