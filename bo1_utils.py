"""Préparation + entraînement final du modèle BO1 (turnover). Reprend exactement la préparation du notebook."""
import numpy as np, pandas as pd, joblib
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve

CONST = ['EmployeeCount', 'Over18', 'StandardHours']
BEST = dict(n_estimators=300, min_samples_split=2, min_samples_leaf=4, max_features='log2', max_depth=20)
TARGET_RECALL = 0.70
# Résultats de validation (notebook BO1_Reduire_Turnover_corrige, section 9)
CV_METRICS = {'recall': (0.713, 0.072), 'precision': (0.402, 0.041), 'f1': (0.512, 0.035)}

def clean(df):
    df = df.copy()
    df['BusinessTravel'] = df['BusinessTravel'].astype(str).str.strip().str.upper().map(
        {'TRAVEL_RARELY': 'Travel_Rarely', 'TRAVEL_FREQUENTLY': 'Travel_Frequently', 'NON-TRAVEL': 'Non-Travel'})
    df['OverTime'] = df['OverTime'].astype(str).str.strip().str.upper().map({'YES': 'Yes', 'NO': 'No'})
    df.loc[(df['Age'] < 18) | (df['Age'] > 70), 'Age'] = np.nan
    df.loc[df['DistanceFromHome'] > 100, 'DistanceFromHome'] = np.nan
    df.loc[df['MonthlyIncome'] < 0, 'MonthlyIncome'] = np.nan
    return df

def _features(df):
    return df.drop(columns=['Attrition', 'EmployeeNumber'] + CONST, errors='ignore')

def transform(df, prep):
    X = _features(clean(df)).fillna(prep['fills'])
    X = pd.get_dummies(X, columns=prep['cat_cols'])
    return X.reindex(columns=prep['columns'], fill_value=0).astype(float)

def train(csv_path='data/Employee_Attrition_non_prepare.csv'):
    raw = pd.read_csv(csv_path, encoding='utf-8-sig').drop_duplicates()
    d = clean(raw)
    F = _features(d)
    cat_cols = F.select_dtypes(include='object').columns.tolist()
    fills = {c: (F[c].mode()[0] if c in cat_cols else float(F[c].median())) for c in F.columns}
    prep = {'fills': fills, 'cat_cols': cat_cols, 'columns': None}
    X = pd.get_dummies(F.fillna(fills), columns=cat_cols, drop_first=True).astype(float)
    prep['columns'] = X.columns.tolist()
    y = (d['Attrition'].astype(str).str.strip().str.title() == 'Yes').astype(int)
    pipe = ImbPipeline([('smote', SMOTE(random_state=42)), ('rf', RandomForestClassifier(random_state=42, n_jobs=1, **BEST))])
    oof = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method='predict_proba', n_jobs=-1)[:, 1]
    p, r, s = precision_recall_curve(y, oof)
    thr = float(s[np.where(r[:-1] >= TARGET_RECALL)[0][-1]])
    pipe.fit(X, y)
    art = {'pipeline': pipe, 'threshold': thr, 'prep': prep, 'n_train': len(X), 'prevalence': float(y.mean()),
           'importances': pd.Series(pipe.named_steps['rf'].feature_importances_, index=X.columns).sort_values(ascending=False).head(15)}
    return art

def predict(art, df):
    proba = art['pipeline'].predict_proba(transform(df, art['prep']))[:, 1]
    return proba, (proba >= art['threshold']).astype(int)

if __name__ == '__main__':
    art = train(); joblib.dump(art, 'model_bo1.joblib')
    print('seuil', round(art['threshold'], 3), '| lignes', art['n_train'], '| variables', len(art['prep']['columns']))
