"""Application d'équipe : un onglet par BO. Chaque membre ne dépose QUE son fichier modules/boN.py."""
import importlib, traceback, streamlit as st

st.set_page_config(page_title="RH & marché du travail · 6 BO", page_icon="📊", layout="wide")

ROSTER = {1: ('🚪', 'Réduire le turnover', 'Briki Oussema'),
          2: ('🔥', 'Prévenir le burn-out', 'Mahmoud Mabrouk'),
          3: ('🏆', 'Planifier les promotions', 'Loua Cherif'),
          4: ('🎯', 'Cibler les viviers de recrutement', 'Ahmed Ghabri'),
          5: ('⚖️', 'Garantir l’équité salariale', 'Sarra Ghedir'),
          6: ('⏱️', 'Optimiser le temps de travail', 'Med Tayeb El Elmi')}

st.title("📊 RH & marché du travail · 6 objectifs métier")
st.caption("Projet Machine Learning · CRISP-DM · Employee Attrition (BO1-BO3) et Adult Census (BO4-BO6)")

def status(n):
    try:
        importlib.import_module(f'modules.bo{n}'); return '✅'
    except ModuleNotFoundError as ex:
        return '🚧' if ex.name == f'modules.bo{n}' else '❌'
    except Exception:
        return '❌'

tabs = st.tabs([f"{e} BO{n} {status(n)}" for n, (e, _, _) in ROSTER.items()])
for tab, (n, (emoji, titre, auteur)) in zip(tabs, ROSTER.items()):
    with tab:
        try:
            mod = importlib.import_module(f'modules.bo{n}')
        except ModuleNotFoundError as ex:
            if ex.name == f'modules.bo{n}':
                st.info(f"🚧 **BO{n} · {titre}** : en cours de réalisation par **{auteur}** (fichier `modules/bo{n}.py` pas encore déposé).")
            else:
                st.error(f"BO{n} : bibliothèque manquante `{ex.name}`. Demander son ajout dans requirements.txt.")
            continue
        except Exception:
            st.error(f"❌ BO{n} : erreur au chargement du module (les autres onglets ne sont pas touchés).")
            st.code(traceback.format_exc()); continue
        try:
            mod.render()
        except Exception:
            st.error(f"❌ BO{n} : erreur pendant l'affichage (les autres onglets ne sont pas touchés).")
            st.code(traceback.format_exc())
