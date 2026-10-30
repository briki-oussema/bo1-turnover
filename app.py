"""Application d'équipe : une page d'accueil + un module par BO.
Le travail de chaque membre reste dans modules/boN.py (fonction render()) ; ce fichier
ne fait que l'habillage (accueil, navigation, thème) et n'appelle que mod.render()."""
import importlib, traceback, streamlit as st

st.set_page_config(page_title="RH Analytics · 6 objectifs métier", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ équipe & contenu (habillage uniquement)
ROSTER = {1: ( 'Réduire le turnover', 'Briki Oussema'),
          2: ( 'Prévenir le burn-out', 'Mahmoud Mabrouk'),
          3: ( 'Planifier les promotions', 'Loua Cherif'),
          4: ('Cibler les viviers de recrutement', 'Ahmed Ghabri'),
          5: ('Garantir l’équité salariale', 'Sarra Ghedir'),
          6: ('Optimiser le temps de travail', 'Med Tayeb El Elmi')}

PITCH = {1: "Classer chaque employé selon son risque de départ, pour agir avant qu'il parte.",
         2: "Anticiper le risque de burn-out des équipes et prévenir l'épuisement.",
         3: "Estimer le délai avant la prochaine promotion de chaque employé.",
         4: "Cibler les viviers de recrutement les plus prometteurs.",
         5: "Mesurer, expliquer et localiser les écarts de rémunération femmes / hommes.",
         6: "Optimiser la répartition du temps de travail des équipes."}

# ------------------------------------------------------------------ style
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
.block-container {padding-top:2.2rem; max-width:1180px;}
section[data-testid="stSidebar"] {min-width:270px;}
.hero{background:linear-gradient(120deg,#4f46e5 0%,#0ea5e9 100%);border-radius:18px;
      padding:30px 38px;color:#fff;margin-bottom:22px;box-shadow:0 10px 30px rgba(79,70,229,.25);}
.hero h1{margin:0 0 8px 0;font-size:2.05rem;font-weight:800;color:#fff;line-height:1.15;}
.hero p{margin:0;font-size:1.06rem;opacity:.96;max-width:760px;line-height:1.5;}
.hero .src{margin-top:14px;font-size:.84rem;opacity:.85;}
.badge{display:inline-block;padding:2px 11px;border-radius:999px;font-size:.72rem;font-weight:700;letter-spacing:.2px;}
.b-ok{background:#dcfce7;color:#166534;}
.b-wip{background:#fef9c3;color:#854d0e;}
.b-err{background:#fee2e2;color:#991b1b;}
div[data-testid="stMetric"]{background:rgba(127,127,127,.06);border:1px solid rgba(127,127,127,.16);
      border-radius:14px;padding:14px 16px;}
.stButton>button{border-radius:10px;font-weight:600;}
.step-n{font-size:1.7rem;font-weight:800;color:#4f46e5;line-height:1;}
.foot{color:#6b7280;font-size:.84rem;border-top:1px solid rgba(127,127,127,.2);
      margin-top:30px;padding-top:14px;line-height:1.7;}
h3 .emoji{font-size:1.3rem;}
</style>
""", unsafe_allow_html=True)

BADGE = {'✅': '<span class="badge b-ok">● En ligne</span>',
         '🚧': '<span class="badge b-wip">○ En cours</span>',
         '❌': '<span class="badge b-err">× Erreur</span>'}


# ------------------------------------------------------------------ statut des modules (inchangé)
@st.cache_data(show_spinner=False)
def status(n):
    try:
        importlib.import_module(f'modules.bo{n}'); return '✅'
    except ModuleNotFoundError as ex:
        return '🚧' if ex.name == f'modules.bo{n}' else '❌'
    except Exception:
        return '❌'


STATUS = {n: status(n) for n in ROSTER}
PRETS = sum(1 for s in STATUS.values() if s == '✅')


def aller(dest):
    st.session_state.nav = dest


# ------------------------------------------------------------------ rendu d'un BO (logique d'origine conservée)
def render_bo(n):
    emoji, titre, auteur = ROSTER[n]
    st.markdown(f"<span class='badge b-ok'>BO{n}</span>  ·  **{auteur}**", unsafe_allow_html=True)
    try:
        mod = importlib.import_module(f'modules.bo{n}')
    except ModuleNotFoundError as ex:
        if ex.name == f'modules.bo{n}':
            st.info(f"🚧 **BO{n} · {titre}** : en cours de réalisation par **{auteur}** "
                    f"(fichier `modules/bo{n}.py` pas encore déposé).")
        else:
            st.error(f"BO{n} : bibliothèque manquante `{ex.name}`. Demander son ajout dans requirements.txt.")
        return
    except Exception:
        st.error(f"❌ BO{n} : erreur au chargement du module (les autres pages ne sont pas touchées).")
        st.code(traceback.format_exc()); return
    try:
        mod.render()
    except Exception:
        st.error(f"❌ BO{n} : erreur pendant l'affichage (les autres pages ne sont pas touchées).")
        st.code(traceback.format_exc())


# ------------------------------------------------------------------ page d'accueil
def render_home():
    st.markdown(
        "<div class='hero'><h1>📊 RH Analytics · 6 objectifs métier</h1>"
        "<p>De la donnée à la décision RH : six modèles de Machine Learning, méthode CRISP-DM, "
        "réunis dans une seule application en ligne. Chaque objectif métier répond à une question "
        "concrète des ressources humaines.</p>"
        "<div class='src'>Jeux de données : IBM Employee Attrition (BO1–BO3) · Adult Census 1994 (BO4–BO6)</div></div>",
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Objectifs métier", "6")
    c2.metric("Outils en ligne", "6 / 6" )
    c3.metric("Méthodologie", "CRISP-DM")
    c4.metric("Membres de l'équipe", str(len(ROSTER)))

    st.markdown("### Les objectifs métier")
    st.caption("Cliquez sur un objectif disponible pour ouvrir son outil interactif.")
    items = list(ROSTER.items())
    for row in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (n, (emoji, titre, auteur)) in zip(cols, items[row:row + 3]):
            with col.container(border=True):
                st.markdown(f"#### <span class='emoji'>{emoji}</span> BO{n}", unsafe_allow_html=True)
                st.markdown(f"**{titre}**  &nbsp; {BADGE[STATUS[n]]}", unsafe_allow_html=True)
                st.caption(PITCH[n])
                st.caption(f"👤 {auteur}")
                if STATUS[n] == '✅':
                    st.button("Ouvrir l'outil  →", key=f"open{n}", type="primary",
                              use_container_width=True, on_click=aller, args=(n,))
                elif STATUS[n] == '🚧':
                    st.button("Bientôt disponible", key=f"open{n}", disabled=True, use_container_width=True)
                else:
                    st.button("Indisponible", key=f"open{n}", disabled=True, use_container_width=True)

    st.markdown("### Comment ça marche")
    etapes = [("1", "Choisir l'objectif", "Sélectionnez un objectif métier dans le menu de gauche ou sur une carte."),
              ("2", "Charger ou saisir les données", "Employé existant, nouveau profil, ou import d'un fichier CSV."),
              ("3", "Lire le verdict et exporter", "Score, métriques et explications clairs, puis export du rapport.")]
    cols = st.columns(3)
    for col, (num, titre, txt) in zip(cols, etapes):
        with col.container(border=True):
            st.markdown(f"<span class='step-n'>{num}</span>", unsafe_allow_html=True)
            st.markdown(f"**{titre}**")
            st.caption(txt)

    equipe = " · ".join(f"{a} (BO{n})" for n, (_, _, a) in ROSTER.items())
    st.markdown(
        f"<div class='foot'><b>Équipe</b> — {equipe}<br>"
        "Projet pédagogique de Machine Learning (CRISP-DM). Outils d'aide à la décision RH : "
        "un score ou un écart est un signal à examiner, jamais une décision individuelle automatique.</div>",
        unsafe_allow_html=True)


# ------------------------------------------------------------------ navigation (barre latérale)
with st.sidebar:
    st.markdown("### 📊 RH Analytics")
    st.caption("6 objectifs métier · CRISP-DM")
    st.divider()
    options = ['home'] + list(ROSTER)

    def fmt(o):
        if o == 'home':
            return '🏠   Accueil'
        emoji, titre, _ = ROSTER[o]
        return f'{emoji}   BO{o} · {titre}  {STATUS[o]}'

    st.radio("Navigation", options, format_func=fmt, key='nav', label_visibility='collapsed')
    st.divider()
    st.caption("✅ en ligne · 🚧 en cours · ❌ erreur")

choix = st.session_state.get('nav', 'home')
if choix == 'home':
    render_home()
else:
    render_bo(choix)
