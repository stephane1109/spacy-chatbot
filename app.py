from __future__ import annotations

# =============================================================================
# Couserans — Lacs & Étangs (fuzzy avec modes de score + toponymes + fragments)
# =============================================================================

from pathlib import Path
from typing import List, Dict
import json
import re

import streamlit as st

# RapidFuzz (équivalent FuzzyWuzzy) : plusieurs scoreurs disponibles
try:
    from rapidfuzz import process, fuzz  # type: ignore
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False

# ---------------- Config ----------------
st.set_page_config(page_title="Couserans • Lacs & Étangs (Fuzzy)")
RACINE = Path(__file__).parent
CHEMIN_JSON = RACINE / "data" / "models_couserans.json"

# Règles de boost
BOOST_TOPONYME = 30.0    # si un toponyme distinctif complet est présent (mot entier)
BOOST_PREFIXE = 40.0     # si la requête est le début d’un toponyme (ex. "beth" => "Bethmale")

# Mots génériques à ignorer pour construire les toponymes
MOTS_GENERIQUES = {
    "lac", "lacs", "étang", "etang", "étangs", "etangs",
    "de", "du", "des", "la", "le", "les", "l", "d"
}

# --------------- Chargement JSON ---------------
@st.cache_resource(show_spinner=False)
def lire_json_entites(chemin_str: str) -> dict | None:
    p = Path(chemin_str)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("entities"), list):
            return data
    except Exception:
        return None
    return None

@st.cache_resource(show_spinner=False)
def construire_index_depuis_json(entites_json: str):
    """
    Produit :
      - alias_vers_canonique : alias.lower() -> canonical
      - meta_par_canonique   : infos (label, category, url)
      - tous_alias           : liste de tous les alias (canonical + aliases)
      - noms_principaux      : liste des canonical (pour l’aide sous le titre)
      - tokens_par_canonique : {canonical: set de tokens (mots) issus canonical+aliases}
    """
    data = json.loads(entites_json)

    alias_vers_canonique: Dict[str, str] = {}
    meta_par_canonique: Dict[str, Dict] = {}
    tous_alias: List[str] = []
    noms_principaux: List[str] = []
    tokens_par_canonique: Dict[str, set] = {}

    def tokeniser(s: str) -> List[str]:
        # mots alphabétiques (accents inclus), min 2 caractères
        return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{2,}", s or "")

    for ent in data.get("entities", []):
        cano = (ent.get("canonical") or "").strip()
        if not cano:
            continue

        noms_principaux.append(cano)
        meta_par_canonique[cano] = {
            "label": ent.get("label") or "LAC",
            "category": ent.get("category"),
            "url": ent.get("url"),
        }

        variantes = [cano] + list(ent.get("aliases", []) or [])
        tokset = set()
        for a in variantes:
            a = (a or "").strip()
            if not a:
                continue
            tous_alias.append(a)
            alias_vers_canonique[a.lower()] = cano
            for t in tokeniser(a):
                tl = t.lower()
                if tl not in MOTS_GENERIQUES:
                    tokset.add(tl)

        tokens_par_canonique[cano] = tokset

    tous_alias = list(dict.fromkeys(tous_alias))
    noms_principaux = list(dict.fromkeys(noms_principaux))
    return alias_vers_canonique, meta_par_canonique, tous_alias, noms_principaux, tokens_par_canonique

# --------------- Sélecteur de scoreur ---------------
def choisir_scorer(nom_mode: str):
    """
    Mappe le nom du mode vers le scoreur RapidFuzz correspondant.
    Modes proposés :
      - 'WRatio' (par défaut)
      - 'partial_ratio'
      - 'token_set_ratio'
      - 'token_sort_ratio'
    """
    table = {
        "WRatio": fuzz.WRatio,
        "partial_ratio": fuzz.partial_ratio,
        "token_set_ratio": fuzz.token_set_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
    }
    return table.get(nom_mode, fuzz.WRatio)

# --------------- Scores bruts ---------------
def calculer_scores(texte: str,
                    tous_alias: List[str],
                    alias_vers_canonique: Dict[str, str],
                    scorer) -> List[Dict]:
    """
    Calcule les scores de similarité entre le texte et CHAQUE alias via le 'scorer' choisi,
    puis agrège au niveau canonical (garde le meilleur alias par canonical).
    """
    if not texte.strip():
        # Pas de texte : renvoie une ligne par canonical avec score 0
        meilleurs: Dict[str, Dict] = {}
        for al in tous_alias:
            cano = alias_vers_canonique.get(al.lower())
            if cano and cano not in meilleurs:
                meilleurs[cano] = {"canonical": cano, "meilleur_alias": al, "score": 0.0}
        return sorted(meilleurs.values(), key=lambda d: (-d["score"], d["canonical"]))

    if not RAPIDFUZZ_OK:
        st.warning("RapidFuzz n'est pas installé : scores à 0.")
        meilleurs: Dict[str, Dict] = {}
        for al in tous_alias:
            cano = alias_vers_canonique.get(al.lower())
            if cano and cano not in meilleurs:
                meilleurs[cano] = {"canonical": cano, "meilleur_alias": al, "score": 0.0}
        return sorted(meilleurs.values(), key=lambda d: (-d["score"], d["canonical"]))

    resultats = process.extract(texte, tous_alias, scorer=scorer, limit=len(tous_alias))

    meilleurs: Dict[str, Dict] = {}
    for alias, score, _ in resultats:
        cano = alias_vers_canonique.get(alias.lower())
        if not cano:
            continue
        cur = meilleurs.get(cano)
        if cur is None or score > cur["score"]:
            meilleurs[cano] = {"canonical": cano, "meilleur_alias": alias, "score": float(score)}

    return sorted(meilleurs.values(), key=lambda d: (-d["score"], d["canonical"]))

# --------------- Toponymes (mots distinctifs) ---------------
def extraire_toponymes_par_modele(tokens_par_canonique: Dict[str, set]) -> Dict[str, set]:
    """
    Reçoit le set de tokens (canonical+aliases) par modèle, déjà filtré des mots génériques.
    Retourne tel quel : {canonical: {bethmale, lers, milouga, ...}}
    """
    return tokens_par_canonique

def appliquer_boost_toponymes(texte: str,
                              scores_par_cano: List[Dict],
                              toponymes_par_modele: Dict[str, set],
                              boost: float = BOOST_TOPONYME) -> List[Dict]:
    """
    Si la requête contient un toponyme complet (mot entier) d’un modèle, +boost à ce modèle.
    """
    if not texte or not scores_par_cano:
        return scores_par_cano

    low = texte.lower()
    for row in scores_par_cano:
        cano = row["canonical"]
        mots = toponymes_par_modele.get(cano, set())
        if not mots:
            continue
        if any(re.search(rf"\b{re.escape(m)}\b", low, flags=re.IGNORECASE) for m in mots):
            row["score"] = min(100.0, float(row["score"]) + float(boost))

    return sorted(scores_par_cano, key=lambda d: (-d["score"], d["canonical"]))

# --------------- Fragments / Préfixes (priorité) ---------------
def extraire_fragments_utilisateur(texte: str) -> List[str]:
    """
    Récupère des fragments 'utiles' tapés par l’utilisateur :
      - tokens alphabétiques
      - longueur >= 3 (pour éviter 'de', 'le', 'la' etc.)
    """
    return [m.lower() for m in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", texte or "")]

def appliquer_boost_prefixes(texte: str,
                             scores_par_cano: List[Dict],
                             tokens_par_canonique: Dict[str, set],
                             boost: float = BOOST_PREFIXE) -> List[Dict]:
    """
    Si l’utilisateur tape un **début** de toponyme (ex. 'beth'), on booste le modèle
    dont un token commence par ce fragment (mot.startswith(fragment)).
    → évite que 'beth' parte sur 'Garbet' (suffixe) et favorise 'Bethmale'.
    """
    frags = extraire_fragments_utilisateur(texte)
    if not frags or not scores_par_cano:
        return scores_par_cano

    for row in scores_par_cano:
        cano = row["canonical"]
        toks = tokens_par_canonique.get(cano, set())
        if not toks:
            continue

        # Si AU MOINS un token commence par un fragment utilisateur → boost
        if any(any(tok.startswith(f) for tok in toks) for f in frags):
            row["score"] = min(100.0, float(row["score"]) + float(boost))

    return sorted(scores_par_cano, key=lambda d: (-d["score"], d["canonical"]))

# --------------- Réponse ---------------
def construire_reponse(scores: List[Dict],
                       meta_par_canonique: Dict[str, Dict],
                       seuil: float) -> str:
    if not scores:
        return "Je n’ai pas compris votre demande. Donnez le nom exact (ou une variante proche)."
    top = scores[0]
    cano = top["canonical"]
    url = (meta_par_canonique.get(cano) or {}).get("url")
    sc = float(top["score"])
    if sc >= float(seuil):
        msg = f"Je comprends que votre demande concerne « {cano} » (score {sc:.1f})."
        return msg + (f" Fiche : {url}" if url else "")
    else:
        msg = (f"Je ne suis pas certain d'avoir compris votre demande. "
               f"Peut-être voulez-vous parler de « {cano} » (score {sc:.1f}).")
        return msg + (f" Fiche : {url}" if url else "")

# =============================================================================
# UI
# =============================================================================
st.title("Couserans — Lacs & Étangs (fuzzy)")

# Charger JSON + index
entites_data = lire_json_entites(str(CHEMIN_JSON))
if not entites_data:
    st.error(f"Fichier introuvable ou invalide : {CHEMIN_JSON}. Clé 'entities' requise.")
    st.stop()

entites_json = json.dumps(entites_data, sort_keys=True, ensure_ascii=False)
alias_vers_canonique, meta_par_canonique, tous_alias, noms_principaux, tokens_par_canonique = \
    construire_index_depuis_json(entites_json)

toponymes_par_modele = extraire_toponymes_par_modele(tokens_par_canonique)

# Aide sous le titre
if noms_principaux:
    st.caption("Exemples : " + " • ".join(noms_principaux[:10]))

# Sidebar : paramètres
st.sidebar.header("Paramètres")
mode = st.sidebar.selectbox(
    "Mode de scoring",
    options=["WRatio", "partial_ratio", "token_set_ratio", "token_sort_ratio"],
    index=0,
    help="Choisis l'algorithme de similarité (comme dans FuzzyWuzzy/RapidFuzz)."
)
seuil = st.sidebar.slider("Seuil de suggestion", 0, 100, 80, 1)
boost_prefix = st.sidebar.slider("Boost préfixe (+)", 0, 60, int(BOOST_PREFIXE), 1)
boost_topo = st.sidebar.slider("Boost toponyme (+)", 0, 60, int(BOOST_TOPONYME), 1)

if st.sidebar.button("Recharger les modèles"):
    lire_json_entites.clear()
    construire_index_depuis_json.clear()
    st.success("Modèles rechargés.")

# État
if "historique" not in st.session_state:
    st.session_state.historique = []
if "derniers_scores" not in st.session_state:
    st.session_state.derniers_scores = []

# Formulaire
with st.form("f"):
    texte = st.text_input("Votre message", "", help="Ex: 'beth', 'lac bethmale', 'info étang lers'")
    ok = st.form_submit_button("Envoyer")

if st.button("Vider la conversation"):
    st.session_state.historique = []
    st.session_state.derniers_scores = []

# Traitement
if ok and texte.strip():
    scorer = choisir_scorer(mode)

    # 1) scores bruts avec le mode choisi
    scores = calculer_scores(texte, tous_alias, alias_vers_canonique, scorer)

    # 2) priorité aux fragments/prefixes (corrige 'beth' -> 'Bethmale')
    scores = appliquer_boost_prefixes(texte, scores, tokens_par_canonique, boost=float(boost_prefix))

    # 3) boost toponymes (mots entiers distinctifs)
    scores = appliquer_boost_toponymes(texte, scores, toponymes_par_modele, boost=float(boost_topo))

    # 4) réponse
    reponse = construire_reponse(scores, meta_par_canonique, seuil)

    # historique
    st.session_state.derniers_scores = scores
    st.session_state.historique.append({"role": "user", "contenu": texte})
    st.session_state.historique.append({"role": "assistant", "contenu": reponse})

# Affichage
for msg in st.session_state.historique:
    with st.chat_message(msg["role"]):
        st.write(msg["contenu"])

st.subheader("Scores — Tous les lacs/étangs")
if st.session_state.derniers_scores:
    lignes = []
    for it in st.session_state.derniers_scores:
        meta = meta_par_canonique.get(it["canonical"], {})
        lignes.append({
            "Nom (canonical)": it["canonical"],
            "Alias (meilleur)": it["meilleur_alias"],
            "Score": round(it["score"], 1),
            "Label": meta.get("label"),
            "Catégorie": meta.get("category"),
            "URL": meta.get("url"),
        })
    st.dataframe(lignes, use_container_width=True)
else:
    st.info("Saisissez un message puis cliquez sur Envoyer.")
