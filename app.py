from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import json
import streamlit as st

# RapidFuzz = moteur de similarité (équivalent FuzzyWuzzy) : on n'utilise que WRatio
try:
    from rapidfuzz import process, fuzz  # type: ignore
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False

# ---------------------- Config ----------------------
st.set_page_config(page_title="Couserans • Lacs & Étangs (WRatio + aliases, insensible à la casse)")  # pas de wide
ROOT = Path(__file__).parent
JSON_PATH = ROOT / "data" / "models_couserans.json"

# ---------------------- Chargement JSON (caché) ----------------------
@st.cache_resource(show_spinner=False)
def load_entities(path_str: str) -> dict | None:
    """
    Lit le JSON des lacs/étangs.
    Format attendu: {"entities":[{"canonical":..., "aliases":[...], "label":..., "category":..., "url":...}, ...]}
    """
    p = Path(path_str)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("entities"), list):
            return data
    except Exception:
        return None
    return None

@st.cache_resource(show_spinner=False)
def build_index(entities_json_str: str):
    """
    Construit les structures pour le matching (case-insensitive) :
      - alias_to_cano      : alias_lower -> canonical
      - orig_by_lower      : alias_lower -> alias tel qu'il apparaît (pour l'affichage)
      - meta_by_cano       : {canonical: {label, category, url}}
      - all_aliases_lower  : liste plate des alias en minuscules (corpus de recherche)
      - examples           : liste de canonicals (affichage sous le titre)
    NOTE: on dédoublonne proprement pour éviter les collisions.
    """
    data = json.loads(entities_json_str)

    alias_to_cano: Dict[str, str] = {}
    orig_by_lower: Dict[str, str] = {}
    meta_by_cano: Dict[str, Dict] = {}
    all_aliases_lower: List[str] = []
    examples: List[str] = []

    for ent in data.get("entities", []):
        cano = (ent.get("canonical") or "").strip()
        if not cano:
            continue

        examples.append(cano)
        meta_by_cano[cano] = {
            "label": ent.get("label") or "LAC",
            "category": ent.get("category"),
            "url": ent.get("url"),
        }

        # On GARDE LES ALIASES (case-insensitive)
        variants = [cano] + list(ent.get("aliases", []) or [])
        for a in variants:
            a = (a or "").strip()
            if not a:
                continue
            low = a.lower()
            if low in alias_to_cano:
                # déjà vu : on ne remplace pas (on garde le premier alias d'origine pour l'affichage)
                continue
            alias_to_cano[low] = cano
            orig_by_lower[low] = a
            all_aliases_lower.append(low)

    # dédoublonnages simples en conservant l'ordre (déjà géré par le "if low in ...")
    examples = list(dict.fromkeys(examples))
    return alias_to_cano, orig_by_lower, meta_by_cano, all_aliases_lower, examples

# ---------------------- Scoring WRatio (case-insensitive) ----------------------
def score_wratio_ci(query: str,
                    all_aliases_lower: List[str],
                    alias_to_cano: Dict[str, str],
                    orig_by_lower: Dict[str, str]) -> List[Dict]:
    """
    Compare la requête à CHAQUE alias **en minuscules** avec WRatio,
    puis agrège au niveau 'canonical' en conservant le meilleur alias (affiché avec sa casse d'origine).
    Sortie triée desc: [{"canonical":..., "meilleur_alias":..., "score":...}, ...]
    """
    # Cas sans saisie → scores à 0 pour affichage de la table
    if not query.strip():
        best: Dict[str, Dict] = {}
        for al_low in all_aliases_lower:
            cano = alias_to_cano.get(al_low)
            if cano and cano not in best:
                best[cano] = {
                    "canonical": cano,
                    "meilleur_alias": orig_by_lower.get(al_low, al_low),
                    "score": 0.0
                }
        return sorted(best.values(), key=lambda d: (-d["score"], d["canonical"]))

    if not RAPIDFUZZ_OK:
        st.warning("RapidFuzz n'est pas installé : scores à 0.")
        best: Dict[str, Dict] = {}
        for al_low in all_aliases_lower:
            cano = alias_to_cano.get(al_low)
            if cano and cano not in best:
                best[cano] = {
                    "canonical": cano,
                    "meilleur_alias": orig_by_lower.get(al_low, al_low),
                    "score": 0.0
                }
        return sorted(best.values(), key=lambda d: (-d["score"], d["canonical"]))

    # IMPORTANT : on met aussi la requête en minuscules
    q_low = query.lower()

    # Scores par alias_lower (tel quel, pas d’autre normalisation)
    results = process.extract(q_low, all_aliases_lower, scorer=fuzz.WRatio, limit=len(all_aliases_lower))

    # Agrégation alias -> canonical : on garde le meilleur alias par canonical
    best: Dict[str, Dict] = {}
    for alias_low, score, _ in results:
        cano = alias_to_cano.get(alias_low)
        if not cano:
            continue
        cur = best.get(cano)
        if cur is None or score > cur["score"]:
            best[cano] = {
                "canonical": cano,
                "meilleur_alias": orig_by_lower.get(alias_low, alias_low),  # alias d'origine pour l'affichage
                "score": float(score)
            }

    return sorted(best.values(), key=lambda d: (-d["score"], d["canonical"]))

# ---------------------- Réponse (seuil unique) ----------------------
def build_reply(scores: List[Dict],
                meta_by_cano: Dict[str, Dict],
                threshold: float) -> str:
    """
    Logique binaire (seuil unique WRatio) :
      - top >= seuil → affirmation
      - top <  seuil → suggestion prudente
    """
    if not scores:
        return "Je n’ai pas compris votre demande. Donnez le nom exact ou une variante proche."

    top = scores[0]
    cano = top["canonical"]
    url = (meta_by_cano.get(cano) or {}).get("url")
    sc = float(top["score"])

    if sc >= float(threshold):
        msg = f"Je comprends que votre demande concerne « {cano} » (WRatio {sc:.1f})."
        return msg + (f" Fiche : {url}" if url else "")
    else:
        msg = (f"Je ne suis pas certain d'avoir compris votre demande. "
               f"Peut-être voulez-vous parler de « {cano} » (WRatio {sc:.1f}).")
        return msg + (f" Fiche : {url}" if url else "")

# ---------------------- UI ----------------------
st.title("Couserans — Lacs & Étangs (WRatio + aliases, insensible à la casse)")
st.markdown(
    """
**Principe.**  
Saisie libre → comparaison **WRatio** contre **tous les alias** (comparaison en *minuscules*) →  
agrégation par **nom canonique** → décision selon un **seuil**.

- *Insensibilité à la casse* : « BETHMALE », « bethmale » ou « Bethmale » donnent les mêmes scores.  
- *Nom canonique* = forme de référence (ex. « Lac de Bethmale »).  
- *Alias* = variantes réelles d’écriture (accents, tirets, fautes usuelles…).  
  Garder des aliases **augmente le rappel** et stabilise les scores.
"""
)

data = load_entities(str(JSON_PATH))
if not data:
    st.error(f"Fichier introuvable ou invalide : {JSON_PATH} (clé 'entities' requise).")
    st.stop()

entities_json = json.dumps(data, ensure_ascii=False, sort_keys=True)
alias_to_cano, orig_by_lower, meta_by_cano, all_aliases_lower, examples = build_index(entities_json)

# Aide : montrer quelques noms de référence à tester
if examples:
    st.caption("Exemples : " + " • ".join(examples[:10]))

# Sidebar : un seul réglage
st.sidebar.header("Paramètre")
threshold = st.sidebar.slider(
    "Seuil WRatio",
    min_value=0, max_value=100, value=80, step=1,
    help="Au-dessus de ce score, la réponse est affirmative ; sinon elle reste prudente."
)

if st.sidebar.button("Recharger"):
    load_entities.clear()
    build_index.clear()
    st.success("JSON rechargé depuis le disque.")

# État
if "hist" not in st.session_state:
    st.session_state.hist = []
if "last_scores" not in st.session_state:
    st.session_state.last_scores = []

# Formulaire
with st.form("f"):
    q = st.text_input("Votre message", "", help="Exemples : 'lac bethmale', 'infos étang lers', 'milouga ?'")
    ok = st.form_submit_button("Envoyer")

if st.button("Vider la conversation"):
    st.session_state.hist = []
    st.session_state.last_scores = []

# Traitement
if ok and q.strip():
    scores = score_wratio_ci(q, all_aliases_lower, alias_to_cano, orig_by_lower)
    reply = build_reply(scores, meta_by_cano, threshold)
    st.session_state.last_scores = scores
    st.session_state.hist.append({"role": "user", "txt": q})
    st.session_state.hist.append({"role": "assistant", "txt": reply})

# Affichage conversation
for m in st.session_state.hist:
    with st.chat_message(m["role"]):
        st.write(m["txt"])

# Tableau des scores
st.subheader("Scores WRatio — agrégés par nom canonique (avec alias, insensible à la casse)")
st.markdown(
    """
- **Alias (meilleur)** : la variante (avec casse d’origine) qui a obtenu le score le plus élevé pour ce modèle.  
- **Score** : WRatio retourné par RapidFuzz (pas de boost, pas d’autre normalisation).  
"""
)
if st.session_state.last_scores:
    rows = []
    for it in st.session_state.last_scores:
        meta = meta_by_cano.get(it["canonical"], {})
        rows.append({
            "Nom (canonical)": it["canonical"],
            "Alias (meilleur)": it["meilleur_alias"],
            "Score": round(it["score"], 1),
            "Label": meta.get("label"),
            "Catégorie": meta.get("category"),
            "URL": meta.get("url"),
        })
    rows = sorted(rows, key=lambda r: (-r["Score"], r["Nom (canonical)"]))
    st.dataframe(rows, use_container_width=True)
else:
    st.info("Saisissez un message puis cliquez sur Envoyer pour voir les scores.")
