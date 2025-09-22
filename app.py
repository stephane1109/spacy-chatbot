from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Set
import json, re, unicodedata
import streamlit as st

# =================== RapidFuzz (WRatio) ===================
try:
    from rapidfuzz import process, fuzz  # type: ignore
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False

# =================== Config ===================
st.set_page_config(page_title="Couserans * ChatBot * RapidFuzz")  # pas de wide
ROOT = Path(__file__).parent
JSON_PATH = ROOT / "data" / "models_couserans.json"

# Mots très génériques (pour ignorer les tokens sans valeur sémantique et départager les ex-aequo)
GENERIC = {"lac","lacs","etang","étang","etangs","étangs","de","du","des","la","le","les","l","d","sur","aux","au"}

# =================== Normalisation ===================
def normaliser(s: str) -> str:
    """Minuscule, sans accents, petite ponctuation → espace, espaces compactés."""
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[-_’'.,/()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens_alpha(s: str, minlen: int = 3) -> List[str]:
    """Extrait des tokens alphabétiques (>= minlen)."""
    return re.findall(r"[a-zà-öø-ÿ]{%d,}" % minlen, s or "")

def retirer_generiques(s: str) -> str:
    """
    Version 'focus' d'un texte : normalise puis retire les mots génériques.
    Sert uniquement à départager les ex æquo (le score WRatio affiché ne change pas).
    """
    n = normaliser(s)
    if not n:
        return ""
    toks = [t for t in n.split() if t not in GENERIC]
    return " ".join(toks)

# =================== JSON ===================
@st.cache_resource(show_spinner=False)
def charger_json(path_str: str) -> dict | None:
    p = Path(path_str)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("entities"), list):
            return data
    except Exception:
        return None
    return None

@st.cache_resource(show_spinner=False)
def construire_index(entites_json_str: str):
    """
    Construit et retourne :
      - alias_norm_to_cano : alias_normalisé -> canonical
      - orig_by_norm       : alias_normalisé -> alias original (affichage)
      - meta_by_cano       : {canonical: {label, category, url}}
      - all_aliases_norm   : liste des aliases normalisés (corpus WRatio)
      - examples           : liste des canoniques (pour l’aide)
      - canonicals_norm    : set(canonical normalisé)
      - aliases_norm_set   : set(alias normalisé)
      - toponyms_norm_set  : set(tous toponymes normalisés) → GARDE-FOU UNIQUEMENT
      - alias_focus_by_norm: alias_normalisé -> alias 'focus' (sans mots génériques) → TRI SECONDAIRE
    """
    data = json.loads(entites_json_str)

    alias_norm_to_cano: Dict[str, str] = {}
    orig_by_norm: Dict[str, str] = {}
    meta_by_cano: Dict[str, Dict] = {}
    all_aliases_norm: List[str] = []
    examples: List[str] = []

    canonicals_norm: Set[str] = set()
    aliases_norm_set: Set[str] = set()
    toponyms_norm_set: Set[str] = set()
    alias_focus_by_norm: Dict[str, str] = {}

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
        cn = normaliser(cano)
        if cn:
            canonicals_norm.add(cn)

        # Aliases (normalisés) → corpus WRatio + version 'focus' (tri)
        variants = [cano] + list(ent.get("aliases", []) or [])
        for a in variants:
            a = (a or "").strip()
            if not a:
                continue
            an = normaliser(a)
            if not an:
                continue
            if an in alias_norm_to_cano:
                continue
            alias_norm_to_cano[an] = cano
            orig_by_norm[an] = a
            all_aliases_norm.append(an)
            aliases_norm_set.add(an)
            alias_focus_by_norm[an] = retirer_generiques(a)

        # Toponymes (normalisés) → GARDE-FOU UNIQUEMENT (pas de scoring)
        for t in (ent.get("toponyms") or []):
            t = (t or "").strip()
            if not t:
                continue
            tn = normaliser(t)
            if tn and tn not in GENERIC:
                toponyms_norm_set.add(tn)

    examples = list(dict.fromkeys(examples))
    return (
        alias_norm_to_cano,
        orig_by_norm,
        meta_by_cano,
        all_aliases_norm,
        examples,
        canonicals_norm,
        aliases_norm_set,
        toponyms_norm_set,
        alias_focus_by_norm,
    )

# =================== Garde-fou : y a-t-il un signal JSON ? ===================
def a_un_signal_json(query: str,
                     canonicals_norm: Set[str],
                     aliases_norm_set: Set[str],
                     toponyms_norm_set: Set[str]) -> bool:
    """
    True si la requête (normalisée) contient :
      - un canonical normalisé, OU
      - un alias normalisé, OU
      - un indice toponymique (préfixe strict/souple) → garde-fou seulement
    Sinon False.
    """
    qn = normaliser(query)
    if not qn:
        return False

    if any(cn in qn for cn in canonicals_norm):
        return True
    if any(al in qn for al in aliases_norm_set):
        return True

    # Indices toponymiques (sans modifier les scores)
    toks = [t for t in tokens_alpha(qn, 3) if t not in GENERIC]
    if not toks:
        return False
    for qt in toks:
        if any(tp.startswith(qt) for tp in toponyms_norm_set):
            return True
        if len(qt) >= 4 and any(tp.startswith(qt[:-1]) for tp in toponyms_norm_set):
            return True
    return False

# =================== Scoring WRatio (agrégé par canonical) + tri secondaire ===================
def scorer_wratio(query: str,
                  all_aliases_norm: List[str],
                  alias_norm_to_cano: Dict[str, str],
                  orig_by_norm: Dict[str, str],
                  alias_focus_by_norm: Dict[str, str]) -> List[Dict]:
    """
    1) Score principal : WRatio(query_norm, alias_norm) → agrégation par canonical (meilleur alias).
    2) Tri secondaire : en cas d’égalité de score, on compare la requête 'focus' à l'alias 'focus'
       pour départager (sans changer le score principal).
    """
    # Cas vide / RapidFuzz absent
    if not query.strip() or not RAPIDFUZZ_OK:
        if not RAPIDFUZZ_OK:
            st.warning("RapidFuzz n'est pas installé : scores à 0.")
        best: Dict[str, Dict] = {}
        for al in all_aliases_norm:
            cano = alias_norm_to_cano.get(al)
            if cano and cano not in best:
                best[cano] = {"canonical": cano, "meilleur_alias": orig_by_norm.get(al, al), "score": 0.0, "focus_score": 0.0}
        return sorted(best.values(), key=lambda d: (-d["score"], -d["focus_score"], d["canonical"]))

    qn = normaliser(query)
    q_focus = retirer_generiques(query)

    # Score principal (WRatio) sur alias normalisés
    results = process.extract(qn, all_aliases_norm, scorer=fuzz.WRatio, limit=len(all_aliases_norm))

    best: Dict[str, Dict] = {}
    for alias_norm, score, _ in results:
        cano = alias_norm_to_cano.get(alias_norm)
        if not cano:
            continue

        # score 'focus' pour départager en cas d'égalité de 'score'
        focus_score = 0.0
        if q_focus:
            alias_focus = alias_focus_by_norm.get(alias_norm, alias_norm)
            if alias_focus:
                focus_score = float(fuzz.WRatio(q_focus, normaliser(alias_focus)))

        cur = best.get(cano)
        if (cur is None) or (score > cur["score"]) or (score == cur["score"] and focus_score > cur.get("focus_score", -1)):
            best[cano] = {
                "canonical": cano,
                "meilleur_alias": orig_by_norm.get(alias_norm, alias_norm),
                "score": float(score),          # score WRatio affiché
                "focus_score": focus_score,     # sert uniquement au tri
            }

    # Tri : score desc, puis focus_score desc, puis nom
    return sorted(best.values(), key=lambda d: (-d["score"], -d.get("focus_score", 0.0), d["canonical"]))

# =================== Réponse (seuil unique) ===================
def construire_reponse(scores: List[Dict],
                       meta_by_cano: Dict[str, Dict],
                       seuil: float) -> str:
    """
    - Si top score >= seuil → réponse affirmative.
    - Sinon → réponse prudente.
    - Si pas de scores (ne devrait pas arriver une fois le garde-fou passé) → message neutre.
    """
    if not scores:
        return "Je n’ai pas compris votre demande. Donnez le nom exact ou une variante proche."

    top = scores[0]
    cano = top["canonical"]
    url = (meta_by_cano.get(cano) or {}).get("url")
    sc = float(top["score"])

    if sc >= float(seuil):
        txt = f"Je comprends que votre demande concerne « {cano} » (WRatio {sc:.1f})."
        return txt + (f" Fiche : {url}" if url else "")
    else:
        txt = (f"Je ne suis pas certain d'avoir compris votre demande. "
               f"Peut-être voulez-vous parler de « {cano} » (WRatio {sc:.1f}).")
        return txt + (f" Fiche : {url}" if url else "")

# =================== UI ===================
st.title("Couserans — ChatBot - RapidFuzz  (WRatio + garde-fou + tri focus)")
st.markdown(
    """
**Flux de décision**

1. On calcule **WRatio** contre **tous les aliases** (normalisés), et on match le **meilleur alias** et son **score**.   

2. **Seuil WRatio**  
   - *score ≥ seuil* → **réponse affirmative**  
   - *score < seuil* → **suggestion prudente**  
   - **Garde-fou** : si la requête ne contient ni *nom canonique*, ni *alias*, ni **toponyme**, on répond : *« Je n’ai pas compris votre question. »*

3. **Ex æquo**  
   En cas d’égalité de score, un **tri secondaire “focus”** compare la requête et les aliases **sans mots génériques** pour départager **sans modifier le score principal**.
    """
)

data = charger_json(str(JSON_PATH))
if not data:
    st.error(f"Fichier introuvable ou invalide : {JSON_PATH} (clé 'entities' requise).")
    st.stop()

entities_json = json.dumps(data, ensure_ascii=False, sort_keys=True)
(alias_norm_to_cano,
 orig_by_norm,
 meta_by_cano,
 all_aliases_norm,
 examples,
 canonicals_norm,
 aliases_norm_set,
 toponyms_norm_set,
 alias_focus_by_norm) = construire_index(entities_json)

if examples:
    st.caption("Exemples : " + " • ".join(examples[:10]))

# Seuil WRatio (unique)
if "seuil_wratio" not in st.session_state:
    st.session_state.seuil_wratio = 80

st.sidebar.header("Paramètre")
seuil = st.sidebar.slider(
    "Seuil WRatio",
    min_value=0, max_value=100,
    value=st.session_state.seuil_wratio, step=1,
    help="Au-dessus de ce score, la réponse est affirmative ; sinon elle reste prudente.",
    key="slider_seuil_wratio"
)
st.session_state.seuil_wratio = seuil

# Rechargement JSON
if st.sidebar.button("Recharger le JSON"):
    charger_json.clear()
    construire_index.clear()
    data = charger_json(str(JSON_PATH))
    if data:
        entities_json = json.dumps(data, ensure_ascii=False, sort_keys=True)
        (alias_norm_to_cano,
         orig_by_norm,
         meta_by_cano,
         all_aliases_norm,
         examples,
         canonicals_norm,
         aliases_norm_set,
         toponyms_norm_set,
         alias_focus_by_norm) = construire_index(entities_json)
        st.success("JSON rechargé.")
    else:
        st.error("Échec du rechargement du JSON.")

# État
if "hist" not in st.session_state:
    st.session_state.hist = []
if "last_scores" not in st.session_state:
    st.session_state.last_scores = []

# Formulaire
with st.form("form_chat"):
    q = st.text_input("Votre message", "", help="Ex : 'etang du garb', 'lac bethmale', 'infos lers'…")
    ok = st.form_submit_button("Envoyer")

if st.button("Vider la conversation"):
    st.session_state.hist = []
    st.session_state.last_scores = []

# Traitement
if ok and q.strip():
    # 1) Garde-fou : aucun signal JSON → réponse directe
    if not a_un_signal_json(q, canonicals_norm, aliases_norm_set, toponyms_norm_set):
        ex = " • ".join(examples[:6]) if examples else ""
        reply = ("Je n’ai pas compris votre question. "
                 "Citez explicitement le nom d’un lac/étang de la liste."
                 + (f" Exemples : {ex}" if ex else ""))
        st.session_state.last_scores = []
        st.session_state.hist.append({"role": "user", "txt": q})
        st.session_state.hist.append({"role": "assistant", "txt": reply})
    else:
        # 2) Matching WRatio (sans boost) + tri secondaire 'focus'
        scores = scorer_wratio(q, all_aliases_norm, alias_norm_to_cano, orig_by_norm, alias_focus_by_norm)
        reply = construire_reponse(scores, meta_by_cano, st.session_state.seuil_wratio)

        st.session_state.last_scores = scores
        st.session_state.hist.append({"role": "user", "txt": q})
        st.session_state.hist.append({"role": "assistant", "txt": reply})

# Affichage conversation
for m in st.session_state.hist:
    with st.chat_message(m["role"]):
        st.write(m["txt"])

# Tableau des scores
st.subheader("Scores — meilleurs par nom canonique (WRatio sur aliases)")
st.markdown(
    """
- **Alias (meilleur)** : la variante d’origine qui obtient le meilleur score WRatio.  
- **Score** : WRatio (sans boost de toponymes).  
- En cas d’ex æquo, l’ordre affiché est départagé par le **score focus** (requête/alias sans mots génériques).
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
