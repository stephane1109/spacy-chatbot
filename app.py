from __future__ import annotations

# ================= Imports =================
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import html

import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher

# RapidFuzz (scores WRatio)
try:
    from rapidfuzz import process, fuzz  # type: ignore
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False


# ================= Config =================
# PAS de layout wide
st.set_page_config(page_title="Salomon NER • Scores de correspondance")
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "models.json"


# ================= Data models =================
@dataclass
class EntityMatch:
    text: str
    start: int
    end: int
    label: str
    canonical: str
    method: str   # "exact"
    score: float  # 0..1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(self.score, 3)
        return d


# ================= Loaders (cachés correctement) =================
@st.cache_resource(show_spinner=False)
def load_nlp():
    """Charge spaCy FR (ou blank FR). Aucun paramètre -> cache OK."""
    try:
        return spacy.load("fr_core_news_sm", disable=["parser", "ner", "lemmatizer", "tagger"])
    except Exception:
        return spacy.blank("fr")

@st.cache_resource(show_spinner=False)
def load_entities(path_str: str) -> Dict:
    """
    Charge le JSON des modèles.
    On passe un str (hashable) au lieu d'un Path pour le cache.
    """
    path = Path(path_str)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("entities"):
            return data
    except Exception:
        pass
    # Fallback d'exemple si le fichier n'est pas dispo sur le Cloud
    return {
        "entities": [
            {
                "canonical": "Salomon Speedcross 6",
                "aliases": ["Speedcross 6", "speedcross6", "speed cross 6"],
                "label": "MODEL",
                "category": "trail",
                "url": "https://www.salomon.com/fr-fr/shop-emea/product/speedcross-6.html"
            },
            {
                "canonical": "Salomon X Ultra 4",
                "aliases": ["X Ultra 4", "x-ultra-4", "xultra4", "X ULTRA4"],
                "label": "MODEL",
                "category": "hiking",
                "url": "https://www.salomon.com/fr-fr/shop-emea/product/x-ultra-4.html"
            },
            {
                "canonical": "Salomon Sense Ride 5",
                "aliases": ["Sense Ride 5", "senseride5", "sense-ride-5"],
                "label": "MODEL",
                "category": "trail",
                "url": "https://www.salomon.com/fr-fr/shop-emea/product/sense-ride-5.html"
            }
        ]
    }

@st.cache_resource(show_spinner=False)
def build_phrase_matcher_from_json(entities_json: str):
    """
    Construit le PhraseMatcher à partir d'une chaîne JSON (hashable).
    Surtout NE PAS passer l'objet nlp en paramètre (non hashable).
    """
    nlp = load_nlp()
    data = json.loads(entities_json)

    pm = PhraseMatcher(nlp.vocab, attr="LOWER")
    alias_to_cano: Dict[str, str] = {}
    cano_meta: Dict[str, Dict] = {}
    all_aliases: List[str] = []

    for ent in data.get("entities", []):
        cano = (ent.get("canonical") or "").strip()
        if not cano:
            continue
        label = ent.get("label") or "MODEL"
        cano_meta[cano] = {
            "label": label,
            "category": ent.get("category"),
            "url": ent.get("url"),
        }
        variants = [cano] + list(ent.get("aliases", []) or [])
        patterns = []
        for a in variants:
            a = (a or "").strip()
            if not a:
                continue
            alias_to_cano[a.lower()] = cano
            all_aliases.append(a)
            patterns.append(nlp.make_doc(a))
        if patterns:
            pm.add(f"CANO::{cano}", patterns)

    all_aliases = list(dict.fromkeys(all_aliases))
    return pm, alias_to_cano, cano_meta, all_aliases


# ================= Exact match + Fuzzy scoring =================
def exact_spans(nlp, pm, text: str, cano_meta: Dict[str, Dict]) -> List[EntityMatch]:
    if not text:
        return []
    doc = nlp.make_doc(text)
    out: List[EntityMatch] = []
    for match_id, start, end in pm(doc):
        rule = nlp.vocab.strings[match_id]  # "CANO::<canonical>"
        cano = rule.split("::", 1)[1] if "::" in rule else None
        span = doc[start:end]
        if not cano or not span.text.strip():
            continue
        label = (cano_meta.get(cano) or {}).get("label", "MODEL")
        out.append(
            EntityMatch(
                text=span.text,
                start=span.start_char,
                end=span.end_char,
                label=label,
                canonical=cano,
                method="exact",
                score=1.0,
            )
        )
    return out


def compute_wratio_scores(user_text: str,
                          aliases: List[str],
                          alias_to_cano: Dict[str, str]) -> List[Dict]:
    """
    Calcule un score WRatio pour CHAQUE alias, puis agrège au niveau canonique
    (on garde le meilleur alias par modèle).
    Retour : liste triée [{canonical, best_alias, score}] desc.
    """
    if not user_text.strip() or not RAPIDFUZZ_OK:
        # Pas de RF -> scores 0
        best_by_cano: Dict[str, Dict] = {}
        for al in aliases:
            cano = alias_to_cano.get(al.lower(), "")
            if not cano:
                continue
            cur = best_by_cano.get(cano)
            if cur is None or 0 > cur["score"]:
                best_by_cano[cano] = {"canonical": cano, "best_alias": al, "score": 0.0}
        return sorted(best_by_cano.values(), key=lambda d: (-d["score"], d["canonical"]))

    # 1) scores alias-level
    results = process.extract(
        user_text,
        aliases,
        scorer=fuzz.WRatio,   # même scorer que dans ton snippet
        limit=len(aliases)
    )

    # 2) agrégation par canonique (meilleur alias)
    best_by_cano: Dict[str, Dict] = {}
    for alias, score, _ in results:
        cano = alias_to_cano.get(alias.lower())
        if not cano:
            continue
        cur = best_by_cano.get(cano)
        if cur is None or score > cur["score"]:
            best_by_cano[cano] = {"canonical": cano, "best_alias": alias, "score": float(score)}

    return sorted(best_by_cano.values(), key=lambda d: (-d["score"], d["canonical"]))


# ================= UI helpers =================
def highlight_html(text: str, matches: List[EntityMatch]) -> str:
    if not text:
        return ""
    spans = sorted(matches, key=lambda m: m.start)
    out = []
    cur = 0
    for m in spans:
        if m.start > cur:
            out.append(html.escape(text[cur:m.start]))
        title = f"{m.label} → {m.canonical} (exact)"
        out.append(
            f"<mark style='background-color:#fff3cd; padding:0 2px; border-radius:2px' "
            f"title='{html.escape(title)}'>{html.escape(text[m.start:m.end])}</mark>"
        )
        cur = m.end
    if cur < len(text):
        out.append(html.escape(text[cur:]))
    return "".join(out)


# ================= Main UI =================
st.title("Chat NER (Salomon) + Scores (WRatio)")
st.caption("• NER exact (PhraseMatcher) • Scores RapidFuzz WRatio pour tous les modèles")

# Charge modèles + matcher (via caches sûrs)
nlp = load_nlp()
entities_data = load_entities(str(DATA_PATH))
entities_json = json.dumps(entities_data, sort_keys=True, ensure_ascii=False)  # hashable
pm, alias_to_cano, cano_meta, all_aliases = build_phrase_matcher_from_json(entities_json)

# État minimal
if "history" not in st.session_state:
    st.session_state.history = []
if "last_scores" not in st.session_state:
    st.session_state.last_scores = []

# Avertissement RapidFuzz
if not RAPIDFUZZ_OK:
    st.info("RapidFuzz n'est pas installé : les scores WRatio seront à 0. Ajoute `rapidfuzz` dans requirements.txt.")

# Formulaire
with st.form("chat_form"):
    user_text = st.text_input("Votre message", value="", help="Ex: 'Je cherche la speedcross6 pour terrain boueux'")
    sent = st.form_submit_button("Envoyer")

if st.button("Vider la conversation"):
    st.session_state.history = []
    st.session_state.last_scores = []

# Traitement
if sent and user_text.strip():
    text = user_text.strip()
    ents = exact_spans(nlp, pm, text, cano_meta)  # pour surlignage exact
    st.session_state.history.append({"role": "user", "content": text, "entities": ents})

    # Scores WRatio pour TOUS les modèles (best alias par modèle)
    scores = compute_wratio_scores(text, all_aliases, alias_to_cano)
    st.session_state.last_scores = scores

# Affichage historique (surlignage exact)
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(highlight_html(msg["content"], msg.get("entities", [])), unsafe_allow_html=True)

# Tableau des scores (tous les modèles)
st.subheader("Scores de correspondance (WRatio) — Tous les modèles")
if st.session_state.last_scores:
    rows = []
    for item in st.session_state.last_scores:
        meta = cano_meta.get(item["canonical"], {})
        rows.append({
            "Modèle (canonical)": item["canonical"],
            "Alias (meilleur)": item["best_alias"],
            "Score": round(item["score"], 3),
            "Label": meta.get("label"),
            "Catégorie": meta.get("category"),
            "URL": meta.get("url"),
        })
    rows = sorted(rows, key=lambda r: (-r["Score"], r["Modèle (canonical)"]))
    st.dataframe(rows, use_container_width=True)
else:
    st.info("Tape un message puis clique sur Envoyer pour calculer les scores.")

# Démo WRatio (comme ton snippet)
with st.expander("Exemple RapidFuzz (companies + WRatio)"):
    st.code(
        """from rapidfuzz import process, fuzz

companies = [
    "Apple Inc.",
    "Apple Incorporated",
    "APPLE INC",
    "Microsoft Corporation",
    "Microsoft Corp.",
    "Google LLC",
    "Alphabet Inc.",
]

matches = process.extract("apple incorporated", companies, scorer=fuzz.WRatio, limit=2)

print("Best matches:")
for match in matches:
    print(f"Match: {match[0]}, Score: {match[1]:.3f}")""",
        language="python",
    )
    if RAPIDFUZZ_OK:
        companies = [
            "Apple Inc.",
            "Apple Incorporated",
            "APPLE INC",
            "Microsoft Corporation",
            "Microsoft Corp.",
            "Google LLC",
            "Alphabet Inc.",
        ]
        demo = process.extract("apple incorporated", companies, scorer=fuzz.WRatio, limit=2)
        st.write("Best matches:")
        for alias, score, _ in demo:
            st.write(f"- Match: {alias}, Score: {score:.3f}")
    else:
        st.warning("RapidFuzz non disponible — impossible d'exécuter la démo.")
