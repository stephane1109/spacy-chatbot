from __future__ import annotations

# ================= Imports =================
from pathlib import Path
from typing import List, Dict
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
st.set_page_config(page_title="Salomon NER • Scores de correspondance")  # pas de wide
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


# ================= Loaders (cache sûrs) =================
@st.cache_resource(show_spinner=False)
def load_nlp():
    try:
        return spacy.load("fr_core_news_sm", disable=["parser", "ner", "lemmatizer", "tagger"])
    except Exception:
        return spacy.blank("fr")

@st.cache_resource(show_spinner=False)
def load_entities(path_str: str) -> Dict:
    path = Path(path_str)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("entities"):
            return data
    except Exception:
        pass
    # Fallback si le JSON n'est pas dispo (exemples Salomon)
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
    Construit PhraseMatcher à partir d'une chaîne JSON (hashable pour le cache).
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


# ================= Exact + WRatio =================
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
    Score WRatio pour chaque alias, agrégé au niveau canonique (meilleur alias).
    Retourne [{canonical, best_alias, score}] trié desc.
    """
    if not user_text.strip() or not RAPIDFUZZ_OK:
        best_by_cano: Dict[str, Dict] = {}
        for al in aliases:
            cano = alias_to_cano.get(al.lower(), "")
            if not cano:
                continue
            cur = best_by_cano.get(cano)
            if cur is None or 0 > cur["score"]:
                best_by_cano[cano] = {"canonical": cano, "best_alias": al, "score": 0.0}
        return sorted(best_by_cano.values(), key=lambda d: (-d["score"], d["canonical"]))

    results = process.extract(
        user_text,
        aliases,
        scorer=fuzz.WRatio,    # on garde WRatio (pas de correctif 100%)
        limit=len(aliases)
    )

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


def build_assistant_reply(user_text: str,
                          exact_entities: List[EntityMatch],
                          scores: List[Dict],
                          cano_meta: Dict[str, Dict]) -> str:
    """
    Réponse simple :
    - Si exact → on confirme le(s) modèle(s) trouvé(s).
    - Sinon → on propose le meilleur candidat WRatio (si score >= 80), avec score affiché.
    """
    if exact_entities:
        canos = list(dict((m.canonical, None) for m in exact_entities).keys())
        if len(canos) == 1:
            cano = canos[0]
            meta = cano_meta.get(cano, {})
            url = meta.get("url")
            base = f"Modèle détecté : {cano}."
            return base + (f" Fiche produit : {url}" if url else "")
        else:
            lines = [f"- {c}" for c in canos]
            return "Modèles détectés :\n" + "\n".join(lines)

    # Pas d'exact → proposer le top fuzzy si suffisamment haut
    if scores:
        top = scores[0]
        if top["score"] >= 80:
            cano = top["canonical"]
            meta = cano_meta.get(cano, {})
            url = meta.get("url")
            ans = f"Je suppose que vous parlez de « {cano} » (score WRatio {top['score']:.0f})."
            return ans + (f" Fiche produit : {url}" if url else "")
    return "Je n’ai pas reconnu de modèle. Donnez le nom exact ou une variante proche."


# ================= Main UI =================
st.title("Chat NER (Salomon) + Scores (WRatio)")
st.caption("• NER exact (PhraseMatcher) • Tableau des scores WRatio pour tous les modèles • Réponse chatbot après chaque requête")

# Charge modèles + matcher (via caches sûrs)
nlp = load_nlp()
entities_data = load_entities(str(DATA_PATH))
entities_json = json.dumps(entities_data, sort_keys=True, ensure_ascii=False)
pm, alias_to_cano, cano_meta, all_aliases = build_phrase_matcher_from_json(entities_json)

# État
if "history" not in st.session_state:
    st.session_state.history = []   # [{role, content, entities}]
if "last_scores" not in st.session_state:
    st.session_state.last_scores = []

if not RAPIDFUZZ_OK:
    st.info("RapidFuzz n'est pas installé : les scores WRatio seront à 0. Ajoute `rapidfuzz` au requirements.txt.")

# Formulaire
with st.form("chat_form"):
    user_text = st.text_input("Votre message", value="", help="Ex: 'Je voudrais des infos sur la Speedcross 6'")
    sent = st.form_submit_button("Envoyer")

if st.button("Vider la conversation"):
    st.session_state.history = []
    st.session_state.last_scores = []

# Traitement
if sent and user_text.strip():
    text = user_text.strip()

    # 1) Exact pour surlignage
    ents = exact_spans(nlp, pm, text, cano_meta)
    st.session_state.history.append({"role": "user", "content": text, "entities": ents})

    # 2) Scores WRatio (tous les modèles)
    scores = compute_wratio_scores(text, all_aliases, alias_to_cano)
    st.session_state.last_scores = scores

    # 3) Réponse chatbot
    reply = build_assistant_reply(text, ents, scores, cano_meta)
    st.session_state.history.append({"role": "assistant", "content": reply, "entities": []})

# Affichage historique
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(highlight_html(msg["content"], msg.get("entities", [])), unsafe_allow_html=True)
        else:
            st.write(msg["content"])

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
    st.info("Saisissez un message puis cliquez sur Envoyer pour voir les scores.")
