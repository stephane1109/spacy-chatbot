from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import html

import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher


# ============== Config ==============
# PAS de layout="wide"
st.set_page_config(page_title="Salomon NER + Fuzzy")
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "models.json"


# ============== Modèles de données ==============
@dataclass
class EntityMatch:
    text: str
    start: int
    end: int
    label: str
    canonical: str
    method: str   # "exact" | "fuzzy"
    score: float  # 0..1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(self.score, 4)
        return d


# ============== Pipeline ==============
class NERPipeline:
    """
    - Exact : spaCy PhraseMatcher (LOWER) alimenté par les alias du JSON.
    - Fuzzy : RapidFuzz partial_ratio sur le MESSAGE ENTIER (pas de fenêtre).
    - Modes :
        * off        -> fuzzy=False
        * balanced   -> fuzzy=True,  threshold=88
        * aggressive -> fuzzy=True,  threshold=82
    """

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

        # spaCy FR si dispo, sinon blank FR
        try:
            self.nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner", "lemmatizer", "tagger"])
        except Exception:
            self.nlp = spacy.blank("fr")

        # RapidFuzz ?
        try:
            from rapidfuzz import process as _p, fuzz as _f  # noqa: F401
            self.has_rapidfuzz = True
        except Exception:
            self.has_rapidfuzz = False

        # Réglages par défaut
        self.fuzzy_preset: str = "balanced" if self.has_rapidfuzz else "off"
        self.enable_fuzzy: bool = bool(self.has_rapidfuzz)
        self.fuzzy_threshold: int = 88  # partial_ratio

        # Données
        self._label_by_cano: Dict[str, str] = {}
        self._aliases_by_cano: Dict[str, List[str]] = {}
        self._lexicon: List[str] = []               # tous alias (inclut canonical)
        self._cano_by_alias_low: Dict[str, str] = {}  # alias.lower() -> cano
        self._pm: Optional[PhraseMatcher] = None

        self._load_entities()
        self._build_phrase_matcher()
        self.set_fuzzy_preset(self.fuzzy_preset)

    # ---------- presets ----------
    def set_fuzzy_preset(self, key: str) -> None:
        if not self.has_rapidfuzz:
            self.enable_fuzzy = False
            self.fuzzy_preset = "off"
            return
        key = (key or "balanced").lower()
        if key == "off":
            self.enable_fuzzy = False
            self.fuzzy_threshold = 100
        elif key == "aggressive":
            self.enable_fuzzy = True
            self.fuzzy_threshold = 82
        else:
            self.enable_fuzzy = True
            self.fuzzy_threshold = 88
        self.fuzzy_preset = key

    # ---------- données ----------
    def _load_entities(self) -> None:
        self._label_by_cano.clear()
        self._aliases_by_cano.clear()
        self._lexicon.clear()
        self._cano_by_alias_low.clear()

        try:
            data = json.loads(self.data_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

        for ent in data.get("entities", []):
            cano = (ent.get("canonical") or "").strip()
            if not cano:
                continue
            label = ent.get("label") or "MODEL"
            aliases = {cano}
            for a in ent.get("aliases", []) or []:
                a = (a or "").strip()
                if a:
                    aliases.add(a)

            self._label_by_cano[cano] = label
            self._aliases_by_cano[cano] = sorted(aliases, key=len, reverse=True)

            for a in aliases:
                self._lexicon.append(a)
                self._cano_by_alias_low[a.lower()] = cano

        # dédoublonner
        self._lexicon = list(dict.fromkeys(self._lexicon))

    def _build_phrase_matcher(self) -> None:
        pm = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for cano, aliases in self._aliases_by_cano.items():
            if not aliases:
                continue
            patterns = [self.nlp.make_doc(a) for a in aliases]
            pm.add(f"CANO::{cano}", patterns)
        self._pm = pm

    # ---------- extraction ----------
    def extract(self, text: str) -> List[EntityMatch]:
        if not text:
            return []
        doc = self.nlp.make_doc(text)
        out: List[EntityMatch] = []

        # 1) Exact (PhraseMatcher)
        if self._pm:
            for match_id, start, end in self._pm(doc):
                rule = self.nlp.vocab.strings[match_id]
                cano = rule.split("::", 1)[1] if "::" in rule else None
                span = doc[start:end]
                if not cano or not span.text.strip():
                    continue
                label = self._label_by_cano.get(cano, "MODEL")
                out.append(EntityMatch(
                    text=span.text,
                    start=span.start_char,
                    end=span.end_char,
                    label=label,
                    canonical=cano,
                    method="exact",
                    score=1.0,
                ))

        # 2) Fuzzy global (message entier)
        if self.enable_fuzzy and self.has_rapidfuzz:
            out.extend(self._fuzzy_full_message(text))

        # Déduplication basique
        out = self._dedupe_overlaps(out)
        return sorted(out, key=lambda m: (m.start, -m.end))

    def _fuzzy_full_message(self, text: str) -> List[EntityMatch]:
        from rapidfuzz import process, fuzz  # type: ignore

        s = (text or "").strip()
        if not s:
            return []

        results = process.extract(
            s,
            self._lexicon,
            scorer=fuzz.partial_ratio,
            score_cutoff=self.fuzzy_threshold,
            limit=5
        )

        out: List[EntityMatch] = []
        for alias, score, _ in results:
            cano = self._cano_by_alias_low.get(alias.lower())
            if not cano:
                continue
            label = self._label_by_cano.get(cano, "MODEL")

            # Localisation best-effort (pour surlignage) : recherche naïve (sans regex)
            s_low = s.lower()
            a_low = alias.lower()
            pos = s_low.find(a_low)
            if pos >= 0:
                start, end = pos, pos + len(alias)
                shown = text[start:end]
            else:
                # si l'alias n'apparaît pas littéralement (cas fuzzy), on met tout le message
                start, end = 0, len(text)
                shown = text

            out.append(EntityMatch(
                text=shown,
                start=start,
                end=end,
                label=label,
                canonical=cano,
                method="fuzzy",
                score=max(0.0, min(1.0, float(score) / 100.0)),
            ))
        return out

    @staticmethod
    def _dedupe_overlaps(items: List[EntityMatch]) -> List[EntityMatch]:
        if not items:
            return []
        # Priorité : exact > meilleur score > span le plus long
        items = sorted(items, key=lambda m: (0 if m.method == "exact" else 1, -(m.score), -(m.end - m.start)))
        kept: List[EntityMatch] = []
        occupied: List[Tuple[int, int]] = []

        def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
            return not (a[1] <= b[0] or b[1] <= a[0])

        for m in items:
            span = (m.start, m.end)
            if any(overlap(span, o) for o in occupied):
                continue
            kept.append(m)
            occupied.append(span)
        return kept


# ============== UI helpers ==============
def highlight_html(text: str, matches: List[EntityMatch]) -> str:
    if not text:
        return ""
    spans = sorted(matches, key=lambda m: m.start)
    out = []
    cur = 0
    for m in spans:
        if m.start > cur:
            out.append(html.escape(text[cur:m.start]))
        title = f"{m.label} → {m.canonical} ({m.method}, {int(m.score*100)}%)"
        out.append(
            f"<mark style='background-color:#fff3cd; padding:0 2px; border-radius:2px' "
            f"title='{html.escape(title)}'>{html.escape(text[m.start:m.end])}</mark>"
        )
        cur = m.end
    if cur < len(text):
        out.append(html.escape(text[cur:]))
    return "".join(out)


def assistant_reply_from_entities(matches: List[EntityMatch]) -> str:
    if not matches:
        return "Je n’ai pas reconnu de modèle. Indiquez le nom exact (ou activez un mode fuzzy) ?"
    canos = list(dict((m.canonical, None) for m in matches).keys())
    if len(canos) == 1:
        return f"Modèle détecté : {canos[0]}"
    return "Modèles détectés : " + ", ".join(canos)


# ============== Session ==============
if "pipeline" not in st.session_state:
    st.session_state.pipeline = NERPipeline(DATA_PATH)
pipeline: NERPipeline = st.session_state.pipeline

if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode_prev" not in st.session_state:
    st.session_state.mode_prev = pipeline.fuzzy_preset
if "mode_changed" not in st.session_state:
    st.session_state.mode_changed = False


# ============== Sidebar (3 modes) ==============
st.sidebar.header("Paramètres")

rf_ok = getattr(pipeline, "has_rapidfuzz", False)
labels = {"off": "Désactivé", "balanced": "Équilibré", "aggressive": "Agressif"}
current = st.session_state.mode_prev if rf_ok else "off"

selected = st.sidebar.radio(
    "Mode de correspondance",
    options=list(labels.keys()),
    index=list(labels.keys()).index(current),
    format_func=lambda k: labels[k],
)

# Appliquer et marquer le changement (pas de st.rerun)
if selected != st.session_state.mode_prev:
    pipeline.set_fuzzy_preset(selected)
    st.session_state.mode_prev = selected
    st.session_state.mode_changed = True
else:
    st.session_state.mode_changed = False

st.sidebar.markdown(
    """
**Modes**  
- **Désactivé** : dictionnaire exact seulement.  
- **Équilibré** : ajoute fuzzy (`partial_ratio ≥ 88`).  
- **Agressif** : fuzzy plus tolérant (`partial_ratio ≥ 82`).  
"""
)

with st.sidebar.expander("Diagnostic"):
    st.write(f"RapidFuzz : **{'oui' if rf_ok else 'non'}**")
    st.write(f"Modèles chargés : **{len(pipeline._aliases_by_cano)}**")
    st.write(f"Alias chargés : **{len(pipeline._lexicon)}**")
    if not rf_ok and selected != "off":
        st.info("RapidFuzz indisponible : le fuzzy est inactif (mode effectif = Désactivé).")


# ============== Main ==============
st.title("Chatbot NER + Fuzzy (Salomon)")

# Historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and msg.get("entities"):
            st.markdown(highlight_html(msg["content"], msg["entities"]), unsafe_allow_html=True)
        else:
            st.write(msg["content"])

# Ré-analyse auto si le mode a changé
if st.session_state.mode_changed:
    last_user = next((m for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
    if last_user:
        ents = pipeline.extract(last_user["content"])
        reply = "(Analyse mise à jour — mode " + labels[selected] + ") " + assistant_reply_from_entities(ents)
        st.session_state.messages.append({"role": "assistant", "content": reply, "entities": []})

# Saisie
with st.form("chat_form"):
    user_text = st.text_input("Votre message", value="")
    sent = st.form_submit_button("Envoyer")

if st.button("Vider la conversation"):
    st.session_state.messages = []

if sent and user_text.strip():
    text = user_text.strip()
    ents = pipeline.extract(text)
    st.session_state.messages.append({"role": "user", "content": text, "entities": ents})
    with st.chat_message("user"):
        st.markdown(highlight_html(text, ents), unsafe_allow_html=True)

    reply = assistant_reply_from_entities(ents)
    st.session_state.messages.append({"role": "assistant", "content": reply, "entities": []})

# Inspector
with st.expander("Inspector"):
    last_user = next((m for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
    if last_user and last_user.get("entities"):
        rows = [m.to_dict() for m in last_user["entities"]]
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("Envoyez un message pour voir les entités détectées.")
