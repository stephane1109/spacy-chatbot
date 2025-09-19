from __future__ import annotations

# ========= Imports =========
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, asdict
import json
import html
import re

import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher


# ========= Config =========
# PAS de layout="wide"
st.set_page_config(page_title="Salomon NER Chatbot")
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "models.json"
RESP_PATH = ROOT / "data" / "responses.json"


# ========= Modèles de données =========
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


# ========= Pipeline NER (ultra simplifié) =========
class NERPipeline:
    """
    - Exact : spaCy PhraseMatcher (alias depuis models.json) + fallback exact par sous-chaîne
              (insensible à la casse, tolère espaces/traits d’union).
    - Fuzzy : RapidFuzz sur le MESSAGE ENTIER (pas de fenêtre), scorer = partial_ratio.
    - Modes :
        * off        -> fuzzy=False
        * balanced   -> fuzzy=True,  threshold=88
        * aggressive -> fuzzy=True,  threshold=82
    """

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

        # spaCy : FR si dispo, sinon "blank"
        try:
            self.nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner", "lemmatizer", "tagger"])
        except Exception:
            self.nlp = spacy.blank("fr")

        # RapidFuzz présent ?
        try:
            from rapidfuzz import process as _p, fuzz as _f  # noqa: F401
            self.has_rapidfuzz = True
        except Exception:
            self.has_rapidfuzz = False

        # Réglages
        self.fuzzy_preset: str = "balanced" if self.has_rapidfuzz else "off"
        self.enable_fuzzy: bool = bool(self.has_rapidfuzz)
        self.fuzzy_threshold: int = 88  # partial_ratio

        # Données
        self._label_by_cano: Dict[str, str] = {}
        self._aliases_by_cano: Dict[str, List[str]] = {}
        self._lexicon: List[str] = []                 # alias + canonical
        self._cano_by_alias_low: Dict[str, str] = {}  # alias.lower() -> canonique

        # Matcher exact
        self._pm: Optional[PhraseMatcher] = None

        # Chargement
        self._load_entities()
        self._build_phrase_matcher()
        self.set_fuzzy_preset(self.fuzzy_preset)

    # ----- Presets -----
    def set_fuzzy_preset(self, key: str) -> None:
        """
        Modes :
        - off        : fuzzy off
        - balanced   : fuzzy on,  threshold=88
        - aggressive : fuzzy on,  threshold=82
        """
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

    # ----- Données -----
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

        # Dédoublonner et trier
        self._lexicon = sorted(list(dict.fromkeys(self._lexicon)), key=len, reverse=True)

    def _build_phrase_matcher(self) -> None:
        pm = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for cano, aliases in self._aliases_by_cano.items():
            if not aliases:
                continue
            patterns = [self.nlp.make_doc(a) for a in aliases]
            pm.add(f"CANO::{cano}", patterns)
        self._pm = pm

    # ----- Extraction -----
    def extract(self, text: str) -> List[EntityMatch]:
        if not text:
            return []
        doc = self.nlp.make_doc(text)

        matches: List[EntityMatch] = []

        # 1) Exact (PhraseMatcher)
        if self._pm:
            for match_id, start, end in self._pm(doc):
                rule = self.nlp.vocab.strings[match_id]  # "CANO::<canonique>"
                cano = rule.split("::", 1)[1] if "::" in rule else None
                span = doc[start:end]
                if not cano or not span.text.strip():
                    continue
                label = self._label_by_cano.get(cano, "MODEL")
                matches.append(
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

        # 1bis) Exact fallback par sous-chaîne (tolère espaces/traits d’union)
        matches.extend(self._exact_substring_matches(text))

        # 2) Fuzzy (MESSAGE ENTIER)
        if self.enable_fuzzy and self.has_rapidfuzz:
            matches.extend(self._fuzzy_full_message(text))

        # Déduplication de chevauchements
        matches = self._dedupe_overlaps(matches)
        return sorted(matches, key=lambda m: (m.start, -m.end))

    def _exact_substring_matches(self, text: str) -> List[EntityMatch]:
        out: List[EntityMatch] = []
        if not text:
            return out
        t = text
        for alias in self._lexicon:
            # Autoriser espaces/traits d’union interchangeables
            pattern = re.escape(alias)
            pattern = pattern.replace(r"\ ", r"[\s\-]+").replace(r"\-", r"[\s\-]+")
            m = re.search(rf"(?i)\b{pattern}\b", t)
            if not m:
                continue
            cano = self._cano_by_alias_low.get(alias.lower())
            if not cano:
                continue
            label = self._label_by_cano.get(cano, "MODEL")
            out.append(EntityMatch(
                text=t[m.start():m.end()],
                start=m.start(),
                end=m.end(),
                label=label,
                canonical=cano,
                method="exact",
                score=1.0,
            ))
        return out

    def _fuzzy_full_message(self, text: str) -> List[EntityMatch]:
        """Compare le message ENTIER au lexique avec partial_ratio."""
        from rapidfuzz import process, fuzz  # type: ignore

        s = (text or "").strip()
        if not s:
            return []

        results = process.extract(
            s,
            self._lexicon,
            scorer=fuzz.partial_ratio,         # tolère texte additionnel autour de l'alias
            score_cutoff=self.fuzzy_threshold, # seuil fixé par le mode
            limit=5
        )

        out: List[EntityMatch] = []
        for alias, score, _ in results:
            cano = self._cano_by_alias_low.get(alias.lower())
            if not cano:
                continue
            label = self._label_by_cano.get(cano, "MODEL")
            # localiser l'alias si possible
            m = re.search(re.escape(alias), text, re.I)
            if m:
                start, end = m.start(), m.end()
                shown = text[start:end]
            else:
                start, end = 0, len(text)
                shown = text
            out.append(
                EntityMatch(
                    text=shown,
                    start=start,
                    end=end,
                    label=label,
                    canonical=cano,
                    method="fuzzy",
                    score=max(0.0, min(1.0, float(score) / 100.0)),
                )
            )
        return out

    def _dedupe_overlaps(self, items: List[EntityMatch]) -> List[EntityMatch]:
        if not items:
            return []
        # Priorité : exact, puis meilleur score, puis plus long
        items = sorted(
            items,
            key=lambda m: (0 if m.method == "exact" else 1, -(m.score), -(m.end - m.start)),
        )
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


# ========= UI helpers =========
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
        styled = (
            f"<mark style='background-color:#fff3cd; padding:0 2px; border-radius:2px' "
            f"title='{html.escape(title)}'>{html.escape(text[m.start:m.end])}</mark>"
        )
        out.append(styled)
        cur = m.end
    if cur < len(text):
        out.append(html.escape(text[cur:]))
    return "".join(out)


# ========= Réponses / intentions (minimal) =========
def load_models_index() -> dict:
    try:
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    idx = {}
    for ent in data.get("entities", []):
        cano = ent.get("canonical")
        if not cano:
            continue
        idx[cano] = {
            "category": ent.get("category"),
            "label": ent.get("label", "MODEL"),
            "url": ent.get("url"),
        }
    return idx


def load_responses_config() -> dict:
    default = {
        "intents": [
            {"name": "advantages", "keywords": ["avantage", "points forts", "bénéf", "pourquoi", "atout"]},
            {"name": "price", "keywords": ["prix", "tarif", "coût", "combien"]},
            {"name": "waterproof", "keywords": ["imperm", "gore-tex", "gtx", "étanch", "membrane"]},
            {"name": "terrain", "keywords": ["terrain", "boue", "montagne", "rocaille", "roche", "pierre", "chemin"]},
        ],
        "responses": {
            "advantages": {
                "generic": {"text": "Voici les points forts de ce modèle: accroche, stabilité et protection.", "url": None},
                "by_category": {
                    "trail": {"text": "Trail: très bonne accroche (crampons), maintien du pied et protection.", "url": None},
                    "hiking": {"text": "Randonnée: confort et maintien, bonne adhérence et durabilité.", "url": None},
                },
                "by_model": {
                    "Salomon Speedcross 6": {"text": "Speedcross 6: accroche agressive, maintien précis, idéale en terrain meuble.", "url": "https://www.salomon.com/fr-fr/shop-emea/product/speedcross-6.html"},
                    "Salomon X Ultra 4": {"text": "X Ultra 4: stabilité en rando, accroche sur terrain technique, confort durable.", "url": "https://www.salomon.com/fr-fr/shop-emea/product/x-ultra-4.html"},
                },
            },
            "price": {"generic": {"text": "Je n’ai pas le prix exact. Consultez la fiche produit.", "url": None}},
            "waterproof": {"generic": {"text": "Version GTX = imperméable (membrane).", "url": None}},
            "terrain": {"generic": {"text": "Précisez le terrain (boue, rocaille, sentier, route)…", "url": None}},
        },
    }

    def merge_cfg(dflt: dict, custom: dict) -> dict:
        merged_intents = list(custom.get("intents", []))
        existing = {i.get("name") for i in merged_intents}
        for it in dflt.get("intents", []):
            if it.get("name") not in existing:
                merged_intents.append(it)

        merged_resp: dict = {}
        custom_resp = custom.get("responses", {})
        all_keys = set(dflt.get("responses", {}).keys()) | set(custom_resp.keys())
        for key in all_keys:
            base = dflt.get("responses", {}).get(key, {})
            cur = custom_resp.get(key, {})
            out = {
                "generic": cur.get("generic", base.get("generic")),
                "by_category": cur.get("by_category", base.get("by_category", {})),
                "by_model": cur.get("by_model", base.get("by_model", {})),
            }
            merged_resp[key] = out

        return {"intents": merged_intents, "responses": merged_resp}

    try:
        user_cfg = json.loads(RESP_PATH.read_text(encoding="utf-8"))
        return merge_cfg(default, user_cfg)
    except FileNotFoundError:
        return default
    except Exception:
        return merge_cfg(default, {})


def detect_intent(nlp, text: str, cfg: dict) -> Optional[str]:
    low = (text or "").lower()

    pm = PhraseMatcher(nlp.vocab, attr="LOWER")
    label_to_intent = {}
    for it in cfg.get("intents", []):
        name = it.get("name")
        kws = [kw for kw in it.get("keywords", []) if kw]
        if not kws:
            continue
        patterns = [nlp.make_doc(kw) for kw in kws]
        pm.add(f"INTENT::{name}", patterns)
        label_to_intent[f"INTENT::{name}"] = name
    doc = nlp.make_doc(text or "")
    matches = pm(doc)
    matched = [label_to_intent.get(nlp.vocab.strings[m_id]) for m_id, _, _ in matches]
    matched = [m for m in matched if m]
    if matched:
        priority = ["advantages", "price", "waterproof", "terrain"]
        for p in priority:
            if p in matched:
                return p
        return matched[0]

    # fallback "contains"
    for it in cfg.get("intents", []):
        for kw in it.get("keywords", []):
            if (kw or "").lower() in low:
                return it.get("name")

    # fuzzy des intentions (optionnel)
    try:
        from rapidfuzz import process, fuzz  # type: ignore
    except Exception:
        return None
    targets: List[str] = []
    kw_to_intents: dict[str, List[str]] = {}
    for it in cfg.get("intents", []):
        name = it.get("name")
        for kw in it.get("keywords", [])[:50]:
            k = (kw or "").lower()
            if not k:
                continue
            targets.append(k)
            kw_to_intents.setdefault(k, []).append(name)
    if not targets:
        return None
    best = process.extractOne(low, targets, scorer=fuzz.WRatio)
    if not best or best[1] < 85:
        return None
    intents_for_kw = kw_to_intents.get(best[0], [])
    priority = ["advantages", "price", "waterproof", "terrain"]
    for p in priority:
        if p in intents_for_kw:
            return p
    return intents_for_kw[0] if intents_for_kw else None


def resolve_text_url(entry, fallback_text: str | None, model_name: str | None = None):
    text, url = None, None
    if isinstance(entry, dict):
        text = entry.get("text")
        url = entry.get("url")
    elif isinstance(entry, str):
        text = entry
    if not text:
        text = fallback_text or ""
    if not url and model_name:
        url = (load_models_index().get(model_name) or {}).get("url")
    return text, url


def assistant_reply(nlp, user_text: str, matches: List[EntityMatch], fallback_canos: list[str] | None = None) -> str:
    models_idx = load_models_index()
    cfg = load_responses_config()

    if matches:
        canos = list(dict((m.canonical, None) for m in matches).keys())
    elif fallback_canos:
        canos = list(dict((c, None) for c in fallback_canos).keys())
    else:
        canos = []

    intent = detect_intent(nlp, user_text, cfg)

    if not canos:
        res_cfg = cfg.get("responses", {}).get(intent or "", {})
        if isinstance(res_cfg, dict) and res_cfg.get("generic") is not None:
            gen_entry = res_cfg.get("generic")
            gen_text = gen_entry.get("text") if isinstance(gen_entry, dict) else gen_entry
            t, u = resolve_text_url(gen_entry, gen_text or "Pouvez-vous préciser le modèle ?")
            return (t or "") + (f" [Fiche produit]({u})" if u else "")
        return "Je n'ai pas encore reconnu de modèle Salomon. Pouvez-vous préciser le nom du modèle ?"

    if not intent:
        if len(canos) == 1:
            return f"Je comprends que vous cherchez des informations sur le modèle : {canos[0]}. Sur quel aspect souhaitez-vous de l’aide (ex: avantages, prix, imperméabilité, terrain) ?"
        else:
            return "Je comprends que vous cherchez des informations sur plusieurs modèles. Sur quel aspect souhaitez-vous de l’aide (ex: avantages, prix, imperméabilité, terrain) ?"

    res_cfg = cfg.get("responses", {}).get(intent, {})

    if len(canos) == 1:
        m = canos[0]
        by_model = res_cfg.get("by_model", {})
        if m in by_model:
            t, u = resolve_text_url(by_model[m], (res_cfg.get("generic") or {}).get("text") if isinstance(res_cfg.get("generic"), dict) else res_cfg.get("generic"), m)
            reply = f"Je comprends que vous cherchez des informations sur le modèle : {m}. {t}"
            if u:
                reply += f" [Fiche produit]({u})"
            return reply
        cat = (models_idx.get(m) or {}).get("category")
        by_cat = res_cfg.get("by_category", {})
        if cat and cat in by_cat:
            t, u = resolve_text_url(by_cat[cat], (res_cfg.get("generic") or {}).get("text") if isinstance(res_cfg.get("generic"), dict) else res_cfg.get("generic"), m)
            reply = f"Je comprends que vous cherchez des informations sur le modèle : {m}. {t}"
            if u:
                reply += f" [Fiche produit]({u})"
            return reply
        gen_entry = res_cfg.get("generic")
        gen_text = gen_entry.get("text") if isinstance(gen_entry, dict) else gen_entry
        t, u = resolve_text_url(gen_entry, gen_text or "Je peux vous donner des informations générales sur ce modèle.", m)
        reply = f"Je comprends que vous cherchez des informations sur le modèle : {m}. {t}"
        if u:
            reply += f" [Fiche produit]({u})"
        return reply

    lines = []
    by_model = res_cfg.get("by_model", {})
    by_cat = res_cfg.get("by_category", {})
    gen_entry = res_cfg.get("generic")
    gen_text = gen_entry.get("text") if isinstance(gen_entry, dict) else gen_entry or "Je peux vous donner des informations générales."
    for m in canos:
        if m in by_model:
            t, u = resolve_text_url(by_model[m], gen_text, m)
            line = f"- {m}: {t}"
            if u:
                line += f" [Fiche produit]({u})"
            lines.append(line)
            continue
        cat = (models_idx.get(m) or {}).get("category")
        if cat and cat in by_cat:
            t, u = resolve_text_url(by_cat[cat], gen_text, m)
            line = f"- {m}: {t}"
            if u:
                line += f" [Fiche produit]({u})"
            lines.append(line)
            continue
        t, u = resolve_text_url(gen_entry, gen_text, m)
        line = f"- {m}: {t}"
        if u:
            line += f" [Fiche produit]({u})"
        lines.append(line)
    return "Je comprends que vous cherchez des informations sur les modèles :\n" + "\n".join(lines)


# ========= Session =========
if "pipeline" not in st.session_state:
    st.session_state.pipeline = NERPipeline(DATA_PATH)
pipeline: NERPipeline = st.session_state.pipeline

if "messages" not in st.session_state:
    st.session_state.messages = []
if "context_canos" not in st.session_state:
    st.session_state.context_canos = []

# ========= Sidebar (3 modes, explication incluse) =========
st.sidebar.header("Paramètres")

rf_installed = getattr(pipeline, "has_rapidfuzz", False)
labels = {"off": "Désactivé", "balanced": "Équilibré", "aggressive": "Agressif"}

selected_key = st.sidebar.radio(
    "Mode de correspondance",
    options=list(labels.keys()),
    index=list(labels.keys()).index(pipeline.fuzzy_preset if rf_installed else "off"),
    format_func=lambda k: labels[k],
)

# Appliquer le mode (pas de rerun manuel ; le radio relance déjà le script)
pipeline.set_fuzzy_preset(selected_key)

# Explication claire des modes
st.sidebar.markdown(
    """
**Explications des modes**  
- **Désactivé** : uniquement les alias exacts (PhraseMatcher + sous-chaîne). Aucune tolérance aux fautes.  
- **Équilibré** : ajoute le fuzzy sur **le message entier** avec `partial_ratio ≥ 88`. Bon compromis précision/robustesse.  
- **Agressif** : fuzzy plus tolérant (`partial_ratio ≥ 82`). Plus de rappels, risque de faux positifs accru.
"""
)

if not rf_installed and selected_key != "off":
    st.sidebar.info("RapidFuzz introuvable : le fuzzy est désactivé (mode effectif = Désactivé).")


# ========= Main (sans layout wide, interface simple) =========
st.title("Chatbot NER – Modèles Salomon")
st.caption("Exact (PhraseMatcher + sous-chaîne) + Fuzzy global (message entier)")

# Historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and msg.get("entities"):
            html_text = highlight_html(msg["content"], msg["entities"])
            st.markdown(html_text, unsafe_allow_html=True)
        else:
            st.write(msg["content"])

# Saisie
with st.form("inline_chat_form"):
    user_input = st.text_input("Votre message", value="", key="inline_chat_text")
    submitted = st.form_submit_button("Envoyer")

prompt = user_input.strip() if submitted and user_input.strip() else None

if st.button("Vider la conversation"):
    st.session_state.messages = []
    st.session_state["context_canos"] = []

if prompt:
    ents = pipeline.extract(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "entities": ents})
    with st.chat_message("user"):
        html_text = highlight_html(prompt, ents)
        st.markdown(html_text, unsafe_allow_html=True)

    fallback_canos = st.session_state.get("context_canos", [])
    reply = assistant_reply(pipeline.nlp, prompt, ents, fallback_canos=fallback_canos)

    used_canos = list(dict((m.canonical, None) for m in ents).keys()) if ents else list(fallback_canos)
    st.session_state["context_canos"] = used_canos
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

# Inspector (simple)
with st.expander("Inspector"):
    last_user = next((m for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
    if last_user and last_user.get("entities"):
        rows = [m.to_dict() for m in last_user["entities"]]
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("Saisissez un message pour voir les entités reconnues.")
