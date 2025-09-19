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


# ========= Config & chemins =========
st.set_page_config(page_title="Salomon NER Chatbot", layout="wide")
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "models.json"
RESP_PATH = ROOT / "data" / "responses.json"


# ========= NER intégré =========
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


class NERPipeline:
    """
    Pipeline NER simplifié :
    - Exact : spaCy PhraseMatcher (tous les alias de models.json)
    - Fuzzy : RapidFuzz (WRatio) sur un lexique d’alias (score normalisé 0..1)
    - Fenêtre n-gram interne fixée à 5 (non exposée dans l’UI)
    """

    def __init__(self, data_path: Path, fuzzy_threshold: int = 88) -> None:
        self.data_path = data_path

        # spaCy : modèle FR si dispo, sinon "blank"
        try:
            self.nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner", "lemmatizer", "tagger"])
        except Exception:
            self.nlp = spacy.blank("fr")

        # RapidFuzz dispo ?
        try:
            from rapidfuzz import process as _p, fuzz as _f  # noqa: F401
            self.has_rapidfuzz = True
        except Exception:
            self.has_rapidfuzz = False

        # Réglages fuzzy (pilotés par presets / UI)
        self.fuzzy_preset: str = "balanced" if self.has_rapidfuzz else "off"
        self.fuzzy_threshold: int = fuzzy_threshold
        self.enable_fuzzy: bool = bool(self.has_rapidfuzz)
        self.min_fuzzy_span_len: int = 5            # longueur min sans espaces
        self.require_keyword_or_digit: bool = True  # garde-fou
        self.keyword_regex_str: str = r"(speed|cross|ultra|sense|ride|xa|pro|x\s?ultra|supercross|speedcross|super|x\s?pro)"

        # Index / lexique
        self._models_data: Dict[str, dict] = {}
        self._label_by_cano: Dict[str, str] = {}
        self._aliases_by_cano: Dict[str, List[str]] = {}
        self._lexicon: List[str] = []
        self._cano_by_alias_low: Dict[str, str] = {}

        # Matcher exact
        self._pm: Optional[PhraseMatcher] = None

        self.reload()
        self.set_fuzzy_preset(self.fuzzy_preset)

    # ----- Presets (Off / Balanced / Aggressive) -----
    def set_fuzzy_preset(self, key: str) -> None:
        """Modes et valeurs :
        - off        -> enable=False, seuil=100, min_len=5
        - balanced   -> enable=True,  seuil=88,  min_len=5
        - aggressive -> enable=True,  seuil=82,  min_len=4
        """
        if not self.has_rapidfuzz:
            self.enable_fuzzy = False
            self.fuzzy_preset = "off"
            return

        presets = {
            "off":        {"enable": False, "thr": 100, "min_len": 5},
            "balanced":   {"enable": True,  "thr": 88,  "min_len": 5},
            "aggressive": {"enable": True,  "thr": 82,  "min_len": 4},
        }
        key = (key or "balanced").lower()
        p = presets.get(key, presets["balanced"])

        self.enable_fuzzy = p["enable"]
        self.fuzzy_threshold = p["thr"]
        self.min_fuzzy_span_len = p["min_len"]
        self.require_keyword_or_digit = True
        self.fuzzy_preset = key

    def set_threshold(self, value: int) -> None:
        """Override manuel du seuil WRatio (0..100)."""
        self.fuzzy_threshold = int(value)

    def set_fuzzy_options(
        self,
        enable_fuzzy: bool,
        min_span_len: int,
        require_kw_or_digit: bool,
        keyword_regex_str: str,
    ) -> None:
        """Options avancées (sans n-gram côté UI)."""
        self.enable_fuzzy = bool(enable_fuzzy and self.has_rapidfuzz)
        self.min_fuzzy_span_len = int(min_span_len)
        self.require_keyword_or_digit = bool(require_kw_or_digit)
        self.keyword_regex_str = keyword_regex_str or self.keyword_regex_str

    # ----- Données & règles -----
    def reload(self) -> None:
        self._load_entities()
        self._build_phrase_matcher()

    def _load_entities(self) -> None:
        self._models_data.clear()
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

            self._models_data[cano] = ent
            self._label_by_cano[cano] = label
            self._aliases_by_cano[cano] = sorted(aliases, key=len, reverse=True)

            for a in aliases:
                self._lexicon.append(a)
                self._cano_by_alias_low[a.lower()] = cano

        # Dédoublonne lexique
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

        # Exact (PhraseMatcher)
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

        # Fuzzy (RapidFuzz) — si activé
        if self.enable_fuzzy and self.has_rapidfuzz:
            matches.extend(self._fuzzy_spans(doc))

        # Résolution des chevauchements
        matches = self._dedupe_overlaps(matches)
        return sorted(matches, key=lambda m: (m.start, -m.end))

    def _fuzzy_spans(self, doc) -> List[EntityMatch]:
        """Fenêtre n-gram FIXÉE à 5 (non exposée). Score = WRatio/100."""
        from rapidfuzz import process, fuzz  # type: ignore

        out: List[EntityMatch] = []
        tokens = list(doc)
        re_kw = re.compile(self.keyword_regex_str, re.I) if self.keyword_regex_str else None

        MAX_NGRAM = 5  # fixé

        for i in range(len(tokens)):
            for n in range(1, MAX_NGRAM + 1):
                j = i + n
                if j > len(tokens):
                    break
                span = doc[tokens[i].i : tokens[j - 1].i + 1]
                s = span.text.strip()
                if not s:
                    continue

                # Garde-fous
                if len(s.replace(" ", "")) < self.min_fuzzy_span_len:
                    continue
                if self.require_keyword_or_digit:
                    if not (any(ch.isdigit() for ch in s) or (re_kw and re_kw.search(s))):
                        continue

                best = process.extractOne(s, self._lexicon, scorer=fuzz.WRatio, score_cutoff=self.fuzzy_threshold)
                if not best:
                    continue
                best_alias, score = best[0], float(best[1])
                cano = self._cano_by_alias_low.get(best_alias.lower())
                if not cano:
                    continue
                label = self._label_by_cano.get(cano, "MODEL")

                out.append(
                    EntityMatch(
                        text=span.text,
                        start=span.start_char,
                        end=span.end_char,
                        label=label,
                        canonical=cano,
                        method="fuzzy",
                        score=max(0.0, min(1.0, score / 100.0)),  # normalisé
                    )
                )
        return out

    def _dedupe_overlaps(self, items: List[EntityMatch]) -> List[EntityMatch]:
        if not items:
            return []
        # Trier: exact en premier, puis meilleur score, puis plus long
        items = sorted(
            items,
            key=lambda m: (
                0 if m.method == "exact" else 1,
                -(m.score),
                -(m.end - m.start),
            ),
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


# ========= Utils UI =========
def highlight_html(text: str, matches: List[EntityMatch]) -> str:
    """Version HTML avec surlignage + tooltip score/méthode."""
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


# ========= Helpers (avant l’UI !) =========
def load_models_index() -> dict:
    """Index {canonical: {category, label, url}} depuis models.json."""
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
    """Charge responses.json ou retourne une config par défaut (texte + URL)."""
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


def detect_intent(text: str, cfg: dict) -> Optional[str]:
    """Retourne l'intention détectée (nom) ou None si pas trouvée (PhraseMatcher + petits fallbacks)."""
    low = (text or "").lower()

    # spaCy PhraseMatcher (mots-clés)
    pm = PhraseMatcher(pipeline.nlp.vocab, attr="LOWER")
    label_to_intent = {}
    for it in cfg.get("intents", []):
        name = it.get("name")
        kws = [kw for kw in it.get("keywords", []) if kw]
        if not kws:
            continue
        patterns = [pipeline.nlp.make_doc(kw) for kw in kws]
        pm.add(f"INTENT::{name}", patterns)
        label_to_intent[f"INTENT::{name}"] = name
    doc = pipeline.nlp.make_doc(text or "")
    matches = pm(doc)
    matched = [label_to_intent.get(pipeline.nlp.vocab.strings[m_id]) for m_id, _, _ in matches]
    matched = [m for m in matched if m]

    if matched:
        priority = ["advantages", "price", "waterproof", "terrain"]
        for p in priority:
            if p in matched:
                return p
        return matched[0]

    # Fallback exact (contain)
    for it in cfg.get("intents", []):
        for kw in it.get("keywords", []):
            if (kw or "").lower() in low:
                return it.get("name")

    # Fallback fuzzy (RapidFuzz) sur les mots-clés
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
    """Supporte {text,url} et string. Fallback URL depuis models.json si absente."""
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


def assistant_reply(user_text: str, matches: List[EntityMatch], fallback_canos: list[str] | None = None) -> str:
    """Réponse déterministe combinant intention + NER."""
    models_idx = load_models_index()
    cfg = load_responses_config()

    # Canonical uniques
    if matches:
        canos = list(dict((m.canonical, None) for m in matches).keys())
    elif fallback_canos:
        canos = list(dict((c, None) for c in fallback_canos).keys())
    else:
        canos = []

    # Intention
    intent = detect_intent(user_text, cfg)

    # Pas de modèle mais une intention générique ?
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

    # On a une intention → chercher une réponse
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

    # Plusieurs modèles
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


# ========= Session state =========
if "pipeline" not in st.session_state:
    st.session_state.pipeline = NERPipeline(DATA_PATH, fuzzy_threshold=88)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "context_canos" not in st.session_state:
    st.session_state.context_canos = []

pipeline: NERPipeline = st.session_state.pipeline


# ========= Sidebar =========
st.sidebar.header("Paramètres")

# RapidFuzz dispo ?
rf_installed = getattr(pipeline, "has_rapidfuzz", False)

# Presets
preset_labels = {"off": "Désactivé", "balanced": "Équilibré", "aggressive": "Agressif (fautes)"}
label_to_key = {v: k for k, v in preset_labels.items()}
cur_preset_key = getattr(pipeline, "fuzzy_preset", "balanced")
if not rf_installed:
    cur_preset_key = "off"

old_preset = cur_preset_key
preset_choice = st.sidebar.radio(
    "Tolérance aux fautes",
    options=list(preset_labels.values()),
    index=[*preset_labels].index(cur_preset_key),
    help="Profils: Désactivé (aucun fuzzy) · Équilibré (WRatio≥88) · Agressif (WRatio≥82).",
)
pipeline.set_fuzzy_preset(label_to_key[preset_choice])
if label_to_key[preset_choice] != old_preset:
    st.rerun()

# Seuil fuzzy (override manuel après preset)
threshold = st.sidebar.slider(
    "Seuil fuzzy (WRatio)",
    min_value=70,
    max_value=100,
    value=pipeline.fuzzy_threshold,
    step=1,
    help="Plus le seuil est haut, moins il y a de faux positifs. Par défaut: 88 (équilibré) / 82 (agressif).",
)
if threshold != pipeline.fuzzy_threshold:
    pipeline.set_threshold(threshold)
    st.rerun()

# Options avancées (sans n-gram)
with st.sidebar.expander("Options avancées (fuzzy)", expanded=False):
    old_enable = bool(getattr(pipeline, "enable_fuzzy", False))
    enable_fuzzy_ui = st.checkbox(
        "Activer le fuzzy matching",
        value=(getattr(pipeline, "enable_fuzzy", False) and rf_installed),
        disabled=not rf_installed,
        help="Active/désactive l’appariement approximatif (RapidFuzz).",
    )
    min_len_ui = st.slider(
        "Longueur minimale d'un span (sans espaces)",
        min_value=3,
        max_value=20,
        value=getattr(pipeline, "min_fuzzy_span_len", 5),
        step=1,
        help="Ignore les spans trop courts.",
    )
    require_ui = st.checkbox(
        "Exiger mot-clé ou chiffre",
        value=getattr(pipeline, "require_keyword_or_digit", True),
        help="Réduit les faux positifs (exige un chiffre ou un mot-clé: ultra, speed, pro, …).",
    )
    kw_regex_ui = st.text_input(
        "Regex mots-clés",
        value=getattr(
            pipeline,
            "keyword_regex_str",
            r"(speed|cross|ultra|sense|ride|xa|pro|x\s?ultra|supercross|speedcross|super|x\s?pro)",
        ),
        help="Définit les tokens ‘marque/modèle’ requis quand l’option est activée.",
    )

    pipeline.set_fuzzy_options(
        enable_fuzzy=enable_fuzzy_ui,
        min_span_len=min_len_ui,
        require_kw_or_digit=require_ui,
        keyword_regex_str=kw_regex_ui,
    )

    if enable_fuzzy_ui != old_enable:
        st.rerun()

# Aide : modes + glossaire
with st.sidebar.expander("Aide · Modes & valeurs", expanded=False):
    st.markdown(
        """
**Modes**  
- **Désactivé** : pas d’approximation (fuzzy off).  
- **Équilibré** : accepte un span si **WRatio ≥ 88** · longueur min **5** (sans espaces) · **mot-clé ou chiffre requis**.  
- **Agressif** : accepte un span si **WRatio ≥ 82** · longueur min **4** · **mot-clé ou chiffre requis**.  

**Score Inspector** = `WRatio / 100` (0..1).  
Le **WRatio** (RapidFuzz) est basé sur la **distance d’édition** (pas un cosinus).
"""
    )

with st.sidebar.expander("Glossaire & Explications", expanded=False):
    st.markdown(
        """
- **Fenêtre (n-gram)** : taille des séquences de tokens testées par le fuzzy. Ici **fixée à 5** en interne
  (on génère automatiquement 1..5 tokens : ex. “speedcross”, “speedcross 6”, …).  
- **Span** : sous-chaîne continue du texte évaluée par le matcher (avec positions start/end).  
- **Dictionnaire / Lexique** : liste de **tous les alias** issus de `models.json` (y compris le nom canonique).  
  C’est contre **ce lexique** que RapidFuzz calcule la similarité.  
- **Alias → Canonique** : si l’alias “Speed Cross 6” matche, on remonte vers **“Salomon Speedcross 6”**.  
- **Garde-fou** “mot-clé ou chiffre” : ne teste en fuzzy que les spans contenant un chiffre (ex. *6*)  
  **ou** un mot-clé (ultra, speed, pro, …) pour réduire les faux positifs.  
- **Exact vs Fuzzy** : exact (PhraseMatcher) → score **1.0** ; fuzzy (RapidFuzz) → score **WRatio/100**.
"""
    )

# ========= MAIN =========
st.title("Chatbot NER – Modèles Salomon")
st.caption("Démonstrateur pédagogique : NER par règles + fuzzy matching (simplifié)")

col_left, c_
