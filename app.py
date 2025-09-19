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
# Pas de layout="wide"
st.set_page_config(page_title="Salomon NER Chatbot")
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
    - Fenêtre d’analyse interne : 1 à 5 tokens (fixe)
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

        # Réglages fuzzy (pilotés par l’UI)
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
        """Options avancées (sans contrôle de fenêtre)."""
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
        """Fenêtre d’analyse 1..5 tokens. Score = WRatio/100."""
        from rapidfuzz import process, fuzz  # type: ignore

        out: List[EntityMatch] = []
        tokens = list(doc)
        re_kw = re.compile(self.keyword_regex_str, re.I) if self.keyword_regex_str else None

        MAX_NGRAM = 5

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
    """Retourne l'intention détectée (nom) ou None si pas trouvée (PhraseMatcher + fallbacks)."""
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

# Widgets avec clés stables, sans st.rerun pour éviter les boucles
preset_labels = {"off": "Désactivé", "balanced": "Équilibré", "aggressive": "Agressif"}
label_to_key = {v: k for k, v in preset_labels.items()}

if "ui_preset" not in st.session_state:
    st.session_state.ui_preset = pipeline.fuzzy_preset if rf_installed else "off"
if "ui_threshold" not in st.session_state:
    st.session_state.ui_threshold = pipeline.fuzzy_threshold
if "ui_enable_fuzzy" not in st.session_state:
    st.session_state.ui_enable_fuzzy = bool(pipeline.enable_fuzzy and rf_installed)
if "ui_min_len" not in st.session_state:
    st.session_state.ui_min_len = pipeline.min_fuzzy_span_len
if "ui_require_kw" not in st.session_state:
    st.session_state.ui_require_kw = pipeline.require_keyword_or_digit
if "ui_kw_regex" not in st.session_state:
    st.session_state.ui_kw_regex = pipeline.keyword_regex_str

preset_choice_label = st.sidebar.radio(
    "Tolérance aux fautes",
    options=list(preset_labels.values()),
    index=[*preset_labels].index(preset_labels.get(st.session_state.ui_preset, "Équilibré")),
    help="Off: aucune approximation. Équilibré: WRatio ≥ 88. Agressif: WRatio ≥ 82.",
    key="preset_radio",
)
st.session_state.ui_preset = label_to_key[preset_choice_label]

st.session_state.ui_threshold = st.sidebar.slider(
    "Seuil fuzzy (WRatio)",
    min_value=70,
    max_value=100,
    value=st.session_state.ui_threshold,
    step=1,
    help="Plus le seuil est haut, moins il y a de correspondances approximatives.",
    key="threshold_slider",
)

with st.sidebar.expander("Options avancées", expanded=False):
    st.session_state.ui_enable_fuzzy = st.checkbox(
        "Activer le fuzzy matching",
        value=st.session_state.ui_enable_fuzzy,
        disabled=not rf_installed,
        help="Active/désactive l’appariement approximatif (RapidFuzz).",
        key="enable_fuzzy_cb",
    )
    st.session_state.ui_min_len = st.slider(
        "Longueur minimale d'un span (sans espaces)",
        min_value=3,
        max_value=20,
        value=st.session_state.ui_min_len,
        step=1,
        help="Ignore les spans trop courts.",
        key="min_len_slider",
    )
    st.session_state.ui_require_kw = st.checkbox(
        "Exiger mot-clé ou chiffre",
        value=st.session_state.ui_require_kw,
        help="Réduit les faux positifs (exige un chiffre ou un mot-clé: ultra, speed, pro…).",
        key="require_kw_cb",
    )
    st.session_state.ui_kw_regex = st.text_input(
        "Regex mots-clés",
        value=st.session_state.ui_kw_regex,
        help="Tokens requis quand l’option est activée.",
        key="kw_regex_input",
    )

# Appliquer l’UI au pipeline (ordre: preset -> overrides)
pipeline.set_fuzzy_preset(st.session_state.ui_preset)
pipeline.set_threshold(st.session_state.ui_threshold)
pipeline.set_fuzzy_options(
    enable_fuzzy=st.session_state.ui_enable_fuzzy,
    min_span_len=st.session_state.ui_min_len,
    require_kw_or_digit=st.session_state.ui_require_kw,
    keyword_regex_str=st.session_state.ui_kw_regex,
)

# Info simple demandée
st.sidebar.caption("Fenêtre d’analyse fixe : 1 à 5 tokens.")

# RapidFuzz info
if not rf_installed:
    st.sidebar.warning("RapidFuzz n'est pas installé. Le fuzzy matching est indisponible.")

# Bouton reload
if st.sidebar.button("Recharger les entités (models.json)"):
    pipeline.reload()
    st.sidebar.info("Règles rechargées depuis models.json")


# ========= MAIN =========
st.title("Chatbot NER – Modèles Salomon")
st.caption("Reconnaissance de modèles par règles + fuzzy matching (simplifié)")

col_left, col_right = st.columns([2, 1])

with col_left:
    if st.button("Vider la conversation"):
        st.session_state.messages = []
        st.session_state["context_canos"] = []
        st.experimental_rerun()

    # Historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user" and msg.get("entities"):
                html_text = highlight_html(msg["content"], msg["entities"])
                st.markdown(html_text, unsafe_allow_html=True)
            else:
                st.write(msg["content"])

    # Saisie inline
    pending = st.session_state.get("pending_prompt")
    with st.form("inline_chat_form"):
        user_input = st.text_input("Votre message", value="", key="inline_chat_text")
        submitted = st.form_submit_button("Envoyer")
    if pending is not None:
        prompt = pending
        st.session_state["pending_prompt"] = None
    elif submitted and user_input.strip():
        prompt = user_input.strip()
    else:
        prompt = None

    if prompt:
        # Message utilisateur
        ents = pipeline.extract(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "entities": ents})
        with st.chat_message("user"):
            html_text = highlight_html(prompt, ents)
            st.markdown(html_text, unsafe_allow_html=True)

        # Réponse assistant (déterministe minimaliste)
        fallback_canos = st.session_state.get("context_canos", [])
        reply = assistant_reply(prompt, ents, fallback_canos=fallback_canos)

        used_canos = list(dict((m.canonical, None) for m in ents).keys()) if ents else list(fallback_canos)
        st.session_state["context_canos"] = used_canos
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

with col_right:
    st.subheader("Inspector")
    if st.session_state.messages:
        last_user = next((m for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
        if last_user and last_user.get("entities"):
            rows = [m.to_dict() for m in last_user["entities"]]
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("Saisissez un message pour voir les entités reconnues.")
    else:
        st.info("Pas encore de messages.")
