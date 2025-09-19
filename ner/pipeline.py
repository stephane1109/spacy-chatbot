from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import spacy
from spacy.pipeline import EntityRuler
import re
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    process = None
    fuzz = None
    HAS_RAPIDFUZZ = False


@dataclass
class EntityMatch:
    start: int
    end: int
    text: str
    label: str
    canonical: str
    method: str  # 'ruler' | 'fuzzy'
    score: float
    pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NERPipeline:
    def __init__(
        self,
        models_path: Path,
        fuzzy_threshold: int = 88,
        max_ngram: int = 5,
        enable_fuzzy: bool = True,
        min_fuzzy_span_len: int = 5,
        require_keyword_or_digit: bool = True,
        keyword_regex: Optional[str] = None,
        fuzzy_preset: str = "balanced",
    ) -> None:
        self.models_path = Path(models_path)
        self.fuzzy_threshold = fuzzy_threshold
        self.max_ngram = max_ngram
        self.enable_fuzzy = bool(enable_fuzzy and HAS_RAPIDFUZZ)
        self.min_fuzzy_span_len = int(min_fuzzy_span_len)
        self.require_keyword_or_digit = bool(require_keyword_or_digit)
        self.keyword_regex_str = (
            keyword_regex
            or r"(speed|cross|ultra|sense|ride|xa|pro|x\s?ultra|supercross|speedcross|super|x\s?pro)"
        )
        self.keyword_regex = re.compile(self.keyword_regex_str, re.IGNORECASE)
        self.has_rapidfuzz = HAS_RAPIDFUZZ
        self.fuzzy_preset = fuzzy_preset
        self._scorers = []
        if HAS_RAPIDFUZZ:
            # Par défaut (balanced): WRatio uniquement
            self._scorers = [fuzz.WRatio]
        self._load_resources()

    def _load_resources(self) -> None:
        # Charger le JSON d'entités
        with open(self.models_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Construire le mapping synonymes -> canonique et les patterns du Ruler
        self.syn_to_canonical: Dict[str, str] = {}
        patterns: List[Dict[str, Any]] = []
        for ent in self.data.get("entities", []):
            label = ent.get("label", "MODEL")
            canonical = ent["canonical"]
            names = set([canonical] + ent.get("synonyms", []))
            for name in names:
                norm = name.strip()
                if not norm:
                    continue
                self.syn_to_canonical[norm.lower()] = canonical
                patterns.append({"label": label, "pattern": norm})

        # Créer un pipeline spaCy FR minimal avec EntityRuler
        self.nlp = spacy.blank("fr")
        self.nlp.max_length = 2_000_000
        ruler: EntityRuler = self.nlp.add_pipe(
            "entity_ruler",
            config={
                "phrase_matcher_attr": "LOWER",
                "overwrite_ents": True,
            },
        )
        ruler.add_patterns(patterns)

    def reload(self) -> None:
        self._load_resources()

    def set_threshold(self, threshold: int) -> None:
        self.fuzzy_threshold = int(threshold)

    def set_fuzzy_options(
        self,
        enable_fuzzy: Optional[bool] = None,
        min_span_len: Optional[int] = None,
        require_kw_or_digit: Optional[bool] = None,
        keyword_regex_str: Optional[str] = None,
        max_ngram: Optional[int] = None,
    ) -> None:
        if enable_fuzzy is not None:
            self.enable_fuzzy = bool(enable_fuzzy and HAS_RAPIDFUZZ)
        if min_span_len is not None:
            self.min_fuzzy_span_len = int(min_span_len)
        if require_kw_or_digit is not None:
            self.require_keyword_or_digit = bool(require_kw_or_digit)
        if keyword_regex_str is not None:
            old_str = getattr(self, "keyword_regex_str", r"(speed|cross|ultra|sense|ride|xa|pro)")
            try:
                self.keyword_regex = re.compile(keyword_regex_str, re.IGNORECASE)
                self.keyword_regex_str = keyword_regex_str
            except re.error:
                # Restaurer l'ancienne regex si la nouvelle est invalide
                self.keyword_regex = re.compile(old_str, re.IGNORECASE)
                self.keyword_regex_str = old_str
        if max_ngram is not None:
            self.max_ngram = int(max_ngram)

    def set_fuzzy_preset(self, preset: str) -> None:
        preset = (preset or "").lower()
        self.fuzzy_preset = preset
        if not HAS_RAPIDFUZZ:
            self.enable_fuzzy = False
            return
        if preset in ("off", "disabled", "desactive", "désactivé", "désactive"):
            self.enable_fuzzy = False
            return
        self.enable_fuzzy = True
        if preset in ("aggressive", "agressif", "agressif (fautes)"):
            self.fuzzy_threshold = 83
            self.min_fuzzy_span_len = 4
            self.require_keyword_or_digit = False
            self.max_ngram = 6
            self._scorers = [fuzz.WRatio, fuzz.token_set_ratio, fuzz.partial_ratio]
        else:
            # balanced/default
            if self.fuzzy_threshold < 80:
                self.fuzzy_threshold = 88
            self.min_fuzzy_span_len = max(4, self.min_fuzzy_span_len)
            self.require_keyword_or_digit = True
            self.max_ngram = max(5, self.max_ngram)
            self._scorers = [fuzz.WRatio]

    def extract(self, text: str) -> List[EntityMatch]:
        if not text:
            return []

        doc = self.nlp(text)
        matches: List[EntityMatch] = []

        # 1) Règles exactes (EntityRuler)
        for ent in doc.ents:
            canonical = self.syn_to_canonical.get(ent.text.lower(), ent.text)
            matches.append(
                EntityMatch(
                    start=ent.start_char,
                    end=ent.end_char,
                    text=ent.text,
                    label=ent.label_,
                    canonical=canonical,
                    method="ruler",
                    score=1.0,
                    pattern=ent.text,
                )
            )

        occupied = [(m.start, m.end) for m in matches]

        # 2) Fuzzy matching (n-grammes sur 1..max_ngram)
        #    On compare chaque fenêtre au lexique (synonymes+canoniques)
        if self.enable_fuzzy and HAS_RAPIDFUZZ and self.fuzzy_threshold > 0:
            tokens = self.nlp.make_doc(text)
            keys = list(self.syn_to_canonical.keys())
            for i in range(len(tokens)):
                for j in range(i + 1, min(len(tokens), i + self.max_ngram) + 1):
                    span = tokens[i:j]
                    s, e = span.start_char, span.end_char
                    if any(self._spans_overlap((s, e), occ) for occ in occupied):
                        continue
                    span_text = span.text
                    if not span_text.strip():
                        continue
                    # Filtre: longueur minimale (sans espaces)
                    if len("".join(ch for ch in span_text if not ch.isspace())) < self.min_fuzzy_span_len:
                        continue
                    # Heuristique: exiger un mot-clé ou un chiffre
                    if self.require_keyword_or_digit:
                        has_digit = any(ch.isdigit() for ch in span_text)
                        has_kw = bool(self.keyword_regex.search(span_text))
                        if not (has_digit or has_kw):
                            continue
                    # Meilleure correspondance parmi les scorers sélectionnés
                    best = None
                    for scorer in self._scorers:
                        cand = process.extractOne(
                            span_text.lower(),
                            keys,
                            scorer=scorer,
                        )
                        if cand is None:
                            continue
                        if (best is None) or (cand[1] > best[1]):
                            best = cand
                    if not best:
                        continue
                    key, score, _ = best
                    if score >= self.fuzzy_threshold:
                        canonical = self.syn_to_canonical.get(key, key)
                        matches.append(
                            EntityMatch(
                                start=s,
                                end=e,
                                text=span_text,
                                label="MODEL",
                                canonical=canonical,
                                method="fuzzy",
                                score=float(score) / 100.0,
                                pattern=key,
                            )
                        )
                        occupied.append((s, e))

        # Trier par position
        matches.sort(key=lambda m: (m.start, -m.end))
        return matches

    def _spans_overlap(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return max(a[0], b[0]) < min(a[1], b[1])
