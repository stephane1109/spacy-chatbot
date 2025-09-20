from __future__ import annotations

# ============== Imports ==============
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
import json
import html

import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher

# RapidFuzz (fuzzy WRatio)
try:
    from rapidfuzz import process, fuzz  # type: ignore
    RAPIDFUZZ_OK = True
except Exception:
    RAPIDFUZZ_OK = False


# ============== Config ==============
st.set_page_config(page_title="Salomon NER • Scores WRatio")  # pas de wide
RACINE = Path(__file__).parent
CHEMIN_JSON = RACINE / "data" / "models.json"


# ============== Données ==============
@dataclass
class CorrespondanceEntite:
    texte: str
    debut: int
    fin: int
    etiquette: str
    canonique: str
    methode: str   # "exact"
    score: float   # 0..1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(self.score, 3)
        return d


# ============== Chargeurs (cache sûrs) ==============
@st.cache_resource(show_spinner=False)
def charger_nlp():
    """Charge spaCy FR (ou blank FR)."""
    try:
        return spacy.load("fr_core_news_sm", disable=["parser", "ner", "lemmatizer", "tagger"])
    except Exception:
        return spacy.blank("fr")

@st.cache_resource(show_spinner=False)
def lire_json_entites(chemin_str: str) -> dict:
    """
    Lit le JSON des modèles. AUCUN fallback ici.
    Retourne le dict si OK, sinon None (on gèrera hors du cache).
    """
    p = Path(chemin_str)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("entities"):
            return data
    except Exception:
        return None
    return None

@st.cache_resource(show_spinner=False)
def construire_phrase_matcher_depuis_json(entites_json: str):
    """
    Construit PhraseMatcher à partir d'une chaîne JSON (hashable pour le cache).
    """
    nlp = charger_nlp()
    data = json.loads(entites_json)

    pm = PhraseMatcher(nlp.vocab, attr="LOWER")
    alias_vers_canonique: Dict[str, str] = {}
    meta_par_canonique: Dict[str, Dict] = {}
    tous_alias: List[str] = []

    for ent in data.get("entities", []):
        cano = (ent.get("canonical") or "").strip()
        if not cano:
            continue
        etiquette = ent.get("label") or "MODEL"
        meta_par_canonique[cano] = {
            "label": etiquette,
            "category": ent.get("category"),
            "url": ent.get("url"),
        }
        variantes = [cano] + list(ent.get("aliases", []) or [])
        motifs = []
        for a in variantes:
            a = (a or "").strip()
            if not a:
                continue
            alias_vers_canonique[a.lower()] = cano
            tous_alias.append(a)
            motifs.append(nlp.make_doc(a))
        if motifs:
            pm.add(f"CANO::{cano}", motifs)

    # dédoublonne la liste brute d’alias (utile pour RapidFuzz)
    tous_alias = list(dict.fromkeys(tous_alias))
    return pm, alias_vers_canonique, meta_par_canonique, tous_alias


# ============== Extraction exacte (surlignage) ==============
def extraire_exacts(nlp, pm, texte: str, meta_par_canonique: Dict[str, Dict]) -> List[CorrespondanceEntite]:
    if not texte:
        return []
    doc = nlp.make_doc(texte)
    sorties: List[CorrespondanceEntite] = []
    for match_id, start, end in pm(doc):
        regle = nlp.vocab.strings[match_id]  # "CANO::<canonical>"
        cano = regle.split("::", 1)[1] if "::" in regle else None
        span = doc[start:end]
        if not cano or not span.text.strip():
            continue
        etiquette = (meta_par_canonique.get(cano) or {}).get("label", "MODEL")
        sorties.append(
            CorrespondanceEntite(
                texte=span.text,
                debut=span.start_char,
                fin=span.end_char,
                etiquette=etiquette,
                canonique=cano,
                methode="exact",
                score=1.0,
            )
        )
    return sorties


# ============== Scores WRatio (sans normalisation) ==============
def calculer_scores_wratio(texte: str,
                           tous_alias: List[str],
                           alias_vers_canonique: Dict[str, str]) -> List[Dict]:
    """
    Calcule WRatio(user_texte, alias) pour CHAQUE alias tel quel (aucune normalisation),
    puis agrège au niveau 'canonical' en conservant le meilleur alias.
    Retourne [{canonical, meilleur_alias, score}] trié par score desc.
    """
    if not texte.strip():
        # rien à scorer
        meilleurs: Dict[str, Dict] = {}
        for al in tous_alias:
            cano = alias_vers_canonique.get(al.lower())
            if cano and cano not in meilleurs:
                meilleurs[cano] = {"canonical": cano, "meilleur_alias": al, "score": 0.0}
        return sorted(meilleurs.values(), key=lambda d: (-d["score"], d["canonical"]))

    if not RAPIDFUZZ_OK:
        st.warning("RapidFuzz n'est pas installé : les scores WRatio seront à 0.")
        meilleurs: Dict[str, Dict] = {}
        for al in tous_alias:
            cano = alias_vers_canonique.get(al.lower())
            if cano and cano not in meilleurs:
                meilleurs[cano] = {"canonical": cano, "meilleur_alias": al, "score": 0.0}
        return sorted(meilleurs.values(), key=lambda d: (-d["score"], d["canonical"]))

    # scores alias-level
    resultats = process.extract(
        texte,
        tous_alias,
        scorer=fuzz.WRatio,
        limit=len(tous_alias)
    )

    # agrégation par canonique
    meilleurs: Dict[str, Dict] = {}
    for alias, score, _ in resultats:
        cano = alias_vers_canonique.get(alias.lower())
        if not cano:
            continue
        cur = meilleurs.get(cano)
        if cur is None or score > cur["score"]:
            meilleurs[cano] = {"canonical": cano, "meilleur_alias": alias, "score": float(score)}

    return sorted(meilleurs.values(), key=lambda d: (-d["score"], d["canonical"]))


# ============== UI helpers ==============
def surligner_html(texte: str, matchs: List[CorrespondanceEntite]) -> str:
    if not texte:
        return ""
    spans = sorted(matchs, key=lambda m: m.debut)
    out = []
    cur = 0
    for m in spans:
        if m.debut > cur:
            out.append(html.escape(texte[cur:m.debut]))
        titre = f"{m.etiquette} → {m.canonique} (exact)"
        out.append(
            f"<mark style='background-color:#fff3cd; padding:0 2px; border-radius:2px' "
            f"title='{html.escape(titre)}'>{html.escape(texte[m.debut:m.fin])}</mark>"
        )
        cur = m.fin
    if cur < len(texte):
        out.append(html.escape(texte[cur:]))
    return "".join(out)


def construire_reponse_assistant(texte: str,
                                 exacts: List[CorrespondanceEntite],
                                 scores: List[Dict],
                                 meta_par_canonique: Dict[str, Dict]) -> str:
    """
    Réponse succincte :
    - si exact → confirme le(s) modèle(s) détecté(s) (+ URL si dispo)
    - sinon → propose le meilleur candidat WRatio si score >= 80
    """
    if exacts:
        canos = list(dict((m.canonique, None) for m in exacts).keys())
        if len(canos) == 1:
            cano = canos[0]
            url = (meta_par_canonique.get(cano) or {}).get("url")
            base = f"Modèle détecté : {cano}."
            return base + (f" Fiche produit : {url}" if url else "")
        return "Modèles détectés : " + ", ".join(canos)

    if scores:
        top = scores[0]
        if top["score"] >= 80:
            cano = top["canonical"]
            url = (meta_par_canonique.get(cano) or {}).get("url")
            ans = f"Je pense que vous parlez de « {cano} » (score WRatio {top['score']:.0f})."
            return ans + (f" Fiche produit : {url}" if url else "")
    return "Je n’ai pas reconnu de modèle. Donnez le nom exact (ou une variante proche)."


# ============== Interface principale ==============
st.title("Chat NER (Salomon) + Scores WRatio")
st.caption("• NER exact (PhraseMatcher) • Scores WRatio pour tous les modèles • Réponse après chaque requête")

# Charger JSON (sans fallback) + matcher
nlp = charger_nlp()
entites_data = lire_json_entites(str(CHEMIN_JSON))
if not entites_data:
    st.error(f"Fichier introuvable ou invalide : {CHEMIN_JSON}. Ajoutez un JSON valide (clé 'entities').")
    st.stop()

entites_json = json.dumps(entites_data, sort_keys=True, ensure_ascii=False)
pm, alias_vers_canonique, meta_par_canonique, tous_alias = construire_phrase_matcher_depuis_json(entites_json)

# État
if "historique" not in st.session_state:
    st.session_state.historique = []   # [{role, contenu, entites}]
if "derniers_scores" not in st.session_state:
    st.session_state.derniers_scores = []

# Formulaire
with st.form("form_chat"):
    saisie = st.text_input("Votre message", value="", help="Ex: 'je veux des infos sur la speedcross 6'")
    envoyer = st.form_submit_button("Envoyer")

if st.button("Vider la conversation"):
    st.session_state.historique = []
    st.session_state.derniers_scores = []

# Traitement
if envoyer and saisie.strip():
    texte = saisie.strip()

    # 1) Exact (surlignage)
    entites = extraire_exacts(nlp, pm, texte, meta_par_canonique)
    st.session_state.historique.append({"role": "user", "contenu": texte, "entites": entites})

    # 2) Scores WRatio (tous les modèles, meilleur alias par modèle)
    scores = calculer_scores_wratio(texte, tous_alias, alias_vers_canonique)
    st.session_state.derniers_scores = scores

    # 3) Réponse chatbot
    reponse = construire_reponse_assistant(texte, entites, scores, meta_par_canonique)
    st.session_state.historique.append({"role": "assistant", "contenu": reponse, "entites": []})

# Affichage historique
for msg in st.session_state.historique:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(surligner_html(msg["contenu"], msg.get("entites", [])), unsafe_allow_html=True)
        else:
            st.write(msg["contenu"])

# Tableau des scores (tous les modèles)
st.subheader("Scores de correspondance (WRatio) — Tous les modèles")
if st.session_state.derniers_scores:
    lignes = []
    for item in st.session_state.derniers_scores:
        meta = meta_par_canonique.get(item["canonical"], {})
        lignes.append({
            "Modèle (canonical)": item["canonical"],
            "Alias (meilleur)": item["meilleur_alias"],
            "Score": round(item["score"], 1),
            "Label": meta.get("label"),
            "Catégorie": meta.get("category"),
            "URL": meta.get("url"),
        })
    lignes = sorted(lignes, key=lambda r: (-r["Score"], r["Modèle (canonical)"]))
    st.dataframe(lignes, use_container_width=True)
else:
    st.info("Saisissez un message puis cliquez sur Envoyer pour voir les scores.")
