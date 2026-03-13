#!/usr/bin/env python3
"""
Retrieval core for the v2 WHO knowledge base index.

Designed exclusively for who_knowledge_vec_v2.mv2 — a 132 K-frame memvid index
built from the WHO/MSF Guidelines CDS Chunking Pipeline.  Every design decision
aligns with that pipeline's agreed chunk schema and retrieval_priority defaults.

Key differences from retrieval_core.py (v1):
  - Single operating mode: always returns top-5, always clinical-grade CDS.
  - No strict_mode / strict_cds_mode — quality_flags are observational only.
  - All content_type vocabulary is v2-native (recommendation, implementation,
    etd, methods_pico, background, research_gap, annex, scope).
  - retrieval_priority is read from each hit's metadata and used as a compound
    score multiplier; superseded chunks (rp == 0.0) are hard-dropped.
  - Enriched per-hit response: headings, doc_type, recommendation_strength,
    evidence_certainty, source_url exposed for CDS display.
  - Title matching uses re.search (substring) throughout — v2 titles are full
    Docling heading paths, not short section labels.

Search modes (search_mode parameter):
  lex  — Tantivy BM25 lexical search (fast, no GPU needed)
  sem  — Semantic vector search (requires EmbedGemma embedder)
  rrf  — Reciprocal Rank Fusion of lex + sem (default, best quality)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from kb.embedder import EmbedGemmaEmbedder
except ImportError:
    from embedder import EmbedGemmaEmbedder  # type: ignore[no-redef]

log = logging.getLogger("kb.retrieval_v2")

KB_DIR     = "/var/www/kbToolUseLora/kb"
V2_INDEX   = os.path.join(KB_DIR, "who_knowledge_vec_v2.mv2")
SOURCE_WHO = "WHO Guidelines"

# ── v2 Content-type boost/demote multipliers ──────────────────────────────
# Values align with retrieval_priority defaults from the implementation plan.
# Compound score = raw_score × _V2_CONTENT_BOOST[ct] × retrieval_priority
# (retrieval_priority is read from the hit itself — 0.3 for background,
#  1.0 for recommendation, etc.  Superseded chunks have rp=0.0 and are
#  hard-filtered before this stage.)
_V2_CONTENT_BOOST: Dict[str, float] = {
    "recommendation": 1.50,   # primary actionable CDS output
    "implementation": 1.30,   # next steps, dosing context
    "etd":            0.60,   # evidence summaries — context, not CDS
    "methods_pico":   0.45,   # PICO framing — methodology
    "background":     0.20,   # introductory prose — heavy demote
    "research_gap":   0.15,   # future research — rarely relevant at PoC
    "annex":          0.25,   # supplementary tables — low clinical priority
    "scope":          0.25,   # scope sections — contextual only
}
# Fallback for unknown future types: 1.0 (neutral, no boost or demote)
_V2_CONTENT_BOOST_DEFAULT = 1.0

# ── Action-priority and context-demote type sets ──────────────────────────
_ACTION_PRIORITY_TYPES = {"recommendation", "implementation"}
_ACTION_CONTEXT_DEMOTE = {
    "etd", "methods_pico", "background", "research_gap", "annex", "scope",
}
_ACTION_CONTEXT_DEMOTE_FACTOR = 0.75
_ACTION_BONUS                 = 1.15   # 15% boost for action-priority types
_ACTIONABLE_TEXT_BONUS        = 1.08   # small lift for protocol-like language

# ── Metadata lines to strip from hit snippets ─────────────────────────────
# These are frame-level metadata fields that can bleed into the snippet text.
_METADATA_LINE_PREFIXES = (
    # common to all memvid indexes
    "title: ", "uri: mv2://", "tags: ", "labels: ",
    "chunk_id: ", "content_type: ", "memvid.embedding.", "memvid.",
    # v2-specific metadata fields appended to snippet by the SDK at read time
    "docling_provenance:", "doc_type: ", "source_url: ",
    "is_current: ", "retrieval_priority: ", "token_count: ",
    "content_hash: ", "extractous_metadata:", "headings:",
    "pdf_file:", "page_numbers:",
    # SDK-appended frame fields that leak into the snippet text
    "track: ", "tags:", "labels:", "chunk_id:", "content_hash:",
    "source: lex", "source: sem", "source: rrf",
)

# ── Corruption filter ─────────────────────────────────────────────────────
# Hard-drop hits whose content contains known extraction artifacts.
_CORRUPTION_RE = re.compile(
    r'·i\b|recursively\b|>loc[_\s]\d+|\blocas\b'
    r'|smoker,\s+and\s+a\s+person\s+who\s+is\s+not\s+smoking'
    r'|ganges\s+river'
    r'|methanibutol|diploplohydria|hydratraemia|hypokatropaenia',
    re.IGNORECASE,
)
_CORRUPTION_SOFT_RE = re.compile(
    r'\b(?:AND OR|OR RF|maximum\s+maximum)\b', re.IGNORECASE,
)

# ── Query synonym expansion ───────────────────────────────────────────────
# Synonyms are appended as a dedup suffix — the original query is preserved.
_SYNONYM_MAP: Dict[str, List[str]] = {
    "adrenaline":        ["epinephrine"],
    "epinephrine":       ["adrenaline"],
    "MgSO4":             ["magnesium sulfate"],
    "magnesium sulfate": ["MgSO4"],
    "paracetamol":       ["acetaminophen"],
    "acetaminophen":     ["paracetamol"],
    "PPH":               ["postpartum hemorrhage", "postpartum haemorrhage"],
    "DKA":               ["diabetic ketoacidosis"],
    "norepinephrine":    ["noradrenaline", "vasopressor"],
    "noradrenaline":     ["norepinephrine", "vasopressor"],
    "ACT":               ["artemisinin combination therapy"],
    "amoxycillin":       ["amoxicillin"],
    "rifampicin":        ["rifampin"],
    "rifampin":          ["rifampicin"],
    "cotrimoxazole":     ["TMP-SMX", "trimethoprim"],
    "dysentery":         ["Shigella", "ciprofloxacin"],
    "antivenom":         ["envenomation", "snake bite"],
    "envenomation":      ["antivenom", "snake bite"],
    "SAM":               ["severe acute malnutrition", "RUTF"],
    "RUTF":              ["therapeutic feeding", "severe acute malnutrition"],
    "haloperidol":       ["antipsychotic", "psychosis"],
    "resuscitation":     ["airway breathing circulation"],
    "artemether":        ["ACT", "lumefantrine"],
    "lumefantrine":      ["artemether", "ACT"],
    "oxytocin":          ["uterotonic"],
    "misoprostol":       ["uterotonic"],
    "tamponade":         ["balloon tamponade", "uterine balloon"],
    "vaccine":           ["vaccination", "immunisation"],
    "vaccination":       ["vaccine", "immunisation"],
    "immunisation":      ["vaccine", "vaccination"],
    "anemia":            ["anaemia"],
    "anaemia":           ["anemia"],
    "labour":            ["labor"],
    "labor":             ["labour"],
    "schistosomiasis":   ["bilharzia", "praziquantel"],
    "praziquantel":      ["schistosomiasis", "bilharzia"],
    "hepatotoxicity":    ["DILI", "drug-induced liver injury"],
}
_SYNONYM_DETECT: Dict[str, re.Pattern] = {
    term: re.compile(r'\b' + re.escape(term) + r'\b', re.I)
    for term in _SYNONYM_MAP
}

# ── Action query detection ────────────────────────────────────────────────
_ACTION_DOSE_RE = re.compile(
    r'\b(?:dose|dosage|mg|mcg|units?|administer|prescri(?:be|ption)'
    r'|infusion|IV\b|IM\b|PO\b|antibiotic|antifungal|antiviral'
    r'|vasopressor|first.?line|empiric(?:al)?|regimen'
    r'|oxytocin|misoprostol|uterotonic|artemether|lumefantrine'
    r'|vaccine|vaccination|antihypertensive|transfusion|tamponade'
    r'|praziquantel|labetalol|uterine balloon|rutf|therapeutic feeding)\b',
    re.IGNORECASE,
)
_ACTION_TASK_RE = re.compile(
    r'\b(?:treat(?:ment|ing)?|manage(?:ment)?|protocol|guideline|algorithm'
    r'|first.?line|empiric(?:al)?|referral|transfer|stabiliz(?:e|ation)'
    r'|pre.?referral|prevention|prophylax(?:is)?)\b',
    re.IGNORECASE,
)
_ACTION_CONDITION_RE = re.compile(
    r'\b(?:treat(?:ment|ing)?|manage(?:ment)?|protocol'
    r'|sepsis|malaria|tuberculosis|TB\b|meningitis|pneumonia'
    r'|eclampsia|pre.?eclampsia|DKA|ketoacidosis|anaphylaxis'
    r'|haemorrhage|hemorrhage|PPH|shock|stroke|infection'
    r'|rabies|snake.?bite|hypertension|ana?emia|schistosomiasis'
    r'|appendicitis|heart failure|depression|malnutrition'
    r'|podoconiosis|tungiasis|noma'
    r'|how\s+to\s+treat|how\s+to\s+manage'
    r'|emergency|acute\s+(?:management|treatment)|severe|critical\s+(?:illness|care)'
    r'|hypertensive\s+(?:emergency|crisis)|airway|resuscitation|cardiac\s+arrest)\b',
    re.IGNORECASE,
)

# ── Actionability text signals ────────────────────────────────────────────
_DOSE_NUMBER_RE = re.compile(
    r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|mmol|mL|ml|g/kg|mg/kg|mcg/kg|IU|units?|%)\b'
    r'|\b\d+(?:\.\d+)?\s*(?:mcg|µg)/(?:kg/)?min\b',
    re.IGNORECASE,
)
_ROUTE_ADMIN_RE = re.compile(
    r'\b(?:IV\b|IM\b|PO\b|SC\b|oral(?:ly)?|intravenous|intramuscular'
    r'|subcutaneous|infusion|injection)\b',
    re.IGNORECASE,
)
_FREQUENCY_RE = re.compile(
    r'\b(?:daily|twice|q\d+[-–]?\d*h?|every\s+\d+(?:\s*[-–]\s*\d+)?\s*h(?:ours?)?'
    r'|stat\b|TID|BID|QID|q8h|q6h|q12h|once\s+daily|four\s+times)\b',
    re.IGNORECASE,
)
_PROTOCOL_TEXT_RE = re.compile(
    r'\b(?:recommend(?:ed|ation)?|first.?line|regimen|dos(?:e|age)'
    r'|mg(?:/kg)?|IV\b|IM\b|oral|administer|give|refer|referral'
    r'|danger signs?|monitor(?:ing)?|algorithm|protocol)\b',
    re.IGNORECASE,
)
_RESCUE_RECOMMEND_RE = re.compile(
    r'(?:(?<!not\s)\brecommend(?:ed|ation)?\b|\bshould\b|\bmust\b|\badminister\b|\bgive\b|\btreat\b'
    r'|first.?line|preferred|indicated)\b',
    re.IGNORECASE,
)

# ── Task-slot: preferred content types per task ───────────────────────────
# v2 types only — no v1 legacy strings.
_TASK_PREFERRED_TYPES: Dict[str, set] = {
    "dose":       {"recommendation", "implementation"},
    "diagnosis":  {"recommendation"},
    "first_line": {"recommendation", "implementation"},
    "referral":   {"recommendation", "implementation"},
    "prevention": {"recommendation"},
}

# ── Source diversity cap ──────────────────────────────────────────────────
_SOURCE_DIVERSITY_MAX = 2   # allow up to 2 hits per PDF before penalising
_SOURCE_DIVERSITY_PENALTY = 0.50

# ── Domain coherence ──────────────────────────────────────────────────────
# For each active condition in the query, penalise hits with zero term overlap.
_DOMAIN_COHERENCE_PENALTY             = 0.65
_DOMAIN_COHERENCE_PENALTY_ACTION_TYPE = 0.20   # harsher for actionable off-target

_CONDITION_COHERENCE: List[Tuple[re.Pattern, re.Pattern]] = [
    (re.compile(r'\b(?:malaria|plasmodium|artesunate|artemisinin)\b', re.I),
     re.compile(r'\b(?:malaria|plasmodium|artemisinin|artesunate|quinine|chloroquine|falciparum|vivax|ACT)\b', re.I)),
    (re.compile(r'\b(?:tuberculosis|TB\b|RHZE|isoniazid)\b', re.I),
     re.compile(r'\b(?:tuberculosis|TB|isoniazid|rifampicin|rifampin|pyrazinamide|ethambutol|DOTS|mycobacterium|RHZE)\b', re.I)),
    (re.compile(r'\b(?:sepsis|septic\s+shock)\b', re.I),
     re.compile(r'\b(?:sepsis|septic|bacteremia|bacteraemia|vasopressor|blood\s+culture|broad.spectrum)\b', re.I)),
    (re.compile(r'\b(?:anaphylaxis|anaphylactic)\b', re.I),
     re.compile(r'\b(?:anaphylaxis|anaphylactic|epinephrine|adrenaline|urticaria|angioedema|allergic)\b', re.I)),
    (re.compile(r'\b(?:meningitis)\b', re.I),
     re.compile(r'\b(?:meningitis|cerebrospinal|CSF|lumbar|ceftriaxone|dexamethasone|meningococcal)\b', re.I)),
    (re.compile(r'\b(?:dengue)\b', re.I),
     re.compile(r'\b(?:dengue|NS1|platelet|haematocrit|hematocrit)\b', re.I)),
    (re.compile(r'\b(?:cholera|vibrio)\b', re.I),
     re.compile(r'\b(?:cholera|vibrio|rice[- ]water stool|acute watery diarrh|ORS|Ringer.s lactate|doxycycline|azithromycin)\b', re.I)),
    (re.compile(r'\b(?:burn(?:s|ed)?|thermal injury|TBSA|parkland)\b', re.I),
     re.compile(r'\b(?:burn(?:s|ed)?|thermal injury|TBSA|parkland|rule of 9s|lund[- ]browder|dressing|silver sulfadiazine)\b', re.I)),
    (re.compile(r'\b(?:rabies|post.?exposure prophylaxis|PEP)\b', re.I),
     re.compile(r'\b(?:rabies|immunoglobulin|vaccine|post.?exposure|bite wound)\b', re.I)),
    (re.compile(r'\b(?:ana?emia|ha?emoglobin|transfusion)\b', re.I),
     re.compile(r'\b(?:ana?emia|ha?emoglobin|transfusion|packed cells?|whole blood|blood transfusion|iron)\b', re.I)),
    (re.compile(r'\b(?:hypertension|hypertensive)\b', re.I),
     re.compile(r'\b(?:hypertension|hypertensive|blood pressure|labetalol|hydralazine|amlodipine|thiazide|ace inhibitor)\b', re.I)),
    (re.compile(r'\b(?:schistosomiasis|bilharzia|praziquantel)\b', re.I),
     re.compile(r'\b(?:schistosomiasis|bilharzia|praziquantel|schistosome|helminth)\b', re.I)),
    (re.compile(r'\b(?:appendicitis|appendectomy|appendicectomy)\b', re.I),
     re.compile(r'\b(?:appendicitis|appendectomy|appendicectomy|periappendic)\b', re.I)),
    (re.compile(r'\b(?:heart failure|cardiac failure|furosemide)\b', re.I),
     re.compile(r'\b(?:heart failure|cardiac failure|furosemide|diuretic|pulmonary oedema|pulmonary edema)\b', re.I)),
    (re.compile(r'\b(?:depression|depressive|ssri|fluoxetine)\b', re.I),
     re.compile(r'\b(?:depression|depressive|ssri|fluoxetine|antidepressant|psychological)\b', re.I)),
    (re.compile(r'\b(?:buruli|mycobacterium ulcerans)\b', re.I),
     re.compile(r'\b(?:buruli|mycobacterium ulcerans|ulcerans|rifampicin|streptomycin|clarithromycin|wound care)\b', re.I)),
    (re.compile(r'\b(?:podoconiosis)\b', re.I),
     re.compile(r'\b(?:podoconiosis|non[- ]filarial elephantiasis|foot hygiene|compression|lymphoedema)\b', re.I)),
    (re.compile(r'\b(?:tungiasis|tunga penetrans|jigger)\b', re.I),
     re.compile(r'\b(?:tungiasis|tunga penetrans|jigger|flea extraction|wound care|secondary infection)\b', re.I)),
    (re.compile(r'\b(?:noma|cancrum oris|necroti[sz]ing gingivitis)\b', re.I),
     re.compile(r'\b(?:noma|cancrum oris|necroti[sz]ing gingivitis|metronidazole|debridement|oral hygiene)\b', re.I)),
    (re.compile(r'\b(?:eclampsia|pre.?eclampsia)\b', re.I),
     re.compile(r'\b(?:eclampsia|magnesium|labetalol|hydralazine|pre.eclampsia|hypertensive)\b', re.I)),
    (re.compile(r'\b(?:dysentery|bloody\s+diarrh?oea|shigella)\b', re.I),
     re.compile(r'\b(?:dysentery|shigella|ciprofloxacin|azithromycin|bloody\s+stool|acute\s+diarr?hoea)\b', re.I)),
    (re.compile(r'\b(?:typhoid|enteric\s+fever|salmonella\s+typhi)\b', re.I),
     re.compile(r'\b(?:typhoid|enteric\s+fever|salmonella\s+typhi|blood\s+culture|widal|ciprofloxacin|ceftriaxone)\b', re.I)),
    (re.compile(r'\b(?:sickle\s*cell|acute\s+chest\s+syndrome|ACS)\b', re.I),
     re.compile(r'\b(?:sickle\s*cell|acute\s+chest\s+syndrome|vaso.?occlusive|hydroxyurea|transfusion|HbS)\b', re.I)),
    (re.compile(r'\b(?:cardiac\s+arrest|cardiopulmonary\s+resuscitation|AED)\b', re.I),
     re.compile(r'\b(?:cardiac\s+arrest|CPR|defibrillat|compressions?|AED|resuscitat|adrenaline|epinephrine|atropine|airway)\b', re.I)),
    (re.compile(r'\b(?:snake.?bite|snake\s+(?:venom|envenomation)|snakebite)\b', re.I),
     re.compile(r'\b(?:antivenom|snake\s+(?:bite|venom)|polyvalent|antitoxin)\b', re.I)),
    (re.compile(r'\b(?:scorpion(?:\s+(?:sting|envenomation))?)\b', re.I),
     re.compile(r'\b(?:scorpion|sting|envenomation|antivenom|analgesic)\b', re.I)),
    (re.compile(r'\b(?:neonatal|newborn)\b.{0,60}\b(?:sepsis|infection|bacteremia|antibiotic)\b', re.I),
     re.compile(r'\b(?:sepsis|septic|ampicillin|gentamicin|bacteremia|bacteraemia|neonatal\s+(?:sepsis|infection))\b', re.I)),
    (re.compile(r'\b(?:severe\s+acute\s+malnutrition|SAM\b|RUTF|marasmus|kwashiorkor)\b', re.I),
     re.compile(r'\b(?:malnutrition|RUTF|kwashiorkor|marasmus|F-75|F-100|therapeutic\s+feeding|wasting|MUAC|undernutrition)\b', re.I)),
]

# ── Condition title / content exclusions ──────────────────────────────────
# Demote hits whose title explicitly names a different disease.
# v2 titles are heading paths — use re.search (substring), not anchors.
_CONDITION_TITLE_EXCLUSIONS: List[Tuple[re.Pattern, re.Pattern]] = [
    (re.compile(r'\b(?:meningitis|meningococcal)\b', re.I),
     re.compile(r'\bpneumonia\b', re.I)),
    (re.compile(r'\b(?:pneumonia|community.acquired\s+pneumonia)\b', re.I),
     re.compile(r'\bmeningitis\b', re.I)),
    (re.compile(r'\b(?:buruli|podoconiosis|tungiasis|noma|cancrum oris|necroti[sz]ing gingivitis)\b', re.I),
     re.compile(r'\b(?:scorpion|snake|envenomation|echinococcosis|PKDL)\b', re.I)),
    (re.compile(r'\b(?:dysentery|shigella|bloody\s+diarrh?oe?a)\b', re.I),
     re.compile(r'\b(?:sexual assault|sti|sexually transmitted|post.?exposure prophylaxis)\b', re.I)),
    (re.compile(r'\b(?:rabies|post.?exposure prophylaxis|PEP)\b', re.I),
     re.compile(r'\b(?:TB|tuberculosis|rifapentine|isoniazid)\b', re.I)),
    (re.compile(r'\b(?:dengue)\b', re.I),
     re.compile(r'\b(?:if intravenous treatment|plan c|severe dehydration|if there is high fever|when to stop vasopressors)\b', re.I)),
    (re.compile(r'\b(?:burn(?:s|ed)?|thermal injury|TBSA|parkland)\b', re.I),
     re.compile(r'\b(?:severe dehydration|if there is high fever|children under 2 months)\b', re.I)),
    (re.compile(r'\b(?:snake.?bite|antivenom)\b', re.I),
     re.compile(r'\bscorpion\b', re.I)),
]
_TITLE_EXCLUSION_PENALTY = 0.60
_TITLE_EXCLUSION_OVERRIDES: List[Tuple[re.Pattern, re.Pattern, float]] = [
    (re.compile(r'\b(?:meningitis|meningococcal)\b', re.I),
     re.compile(r'\bpneumonia\b', re.I), 0.35),
    (re.compile(r'\b(?:pneumonia|community.acquired\s+pneumonia)\b', re.I),
     re.compile(r'\bmeningitis\b', re.I), 0.35),
]
_CONDITION_CONTENT_EXCLUSIONS: List[Tuple[re.Pattern, re.Pattern, float]] = [
    (re.compile(r'\b(?:dysentery|shigella|bloody\s+diarrh?oe?a)\b', re.I),
     re.compile(r'\b(?:sexual assault|sexually transmitted|post.?exposure prophylaxis'
                r'|sex transm infect|hiv pep|pelvic inflammatory disease|lower abdominal pain'
                r'|gonorrhoeae|trachomatis|vaginosis)\b', re.I), 0.12),
    (re.compile(r'\b(?:typhoid|enteric\s+fever|salmonella\s+typhi)\b', re.I),
     re.compile(r'\b(?:ophthalmia|conjunctivitis|eye drops?|neonat\w+|sex transm infect)\b', re.I), 0.10),
]

# ── Population filter ─────────────────────────────────────────────────────
# PDF filenames as stored in meta["pdf_file"] of v2 hits.
# _PAEDIATRIC_PDFS: populated after reviewing v2 corpus for paediatric-only PDFs.
# Currently empty — the corpus has no confirmed paediatric-only PDF yet.
_PAEDIATRIC_PDFS: set = set()
_OBG_PDFS: set = {"MSF_OBG.pdf"}   # confirmed present in v2
_PAEDIATRIC_QUERY_RE = re.compile(
    r'\b(?:child(?:ren)?|infant|neonate|neonatal|paediatric|pediatric'
    r'|newborn|under.?5|under.?one)\b',
    re.IGNORECASE,
)
_OBG_QUERY_RE = re.compile(
    r'\b(?:pregnan|obstetric|maternal|postpartum|eclampsia|labour|labor'
    r'|antenatal|postnatal|PPH|placenta|foetal|fetal|neonatal)\b',
    re.IGNORECASE,
)
_ADULT_NONOBG_RE = re.compile(
    r'\b(?:DKA|diabetic\s+ketoacidosis|stroke\b|pulmonary\s+embolism\b'
    r'|cardiac\s+arrest|myocardial\s+infarction|septic\s+shock)\b',
    re.IGNORECASE,
)

# ── Intent-rerank: condition vocabulary ──────────────────────────────────
# Each entry: (condition_id, query_detect_re, chunk_match_re, chunk_exclude_re|None)
def _cre(q: str, c: str, x: Optional[str] = None, f: int = re.IGNORECASE
         ) -> Tuple[re.Pattern, re.Pattern, Optional[re.Pattern]]:
    return (re.compile(q, f), re.compile(c, f), re.compile(x, f) if x else None)

_CONDITION_VOCAB: List[Tuple[str, re.Pattern, re.Pattern, Optional[re.Pattern]]] = [
    ("meningitis",   *_cre(
        r'\bmeningitis\b|\bmeningococcal\b',
        r'\bmeningitis\b|\bmeningococcal\b',
        r'\brickettsia\b|\bmalaria\b(?!.*meningitis)')),
    ("malaria",      *_cre(
        r'\bmalaria\b|\bfalciparum\b|\bvivax\b|\bartemether\b|\bartesunate\b',
        r'\bmalaria\b|\bfalciparum\b|\bvivax\b|\bartemether\b|\bartesunate\b',
        r'\bmeningitis\b|\btuberculosis\b|\bpneumonia\b(?!.*malaria)')),
    ("tuberculosis", *_cre(
        r'\btuberculosis\b|\b(?:ds-?|dr-?|mdr-?|xdr-?)?tb\b',
        r'\btuberculosis\b|\btb\b',
        r'\bmalaria\b|\bmeningitis\b(?!.*tb)')),
    ("asthma",       *_cre(
        r'\basthma\b|\bstatus asthmaticus\b',
        r'\basthma\b|\bstatus asthmaticus\b|\bbronchospasm\b',
        r'\bpneumonia\b|\bmalaria\b(?!.*asthma)')),
    ("sepsis",       *_cre(
        r'\bsepsis\b|\bseptic shock\b|\bbacteremia\b',
        r'\bsepsis\b|\bseptic\b|\bbacteremia\b',
        r'\bmalaria\b(?!.*sepsis)|\bmeningitis\b(?!.*sepsis)')),
    ("pneumonia",    *_cre(
        r'\bpneumonia\b',
        r'\bpneumonia\b',
        r'\btuberculosis\b|\bmalaria\b(?!.*pneumonia)|\basthma\b(?!.*pneumonia)')),
    ("eclampsia",    *_cre(
        r'\beclampsia\b|\bpre.?eclampsia\b',
        r'\beclampsia\b|\bpre.?eclampsia\b|\bmagnesium sulfate\b')),
    ("malnutrition", *_cre(
        r'\bmalnutrition\b|\bsam\b|\bRUTF\b|\bF-?75\b|\bF-?100\b',
        r'\bmalnutrition\b|\bRUTF\b|\bF-?75\b|\bF-?100\b|\bSAM\b',
        r'\bmalaria\b|\btuberculosis\b(?!.*nutrit)')),
    ("diarrhea",     *_cre(
        r'\bdiarr[ho]+ea?\b|\bORS\b|\boral rehydration\b',
        r'\bdiarr[ho]+ea?\b|\boral rehydration\b|\bORS\b',
        r'\bdysentery\b|\brickettsia\b')),
    ("dysentery",    *_cre(
        r'\bdysentery\b|\bshigella\b',
        r'\bdysentery\b|\bshigella\b|\bbloody.{0,20}stool\b',
        r'\brickettsia\b|\bdengue\b|\bophthalmia\b')),
    ("cholera",      *_cre(
        r'\bcholera\b|\bvibrio\b',
        r'\bcholera\b|\bvibrio\b',
        r'\brickettsia\b|\bmalaria\b')),
    ("dengue",       *_cre(
        r'\bdengue\b',
        r'\bdengue\b|\bNS1\b|\bhaematocrit\b|\bhematocrit\b|\bplatelet\b',
        r'\bcholera\b|\bsevere dehydration\b|\bORS\b(?!.*dengue)')),
    ("burns",        *_cre(
        r'\bburn(?:s|ed)?\b|\bthermal injury\b|\bTBSA\b|\bparkland\b',
        r'\bburn(?:s|ed)?\b|\bthermal injury\b|\bTBSA\b|\bparkland\b|\brule of 9s\b',
        r'\bdengue\b|\bdiarrh?oe?a\b|\bcholera\b|\bsevere dehydration\b')),
    ("buruli_ulcer", *_cre(
        r'\bburuli\b|\bmycobacterium ulcerans\b',
        r'\bburuli\b|\bmycobacterium ulcerans\b|\bulcerans\b',
        r'\bscorpion\b|\bsnake\b|\benvenomation\b|\bPKDL\b')),
    ("podoconiosis", *_cre(
        r'\bpodoconiosis\b',
        r'\bpodoconiosis\b|\bnon[- ]filarial elephantiasis\b',
        r'\blymphatic filariasis\b|\benvenomation\b|\bscorpion\b|\bsnake\b')),
    ("tungiasis",    *_cre(
        r'\btungiasis\b|\btunga penetrans\b',
        r'\btungiasis\b|\btunga penetrans\b|\bjigger\b',
        r'\bechinococcosis\b|\bPKDL\b|\benvenomation\b|\bscorpion\b|\bsnake\b')),
    ("noma",         *_cre(
        r'\bnoma\b|\bcancrum oris\b|\bnecroti[sz]ing gingivitis\b',
        r'\bnoma\b|\bcancrum oris\b|\bnecroti[sz]ing gingivitis\b|\boral gangrene\b',
        r'\bautism\b|\benvenomation\b|\bscorpion\b|\bsnake\b')),
    ("typhoid",      *_cre(
        r'\btyphoid\b|\benteric fever\b|\bsalmonella typhi\b',
        r'\btyphoid\b|\benteric fever\b|\bsalmonella typhi\b',
        r'\bophthalmia\b|\bconjunctivitis\b|\bneonat\w+\b')),
    ("sickle_cell",  *_cre(
        r'\bsickle\s*cell\b|\bacute chest syndrome\b|\bACS\b',
        r'\bsickle\s*cell\b|\bacute chest syndrome\b|\bvaso.?occlusive\b|\bhydroxyurea\b|\btransfusion\b',
        r'\bpneumonia\b(?!.*sickle)|\bophthalmia\b')),
    ("scorpion",     *_cre(
        r'\bscorpion(?:\s+(?:sting|envenomation))?\b',
        r'\bscorpion\b|\bsting\b|\benvenomation\b|\bantivenom\b',
        r'\bsnake\b')),
    ("hiv",          *_cre(
        r'\bHIV\b|\bART\b|\bantiretroviral\b|\bPMTCT\b',
        r'\bHIV\b|\bantiretroviral\b|\bARV\b')),
    ("rabies",       *_cre(
        r'\brabies\b|\bpost.?exposure prophylaxis\b|\bPEP\b',
        r'\brabies\b|\bimmunoglobulin\b|\bvaccine\b|\bpost.?exposure\b',
        r'\bTB\b|\btuberculosis\b|\brifapentine\b|\bisoniazid\b'
        r'|\bmalaria\b|\bRTS,S\b|\bcasirivimab\b|\bimdevimab\b')),
    ("snakebite",    *_cre(
        r'\bsnake.?bite\b|\benvenomation\b|\bantivenom\b',
        r'\bsnake.?bite\b|\bsnake\b.{0,40}\b(?:antivenom|venom|bite|envenomation)\b'
        r'|\bpolyvalent\s+antivenom\b',
        r'\bscorpion\b')),
    ("tbi",          *_cre(
        r'\btraumatic brain\b|\btbi\b|\bhead injur\b',
        r'\brain injur\b|\bhead injur\b|\btbi\b',
        r'\bstroke\b(?!.*brain)')),
    ("hypertension", *_cre(
        r'\bhypertension\b|\bhypertensiv(?:e|ion)\b|\bhypertensive emergency\b',
        r'\bhypertensiv\b|\bblood pressure\b|\blabetalol\b|\bhydralazine\b|\bamlodipine\b',
        r'\bmalaria\b|\bmeningitis\b')),
    ("anemia",       *_cre(
        r'\bana?emia\b|\bha?emoglobin\b|\btransfusion\b|\bpacked cells?\b',
        r'\bana?emia\b|\bha?emoglobin\b|\btransfusion\b|\bpacked cells?\b|\bwhole blood\b|\bblood transfusion\b')),
    ("schistosomiasis", *_cre(
        r'\bschistosomiasis\b|\bbilharzia\b|\bpraziquantel\b',
        r'\bschistosomiasis\b|\bbilharzia\b|\bpraziquantel\b|\bschistosome\b',
        r'\bmalaria\b|\bfilariasis\b')),
    ("appendicitis", *_cre(
        r'\bappendicitis\b|\bappendectomy\b|\bappendicectomy\b',
        r'\bappendicitis\b|\bappendectomy\b|\bappendicectomy\b|\bperiappendic',
        r'\bcaesarean\b|\bcesarean\b|\boperative vaginal\b|\bvaginal birth\b|\blabou?r\b')),
    ("heart_failure", *_cre(
        r'\bheart failure\b|\bcardiac failure\b|\bfurosemide\b',
        r'\bheart failure\b|\bcardiac failure\b|\bfurosemide\b|\bdiuretic\b|\bpulmonary oedema\b|\bpulmonary edema\b')),
    ("depression",   *_cre(
        r'\bdepression\b|\bdepressive\b|\bssri\b|\bfluoxetine\b',
        r'\bdepression\b|\bdepressive\b|\bssri\b|\bfluoxetine\b|\bantidepressant\b|\bpsychological\b')),
    ("psychosis",    *_cre(
        r'\bpsychosis\b|\bhaloperidol\b|\bpsychotic\b',
        r'\bpsychosis\b|\bhaloperidol\b|\bpsychotic\b|\bschizophrenia\b')),
    ("pph",          *_cre(
        r'\bpostpartum hemorrhage\b|\bPPH\b|\boxytocin\b|\buterine atony\b',
        r'\bPPH\b|\buterine\b|\boxytocin\b|\bpostpartum.*bleed\b')),
    ("ectopic",      *_cre(
        r'\bectopic pregnancy\b|\bectopic\b',
        r'\bectopic\b|\bfallopi\b')),
]

# ── Intent-rerank: severity slot ─────────────────────────────────────────
_SEV_LIGHT_Q = re.compile(
    r'\buncomplicated\b|\bsimple\b|\bmild\b'
    r'|\bds[- ]?tb\b|\bdrug[- ]?sensitive\b|\bfirst.?line oral\b|\boutpatient\b',
    re.IGNORECASE,
)
_SEV_HEAVY_Q = re.compile(
    r'\bsevere\b|\bcritical\b|\bshock\b|\bICU\b|\bintensive care\b|\bemergency\b',
    re.IGNORECASE,
)
_SEV_HEAVY_CHUNK = re.compile(
    r'(?<!un)\bcomplicated\b|\bsevere\b|\bcritical\b|\bshock\b', re.IGNORECASE,
)
_SEV_LIGHT_CHUNK = re.compile(
    r'\buncomplicated\b|\boral.{0,20}(?:treatment|therapy)\b|\boutpatient\b', re.IGNORECASE,
)
_DS_TB_Q     = re.compile(r'\bds[- ]?tb\b|\bdrug[- ]?sensitive\b', re.IGNORECASE)
_DR_TB_CHUNK = re.compile(
    r'\bdrug[- ]?resistant\b|\bmdr[- ]?tb\b|\bxdr[- ]?tb\b'
    r'|\bbedaquiline\b|\bpretomanid\b|\blinezolid\b',
    re.IGNORECASE,
)

# ── Intent-rerank: population slot ───────────────────────────────────────
_POP_ADULT_Q   = re.compile(r'\badult\b|\bwoman\b|\bwomen\b|\bpregnant\b|\bmaternal\b', re.IGNORECASE)
_POP_CHILD_Q   = re.compile(r'\bchild(?:ren)?\b|\bpediatric\b|\bpaediatric\b|\bunder.?5\b', re.IGNORECASE)
_POP_NEONATE_Q = re.compile(r'\bneonatal\b|\bnewborn\b|\bneonate\b', re.IGNORECASE)
_POP_NEONATE_CHUNK = re.compile(
    r'\bchildren under \d\b|\bneonatal\b|\bunder 2 months\b|\bnewborn\b|\bneonate\b',
    re.IGNORECASE,
)
_POP_CHILD_CHUNK = re.compile(
    r'\bchild(?:ren)?\b|\bpediatric\b|\bpaediatric\b|\bunder.?5\b|\binfant\b', re.IGNORECASE,
)
_POP_ADULT_CHUNK = re.compile(
    r'\badult\b|\bpatients?.{0,10}(?:aged?|over) \d{2}', re.IGNORECASE,
)

# ── Intent-rerank: task slot ──────────────────────────────────────────────
_TASK_DOSE_Q      = re.compile(r'\bdose\b|\bdosage\b|\bmg(?:/kg)?\b|\bregimen\b|\badminister\b|\binfusion\b|\bhow much\b', re.IGNORECASE)
_TASK_DIAGNOSE_Q  = re.compile(r'\bdiagnos(?:e|is|tic)\b|\bcriteria\b|\bsigns?\b|\bdifferential\b|\bidentif\b', re.IGNORECASE)
_TASK_FIRSTLINE_Q = re.compile(r'\bfirst.?line\b|\bempiric\b|\binitial treatment\b', re.IGNORECASE)
_TASK_REFERRAL_Q  = re.compile(r'\breferral\b|\bwhen to refer\b|\bbefore referral\b|\btransfer\b', re.IGNORECASE)
_TASK_PREVENT_Q   = re.compile(r'\bprevention\b|\bprophylax\b|\bprevent\b|\bvaccin\b', re.IGNORECASE)

_ACS_QUERY_RE = re.compile(r'\b(?:acute\s+chest\s+syndrome|ACS)\b', re.IGNORECASE)
_ACS_CHUNK_RE = re.compile(
    r'\b(?:acute\s+chest\s+syndrome|ACS|hypox(?:ia|emic)|chest pain|respiratory distress)\b',
    re.IGNORECASE,
)
_TREATMENT_COMPONENT_Q = re.compile(
    r'\b(?:vasopressor|norepinephrine|noradrenaline|epinephrine|adrenaline'
    r'|dopamine|dobutamine|ceftriaxone|ciprofloxacin|amoxicillin|dose|dosage'
    r'|mg(?:/kg)?|infusion|iv\b|im\b|oral|oxytocin|misoprostol|artemether'
    r'|lumefantrine|vaccine|transfusion|tamponade|praziquantel|labetalol)\b',
    re.IGNORECASE,
)

# ── Low-confidence threshold ──────────────────────────────────────────────
_LOW_CONFIDENCE_SCORE = 0.012
_LOW_PRIORITY_TOP_THRESHOLD = 0.5   # flag if top hit rp < this value


# ─────────────────────────────────────────────────────────────────────────────
# Pure helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _is_action_query(query: str) -> bool:
    """True when query has a condition signal and either dose or operational task intent."""
    return bool(_ACTION_CONDITION_RE.search(query) and
                (_ACTION_DOSE_RE.search(query) or _ACTION_TASK_RE.search(query)))


def _expand_query(query: str) -> str:
    """Append synonym aliases as a dedup suffix; never mutate the original query."""
    aliases: set = set()
    for term, expansions in _SYNONYM_MAP.items():
        if _SYNONYM_DETECT[term].search(query):
            aliases.update(expansions)
    query_lower = query.lower()
    aliases = {a for a in aliases if a.lower() not in query_lower}
    return (query + " " + " ".join(sorted(aliases))).strip() if aliases else query


def _actionability_score(content: str) -> float:
    """Returns 1.00–1.21× based on presence of dose/route/frequency signals."""
    sample = content[:2000]
    n = (bool(_DOSE_NUMBER_RE.search(sample)) +
         bool(_ROUTE_ADMIN_RE.search(sample)) +
         bool(_FREQUENCY_RE.search(sample)))
    return 1.0 + 0.07 * n


def _has_background_rescue_signals(hit: Dict[str, Any]) -> bool:
    """True for background chunks that look like directly usable protocol text."""
    if hit.get("content_type") != "background":
        return False
    text = ((hit.get("title") or "") + "\n" + (hit.get("content") or ""))[:2000]
    if not _PROTOCOL_TEXT_RE.search(text):
        return False
    protocol_density = sum(
        bool(rx.search(text))
        for rx in (_DOSE_NUMBER_RE, _ROUTE_ADMIN_RE, _FREQUENCY_RE)
    )
    return bool(_RESCUE_RECOMMEND_RE.search(text) or protocol_density >= 2)


def _is_actionable_hit(hit: Dict[str, Any]) -> bool:
    """True if hit is in action-priority types or contains protocol-like language."""
    if hit.get("rescued_actionable"):
        return True
    if hit.get("content_type") in _ACTION_PRIORITY_TYPES:
        return True
    text = ((hit.get("title") or "") + "\n" + (hit.get("content") or ""))[:2000]
    return bool(_PROTOCOL_TEXT_RE.search(text))


def _is_promotable_action_hit(hit: Dict[str, Any]) -> bool:
    """Stricter notion of actionable for top-hit promotion."""
    return bool(
        hit.get("content_type") in _ACTION_PRIORITY_TYPES
        or hit.get("rescued_actionable")
    )


def _extract_hits(raw: Any) -> List[Dict[str, Any]]:
    """Normalize mem.find() return shape."""
    if isinstance(raw, dict):
        hits = raw.get("hits", [])
        return hits if isinstance(hits, list) else []
    if isinstance(raw, list):
        return raw
    return []


def _parse_headings(raw: Any) -> List[str]:
    """Parse headings field from metadata — stored as a JSON string in memvid."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(h) for h in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        return [raw] if raw else []
    return []


def _rrf_merge(
    lex_hits: List[Dict[str, Any]],
    sem_hits: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion.

    Score = sum of 1/(k + rank + 1) across both lists.
    Merge key: frame_id with #page-N suffix stripped.
    k=60 is the standard bias term that dampens rank-1 advantage.
    """
    def _pid(fid: str) -> str:
        return fid.split("#")[0] if fid else fid

    scores: Dict[str, Dict[str, Any]] = {}
    for rank, hit in enumerate(lex_hits):
        fid = _pid(hit.get("frame_id") or f"l{rank}")
        if fid not in scores:
            scores[fid] = {"hit": hit, "rrf": 0.0}
        scores[fid]["rrf"] += 1.0 / (k + rank + 1)
    for rank, hit in enumerate(sem_hits):
        fid = _pid(hit.get("frame_id") or f"s{rank}")
        if fid not in scores:
            scores[fid] = {"hit": hit, "rrf": 0.0}
        scores[fid]["rrf"] += 1.0 / (k + rank + 1)
    merged = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)
    return [dict(m["hit"], score=m["rrf"]) for m in merged]


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_hit(
    hit: Dict[str, Any],
    source_name: str,
    snippet_chars: int,
) -> Optional[Dict[str, Any]]:
    """Convert a raw memvid hit into the stable v2 response shape.

    Exposes all v2 schema fields from hit.metadata so downstream pipeline
    stages and callers have full access without re-reading the index.
    """
    score = hit.get("score", 0)
    raw_content = hit.get("snippet", hit.get("frame", ""))
    if not raw_content:
        return None

    # Strip metadata lines that bleed into snippet text
    lines = str(raw_content).split("\n")
    clean_lines = [ln for ln in lines if not any(
        ln.strip().startswith(p) for p in _METADATA_LINE_PREFIXES
    )]
    content = "\n".join(clean_lines).strip()
    if not content:
        return None

    frame_id = hit.get("frame_id")

    # Strip #page-N pagination suffix from URI
    uri = hit.get("uri", "")
    if "#page-" in uri:
        uri = uri.split("#")[0]

    meta = hit.get("metadata", {}) or {}

    # ── v2 metadata fields ────────────────────────────────────────────────
    content_type = meta.get("content_type", "background")

    rp_raw = meta.get("retrieval_priority")
    retrieval_priority: float = 1.0
    if rp_raw is not None:
        try:
            retrieval_priority = float(rp_raw)
        except (ValueError, TypeError):
            pass

    ic_raw = meta.get("is_current")
    is_current: bool = True
    if ic_raw is not None:
        is_current = str(ic_raw).lower() not in ("false", "0", "no")

    headings = _parse_headings(meta.get("headings"))

    # recommendation_strength / evidence_certainty: stored as "null" string when absent
    def _nullable_str(val: Any) -> Optional[str]:
        if val is None or str(val).lower() in ("null", "none", ""):
            return None
        return str(val)

    return {
        "score":                   float(score),
        "title":                   hit.get("title", ""),
        "content":                 content[:snippet_chars],
        "source":                  source_name,
        "uri":                     uri,
        "frame_id":                str(frame_id) if frame_id is not None else "",
        # v2 schema fields
        "content_type":            content_type,
        "retrieval_priority":      retrieval_priority,
        "is_current":              is_current,
        "headings":                headings,
        "doc_type":                meta.get("doc_type", ""),
        "recommendation_strength": _nullable_str(meta.get("recommendation_strength")),
        "evidence_certainty":      _nullable_str(meta.get("evidence_certainty")),
        "source_url":              meta.get("source_url", ""),
        "pdf_file":                meta.get("pdf_file", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Re-ranking stages
# ─────────────────────────────────────────────────────────────────────────────

def _apply_cds_boost(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compound score multiplier: content_boost × retrieval_priority.

    This is the primary CDS signal:
      - content_boost  encodes clinical utility of the chunk type (0.15–1.50×)
      - retrieval_priority encodes corpus-level importance (0.1–1.0, set at index build)
    A background chunk at rp=0.3 gets: 0.20 × 0.3 = 0.06× of its raw score.
    A recommendation chunk at rp=1.0 gets: 1.50 × 1.0 = 1.50×.
    """
    for h in hits:
        ct = h.get("content_type", "background")
        boost = _V2_CONTENT_BOOST.get(ct, _V2_CONTENT_BOOST_DEFAULT)
        rp    = h.get("retrieval_priority", 1.0)
        h["score"] = h["score"] * boost * rp
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _apply_action_pipeline(
    hits: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Three action-query boosts applied in sequence:
    1. Context demote: 0.75× for non-actionable types
    2. Actionability text bonus: 1.00–1.21× from dose/route/frequency signals
    3. Action-type bonus: 1.15× for recommendation/implementation
    """
    # 1. Context demote
    for h in hits:
        if h.get("content_type") in _ACTION_CONTEXT_DEMOTE:
            h["score"] *= _ACTION_CONTEXT_DEMOTE_FACTOR

    # 2. Actionability text bonus
    for h in hits:
        h["score"] *= _actionability_score(h.get("content", ""))

    # 3. Action-type bonus
    for h in hits:
        if h.get("content_type") in _ACTION_PRIORITY_TYPES:
            h["score"] *= _ACTION_BONUS

    # Small extra lift for protocol-like language in any chunk
    for h in hits:
        if _is_actionable_hit(h):
            h["score"] *= _ACTIONABLE_TEXT_BONUS

    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _apply_domain_coherence(
    hits: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """Penalise hits with zero overlap with any active query condition."""
    active_checks = [content_re for query_re, content_re in _CONDITION_COHERENCE
                     if query_re.search(query)]
    if not active_checks:
        return hits
    for h in hits:
        combined = h.get("title", "") + " " + h.get("content", "")[:1000]
        if not any(cr.search(combined) for cr in active_checks):
            if h.get("content_type") in _ACTION_PRIORITY_TYPES:
                h["score"] *= _DOMAIN_COHERENCE_PENALTY_ACTION_TYPE
            else:
                h["score"] *= _DOMAIN_COHERENCE_PENALTY
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _apply_condition_exclusions(
    hits: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """Demote hits whose title path or content names a different disease."""
    # Title exclusions — use re.search because v2 titles are heading paths
    for query_re, title_re in _CONDITION_TITLE_EXCLUSIONS:
        if query_re.search(query):
            for h in hits:
                if title_re.search(h.get("title", "")):
                    h["score"] *= _TITLE_EXCLUSION_PENALTY
    for query_re, title_re, factor in _TITLE_EXCLUSION_OVERRIDES:
        if query_re.search(query):
            for h in hits:
                if title_re.search(h.get("title", "")):
                    h["score"] *= factor

    # Content exclusions
    for query_re, content_re, factor in _CONDITION_CONTENT_EXCLUSIONS:
        if query_re.search(query):
            for h in hits:
                combined = (h.get("title", "") + " " + h.get("content", "")[:1200])
                if content_re.search(combined):
                    h["score"] *= factor

    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _apply_background_rescue(
    hits: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """Rescue narrowly-scoped background hits that contain protocol-grade guidance."""
    if not hits:
        return hits

    condition: Optional[str] = None
    cond_c_re: Optional[re.Pattern] = None
    cond_x_re: Optional[re.Pattern] = None
    for cid, q_re, c_re, x_re in _CONDITION_VOCAB:
        if q_re.search(query):
            condition, cond_c_re, cond_x_re = cid, c_re, x_re
            break

    active_checks = [content_re for q_re, content_re in _CONDITION_COHERENCE
                     if q_re.search(query)]

    for h in hits:
        h["rescued_actionable"] = False
        if not _has_background_rescue_signals(h):
            continue

        combined = (h.get("title", "") + " " + h.get("content", "")[:2000]).lower()
        on_target = False
        if condition and cond_c_re:
            on_target = bool(cond_c_re.search(combined))
            off_target = bool(cond_x_re.search(combined)) if cond_x_re else False
            if off_target and not on_target:
                continue
        elif active_checks:
            on_target = any(cr.search(combined) for cr in active_checks)

        if not on_target:
            continue

        # Keep this conservative: only strongly protocol-like background chunks
        # get a score rescue, to compensate for imperfect chunk typing.
        h["score"] *= 3.0
        h["rescued_actionable"] = True

    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _apply_population_filter(
    hits: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """Demote paediatric-specific PDFs for non-paediatric queries.
    Demote OBG-specific PDFs for explicit adult non-OBG queries.
    Uses meta["pdf_file"] directly — no URI parsing needed for v2.
    """
    if _PAEDIATRIC_QUERY_RE.search(query):
        return hits   # paediatric query: no penalty for either filter

    for h in hits:
        if h.get("pdf_file") in _PAEDIATRIC_PDFS:
            h["score"] *= 0.55

    if _ADULT_NONOBG_RE.search(query) and not _OBG_QUERY_RE.search(query):
        for h in hits:
            if h.get("pdf_file") in _OBG_PDFS:
                h["score"] *= 0.60

    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _apply_source_diversity(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Penalise 3rd+ hits from the same PDF to prevent hub-chunk dominance.
    Uses meta["pdf_file"] directly from the normalised hit.
    """
    seen: Dict[str, int] = {}
    result = []
    for h in hits:
        pdf = h.get("pdf_file", "") or ""
        count = seen.get(pdf, 0)
        if count < _SOURCE_DIVERSITY_MAX:
            result.append(h)
        else:
            result.append(dict(h, score=h["score"] * _SOURCE_DIVERSITY_PENALTY))
        seen[pdf] = count + 1
    return sorted(result, key=lambda x: x["score"], reverse=True)


def _apply_soft_corruption_demote(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Soft-demote noisy chunks that evade the hard corruption hard-filter."""
    for h in hits:
        sample = (h.get("title", "") + " " + h.get("content", "")[:1200])
        if _CORRUPTION_SOFT_RE.search(sample):
            h["score"] *= 0.45
    return sorted(hits, key=lambda x: x["score"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Intent-constrained reranking (4-slot)
# ─────────────────────────────────────────────────────────────────────────────

def _intent_rerank(
    hits: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """4-slot intent reranker.

    Extracts condition / severity / population / task from the query, then
    scores each hit for slot alignment.  Final score:
        raw_score × max(penalty, 0.10) × (1 + boost)

    Penalties (< 1.0) stack multiplicatively for mismatching slots.
    Boosts (additive to 1.0) accumulate for confirmed matching slots.
    Floor of 0.10 prevents any hit from disappearing entirely.
    """
    if not hits:
        return hits

    # ── Slot extraction ───────────────────────────────────────────────────
    condition: Optional[str] = None
    cond_c_re: Optional[re.Pattern] = None
    cond_x_re: Optional[re.Pattern] = None
    for cid, q_re, c_re, x_re in _CONDITION_VOCAB:
        if q_re.search(query):
            condition, cond_c_re, cond_x_re = cid, c_re, x_re
            break

    sev_light   = bool(_SEV_LIGHT_Q.search(query))
    sev_heavy   = bool(_SEV_HEAVY_Q.search(query)) and not sev_light
    ds_tb       = bool(_DS_TB_Q.search(query))
    pop_adult   = bool(_POP_ADULT_Q.search(query))
    pop_neonate = bool(_POP_NEONATE_Q.search(query))
    pop_child   = bool(_POP_CHILD_Q.search(query)) and not pop_adult and not pop_neonate

    task: Optional[str] = None
    for task_id, task_re in (
        ("dose",       _TASK_DOSE_Q),
        ("diagnosis",  _TASK_DIAGNOSE_Q),
        ("first_line", _TASK_FIRSTLINE_Q),
        ("referral",   _TASK_REFERRAL_Q),
        ("prevention", _TASK_PREVENT_Q),
    ):
        if task_re.search(query):
            task = task_id
            break

    active_coherence = [content_re for q_re, content_re in _CONDITION_COHERENCE
                        if q_re.search(query)]

    # ── Per-hit scoring ───────────────────────────────────────────────────
    def _adjusted(hit: Dict[str, Any]) -> float:
        title    = (hit.get("title") or "").strip()
        content  = hit.get("content") or ""
        ct       = hit.get("content_type") or ""
        combined = title.lower() + " " + content.lower()
        chunk_on_target = False

        penalty       = 1.0
        boost         = 0.0
        matched_slots = 0

        # v2: annex content_type → hard demote (no title regex needed)
        if ct == "annex":
            penalty *= 0.25

        # ── Condition slot ────────────────────────────────────────────────
        if condition and cond_c_re:
            chunk_on_target  = bool(cond_c_re.search(combined))
            chunk_off_target = bool(cond_x_re.search(combined)) if cond_x_re else False
            if chunk_off_target and not chunk_on_target:
                penalty *= 0.12
            elif chunk_off_target and chunk_on_target:
                penalty *= 0.55
            elif chunk_on_target:
                boost += 0.25
                matched_slots += 1

        # Condition-specific hard overrides
        if condition == "snakebite" and re.search(r'\bscorpion\b', combined, re.I):
            penalty *= 0.05
            if boost >= 0.25:
                boost -= 0.25
            if matched_slots > 0:
                matched_slots -= 1
        if condition == "rabies" and re.search(
                r'\bmalaria\b|\bRTS,S\b|\bcasirivimab\b|\bimdevimab\b', combined, re.I):
            penalty *= 0.20
        if (condition == "sickle_cell"
                and _ACS_QUERY_RE.search(query)
                and not _ACS_CHUNK_RE.search(combined)):
            penalty *= 0.30
            if boost >= 0.25:
                boost -= 0.25
            if matched_slots > 0:
                matched_slots -= 1

        # v2: demote non-actionable chunks with weak on-condition signal
        # (replaces _GENERIC_SECTION_TITLE_RE which was v1-specific)
        if condition and ct in {"background", "methods_pico"} and not chunk_on_target:
            penalty *= 0.45

        if (condition and chunk_on_target
                and active_coherence
                and not any(cr.search(combined) for cr in active_coherence)):
            penalty *= 0.55

        # ── Severity slot ─────────────────────────────────────────────────
        if sev_light:
            if _SEV_HEAVY_CHUNK.search(combined):
                penalty *= 0.35
            elif _SEV_LIGHT_CHUNK.search(combined):
                boost += 0.10
                matched_slots += 1
        elif sev_heavy:
            if _SEV_HEAVY_CHUNK.search(combined):
                boost += 0.10
                matched_slots += 1

        if ds_tb and _DR_TB_CHUNK.search(combined):
            penalty *= 0.25

        # ── Population slot ───────────────────────────────────────────────
        if pop_adult:
            if _POP_NEONATE_CHUNK.search(combined):
                penalty *= 0.30
            elif _POP_CHILD_CHUNK.search(combined) and not _POP_ADULT_CHUNK.search(combined):
                penalty *= 0.60
            elif _POP_ADULT_CHUNK.search(combined):
                boost += 0.10
                matched_slots += 1
        elif pop_neonate:
            if _POP_NEONATE_CHUNK.search(combined):
                boost += 0.20
                matched_slots += 1
        elif pop_child:
            if _POP_CHILD_CHUNK.search(combined) and not _POP_NEONATE_CHUNK.search(combined):
                boost += 0.10
                matched_slots += 1

        # ── Task slot ─────────────────────────────────────────────────────
        if task and ct in _TASK_PREFERRED_TYPES.get(task, set()):
            boost += 0.10
            matched_slots += 1

        # ── Multi-slot match bonus ────────────────────────────────────────
        if matched_slots >= 3:
            boost += 0.15
        elif matched_slots == 2:
            boost += 0.05

        penalty = max(penalty, 0.10)
        return hit["score"] * penalty * (1.0 + boost)

    adjusted: List[Tuple[Dict[str, Any], float]] = [(h, _adjusted(h)) for h in hits]

    # Sev-light guard: if all on-condition chunks are heavy-only, soften one
    if sev_light and condition and cond_c_re:
        on_cond: List[Tuple[Dict[str, Any], float]] = []
        has_light = False
        for h, sc in adjusted:
            combined = ((h.get("title") or "") + " " + (h.get("content") or "")).lower()
            if cond_c_re.search(combined):
                on_cond.append((h, sc))
                if _SEV_LIGHT_CHUNK.search(combined):
                    has_light = True
        if on_cond and not has_light:
            best_h, _ = max(on_cond, key=lambda item: item[1])
            relax = 0.50 / 0.35
            adjusted = [
                (h, (sc * relax) if h is best_h else sc)
                for h, sc in adjusted
            ]

    adjusted.sort(key=lambda item: item[1], reverse=True)
    for h, adj in adjusted:
        h["score"] = float(adj)
    return [h for h, _ in adjusted]


# ─────────────────────────────────────────────────────────────────────────────
# Alignment guardrails
# ─────────────────────────────────────────────────────────────────────────────

def _promote_aligned_top_hit(
    hits: List[Dict[str, Any]],
    query: str,
) -> List[Dict[str, Any]]:
    """Always-on guardrail: if top hit is off-condition and a strong aligned
    alternative exists (score ≥ 70% of top), reorder the list.
    No score mutation — positional swap only.
    """
    if len(hits) < 2:
        return hits

    condition: Optional[str] = None
    cond_c_re: Optional[re.Pattern] = None
    cond_x_re: Optional[re.Pattern] = None
    for cid, q_re, c_re, x_re in _CONDITION_VOCAB:
        if q_re.search(query):
            condition, cond_c_re, cond_x_re = cid, c_re, x_re
            break

    if condition and cond_c_re:
        def _aligned(h: Dict[str, Any]) -> bool:
            combined = ((h.get("title", "") + " " + h.get("content", "")[:1200]).lower())
            on  = bool(cond_c_re.search(combined))
            off = bool(cond_x_re.search(combined)) if cond_x_re else False
            return on and not off
    else:
        active_checks = [content_re for q_re, content_re in _CONDITION_COHERENCE
                         if q_re.search(query)]
        if not active_checks:
            return hits

        def _aligned(h: Dict[str, Any]) -> bool:
            combined = h.get("title", "") + " " + h.get("content", "")[:1200]
            return any(cr.search(combined) for cr in active_checks)

    action_query = _is_action_query(query)
    if _aligned(hits[0]) and (not action_query or _is_promotable_action_hit(hits[0])):
        return hits

    for idx, cand in enumerate(hits[1:], start=1):
        if not _aligned(cand):
            continue
        if action_query:
            if not _is_promotable_action_hit(cand):
                continue
        elif not _is_actionable_hit(cand):
            continue
        if cand.get("score", 0.0) < hits[0].get("score", 0.0) * 0.70:
            continue
        return [cand] + hits[:idx] + hits[idx + 1:]
    return hits


def _safe_top1_guardrail(
    hits: List[Dict[str, Any]],
    query: str,
) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    """Optional stricter guardrail (only called when safe_top1_guardrail=True).

    Conditions to trigger a swap:
      - Top hit fails condition_match_re AND coherence check (or ACS miss)
      - Exempt if query is treatment-component and top is actionable type
    Replacement must: pass condition + coherence, be actionable type if action
    query, and score ≥ 70% of top.

    Returns: (possibly reordered hits, swapped flag, reason string or None)
    """
    if len(hits) < 2:
        return hits, False, None

    condition_match_re: Optional[re.Pattern] = None
    for _cid, q_re, c_re, _x_re in _CONDITION_VOCAB:
        if q_re.search(query):
            condition_match_re = c_re
            break
    if condition_match_re is None:
        return hits, False, None

    active_coherence = [content_re for q_re, content_re in _CONDITION_COHERENCE
                        if q_re.search(query)]

    top = hits[0]
    top_combined = (top.get("title", "") + " " + top.get("content", "")[:1200])
    coherence_miss = (not active_coherence) or (
        not any(cr.search(top_combined) for cr in active_coherence))
    acs_miss = bool(_ACS_QUERY_RE.search(query) and not _ACS_CHUNK_RE.search(top_combined))
    top_actionable = top.get("content_type") in _ACTION_PRIORITY_TYPES
    component_query = bool(_TREATMENT_COMPONENT_Q.search(query))

    off_condition = (
        (((not condition_match_re.search(top_combined)) and coherence_miss) or acs_miss)
        and not (component_query and top_actionable)
    )
    if not off_condition:
        return hits, False, None

    action_query = _is_action_query(query)
    top_score    = float(top.get("score", 0.0))
    for idx, cand in enumerate(hits[1:], start=1):
        cand_combined = (cand.get("title", "") + " " + cand.get("content", "")[:1200])
        if not condition_match_re.search(cand_combined):
            continue
        if active_coherence and not any(cr.search(cand_combined) for cr in active_coherence):
            continue
        if action_query and cand.get("content_type") not in _ACTION_PRIORITY_TYPES:
            continue
        if float(cand.get("score", 0.0)) < top_score * 0.70:
            continue
        reordered = [cand] + hits[:idx] + hits[idx + 1:]
        return reordered, True, "off_condition_top_replaced_by_aligned_candidate"

    return hits, False, None


# ─────────────────────────────────────────────────────────────────────────────
# KBRetriever
# ─────────────────────────────────────────────────────────────────────────────

class KBRetriever:
    """Thread-safe v2 KB retriever with lazy-loaded memvid handle and embedder."""

    def __init__(self, who_index: str = V2_INDEX) -> None:
        self._index_path  = who_index
        self._mem         = None
        self._embedder: Optional[EmbedGemmaEmbedder] = None
        self._lock        = threading.Lock()

    def initialize(self, enable_vec: bool = True) -> None:
        """Open the index.  Call once at startup; thread-safe double-checked lock."""
        if self._mem is not None:
            return
        with self._lock:
            if self._mem is not None:
                return
            import memvid_sdk
            self._mem = memvid_sdk.use(
                "basic",
                self._index_path,
                read_only=True,
                enable_vec=enable_vec,
                enable_lex=True,
            )
            log.info("v2 index loaded: %s", self._index_path)

    def _get_embedder(self) -> Optional[EmbedGemmaEmbedder]:
        """Lazy-init embedder; returns None if model unavailable."""
        if self._embedder is None:
            emb = EmbedGemmaEmbedder()
            if emb.available:
                self._embedder = emb
        return self._embedder

    # ── Low-level fetch methods ───────────────────────────────────────────

    def _find_lex(self, query: str, k: int, snippet_chars: int) -> Any:
        return self._mem.find(query, k=k, snippet_chars=snippet_chars, mode="lex")

    def _find_sem(self, query: str, k: int, snippet_chars: int) -> Any:
        embedder = self._get_embedder()
        if embedder is None:
            raise RuntimeError("Embedding server unavailable for semantic search")
        return self._mem.find(
            query, k=k, snippet_chars=snippet_chars, mode="sem", embedder=embedder,
        )

    # ── Core search pipeline ──────────────────────────────────────────────

    def _search_pipeline(
        self,
        query: str,
        snippet_chars: int,
        search_mode: str,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Fetch → normalise → re-rank pipeline.  Always returns ≤15 candidates
        before the final top-5 trim in search().
        """
        errors: List[str] = []
        K_INTERNAL = 15   # fixed 3× fetch pool, always

        def _collect(raw: Any) -> List[Dict[str, Any]]:
            result = []
            for item in _extract_hits(raw):
                norm = _normalize_hit(item, SOURCE_WHO, snippet_chars)
                if norm is not None:
                    result.append(norm)
            return result

        # ── Stage 0: Fetch ────────────────────────────────────────────────
        try:
            if search_mode == "sem":
                hits = _collect(self._find_sem(query, K_INTERNAL, snippet_chars))
            elif search_mode == "rrf":
                lex_raw = self._find_lex(query, K_INTERNAL, snippet_chars)
                sem_raw = self._find_sem(query, K_INTERNAL, snippet_chars)
                hits = _rrf_merge(_collect(lex_raw), _collect(sem_raw))
            else:
                hits = _collect(self._find_lex(query, K_INTERNAL, snippet_chars))
        except Exception as exc:
            errors.append(f"fetch error ({search_mode}): {exc}")
            return [], errors

        if not hits:
            return [], errors

        # ── Stage 1: Hard-drop corrupted chunks ───────────────────────────
        hits = [h for h in hits
                if not _CORRUPTION_RE.search(h.get("content", "")[:2000])]

        # ── Stage 2: Hard-drop superseded chunks (retrieval_priority == 0.0) ─
        hits = [h for h in hits if h.get("retrieval_priority", 1.0) > 0.0]

        if not hits:
            return [], errors

        # ── Stage 3: CDS boost (content_boost × retrieval_priority) ──────
        hits = _apply_cds_boost(hits)

        # ── Stage 4: Action-query boosts (context demote + actionability + type bonus) ─
        if _is_action_query(query):
            hits = _apply_action_pipeline(hits)

        # ── Stage 5: Population filter ────────────────────────────────────
        hits = _apply_population_filter(hits, query)

        # ── Stage 6: Domain coherence penalty ────────────────────────────
        hits = _apply_domain_coherence(hits, query)

        # ── Stage 7: Condition title/content exclusions ───────────────────
        hits = _apply_condition_exclusions(hits, query)

        # ── Stage 8: Rescue actionable background chunks ─────────────────
        hits = _apply_background_rescue(hits, query)

        # ── Stage 9: Soft corruption demote ──────────────────────────────
        hits = _apply_soft_corruption_demote(hits)

        # ── Stage 10: Source diversity cap ───────────────────────────────
        hits = _apply_source_diversity(hits)
        hits.sort(key=lambda x: x["score"], reverse=True)

        return hits, errors

    def search(
        self,
        query: str,
        k: int = 5,
        snippet_chars: int = 15_000,
        search_mode: str = "rrf",
        threshold: float = 0.0,
        safe_top1_guardrail: bool = False,
    ) -> Dict[str, Any]:
        """Query the v2 KB and return a stable structured response.

        Always returns top-5 results (or fewer if the corpus has < 5 matching
        non-superseded chunks above the optional threshold).

        search_mode:
          lex  — BM25 lexical (fast, no GPU)
          sem  — Semantic vector (requires EmbedGemma model)
          rrf  — Reciprocal Rank Fusion of lex + sem (default, best quality)

        quality_flags is a list of diagnostic strings — never causes results
        to be dropped or zeroed out.
        """
        start = time.time()
        self.initialize()

        sm    = (search_mode or "rrf").lower()
        query = _expand_query(query)

        # ── Pipeline ──────────────────────────────────────────────────────
        hits, errors = self._search_pipeline(query, snippet_chars, sm)

        # ── Caller-supplied score threshold ───────────────────────────────
        if threshold > 0.0:
            hits = [h for h in hits if h["score"] >= threshold]

        # ── Intent-constrained reranking ──────────────────────────────────
        hits = _intent_rerank(hits, query)

        # ── Always-on alignment guardrail (55% threshold) ─────────────────
        hits = _promote_aligned_top_hit(hits, query)

        # ── Optional strict guardrail (70% threshold, actionable types) ───
        top1_swapped      = False
        top1_swap_reason: Optional[str] = None
        if safe_top1_guardrail and len(hits) > 1:
            hits, top1_swapped, top1_swap_reason = _safe_top1_guardrail(hits, query)

        # ── Top-k trim after all reranking / guardrails ──────────────────
        hits = hits[:k]

        # ── Quality flags (observational only — never drops results) ──────
        quality_flags: List[str] = []

        if any(_CORRUPTION_RE.search(h.get("content", "")[:300]) for h in hits[:3]):
            quality_flags.append("corruption_in_top3")

        if hits and hits[0].get("score", 0.0) < _LOW_CONFIDENCE_SCORE:
            quality_flags.append("low_confidence")

        if hits:
            top = hits[0]
            condition_match_re: Optional[re.Pattern] = None
            for _cid, q_re, c_re, _x_re in _CONDITION_VOCAB:
                if q_re.search(query):
                    condition_match_re = c_re
                    break
            if condition_match_re is not None:
                top_combined = (top.get("title", "") + " " + top.get("content", "")[:1200])
                active_coh   = [cr for q_re, cr in _CONDITION_COHERENCE if q_re.search(query)]
                coherence_miss = (not active_coh) or (
                    not any(cr.search(top_combined) for cr in active_coh))
                acs_miss = bool(
                    _ACS_QUERY_RE.search(query) and not _ACS_CHUNK_RE.search(top_combined))
                top_actionable = _is_actionable_hit(top)
                component_q    = bool(_TREATMENT_COMPONENT_Q.search(query))
                if (((not condition_match_re.search(top_combined)) and coherence_miss)
                        or acs_miss) and not (component_q and top_actionable):
                    quality_flags.append("off_condition_top")

        if _is_action_query(query) and hits:
            if not any(_is_actionable_hit(h) for h in hits[:3]):
                quality_flags.append("no_actionable_in_top3")

        if hits and hits[0].get("retrieval_priority", 1.0) < _LOW_PRIORITY_TOP_THRESHOLD:
            quality_flags.append("low_priority_top")

        hit = hits[0] if hits else None
        latency_ms = (time.time() - start) * 1000.0

        return {
            "query":            query,
            "hit":              hit,
            "hits":             hits,
            "latency_ms":       latency_ms,
            "errors":           errors,
            "quality_flags":    quality_flags,
            "top1_swapped":     top1_swapped,
            "top1_swap_reason": top1_swap_reason,
        }

    def stats(self) -> Dict[str, Any]:
        """Expose index stats for diagnostics."""
        self.initialize()
        out: Dict[str, Any] = {"index_loaded": self._mem is not None}
        if self._mem is not None:
            try:
                out["index_stats"] = self._mem.stats()
            except Exception:
                out["index_stats"] = {}
        return out
