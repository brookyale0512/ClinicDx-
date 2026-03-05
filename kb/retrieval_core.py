#!/usr/bin/env python3
"""
Shared retrieval core for local KB access.

This module centralizes memvid index loading and query logic so both daemon and
direct callers can share the same behavior.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from kb.embedder import EmbedGemmaEmbedder
except ImportError:
    from embedder import EmbedGemmaEmbedder  # type: ignore[no-redef]

KB_DIR = "/var/www/kbToolUseLora/kb"
WHO_INDEX = os.path.join(KB_DIR, "who_knowledge.mv2")

SOURCE_WHO = "WHO Guidelines"

# ── Fix 4: Content-type CDS boost/demote multipliers ──────────────────────
# Promotes actionable clinical chunks; demotes GRADE/GDG methodology chunks
_CDS_CONTENT_BOOST = {
    "dosage_table":         1.50,
    "treatment_protocol":   1.40,
    "danger_signs":         1.40,
    "recommendation":       1.30,
    "imci_classification":  1.30,
    "fluid_rehydration":    1.30,
    "diagnostic_criteria":  1.20,
    "adverse_effects":      1.10,
    "monitoring_schedule":  1.05,
    "contraindication":     1.10,
    "drug_interaction":     1.10,
    "regular":              1.00,
    "programme_table":      0.75,
    "evidence_profile":     0.20,   # GRADE methodology — needs raw>0.25 to survive 0.05 action floor
    "evidence_to_decision": 0.10,   # GDG deliberation — needs raw>0.50 to survive; typically below floor
}

# ── Fix 5: Metadata line prefixes to strip from content snippets ──────────
_METADATA_LINE_PREFIXES = (
    "title: ", "uri: mv2://", "tags: ", "labels: ",
    "chunk_id: ", "content_type: ", "extractous_metadata:",
    "memvid.embedding.", "page_numbers:", "pdf_file:",
    "headings:", "hierarchy_level:", "memvid.",
)

# ── Fix R3: Weak-title penalty (retrieval-time, regular content_type only) ─
_WEAK_TITLE_PENALTY_RE = re.compile(
    r'^(?:remarks?|symptomatic\s+treatment|additional\s+considerations?'
    r'|examples?|harms?|further\s+information'
    r'|\d{4}$'                          # bare year: "2019"
    r'|[A-Z]\.\s+\w{1,8}$'             # "B. Doses", "A. Notes"
    r'|\d+[\.\)]\s*(?:if|in|for|when|the|a)\b'  # "2) If the pathogen is unknown"
    r')',
    re.IGNORECASE,
)
_TITLE_ONLY_FRAGMENT_RE = re.compile(
    r'^\s*(?:chapter\s+\d+[:.]?|recommendations?|new\s+recommendation'
    r'|summary|conclusion|background|introduction|foreword|footnotes?'
    r'|second edition|©\s*world health organization.*)\s*$',
    re.IGNORECASE,
)

# ── Fix R4: Action-query detection and re-rank bonus ──────────────────────
_ACTION_DOSE_RE = re.compile(
    r'\b(?:dose|dosage|mg|mcg|units?|administer|prescri(?:be|ption)'
    r'|infusion|IV\b|IM\b|PO\b|antibiotic|antifungal|antiviral'
    r'|vasopressor|first.?line|empiric(?:al)?|regimen)\b',
    re.IGNORECASE,
)
_ACTION_TASK_RE = re.compile(
    r'\b(?:treat(?:ment|ing)?|manage(?:ment)?|protocol|guideline|algorithm'
    r'|first.?line|empiric(?:al)?|referral|transfer|stabiliz(?:e|ation)'
    r'|pre.?referral|prevention|prophylax(?:is)?)\b',
    re.IGNORECASE,
)
# F6: Extended emergency lexicon
_ACTION_CONDITION_RE = re.compile(
    r'\b(?:treat(?:ment|ing)?|manage(?:ment)?|protocol'
    r'|sepsis|malaria|tuberculosis|TB\b|meningitis|pneumonia'
    r'|eclampsia|pre.?eclampsia|DKA|ketoacidosis|anaphylaxis'
    r'|haemorrhage|hemorrhage|PPH|shock|stroke|infection'
    r'|how\s+to\s+treat|how\s+to\s+manage'
    r'|emergency|acute\s+(?:management|treatment)|severe|critical\s+(?:illness|care)'
    r'|hypertensive\s+(?:emergency|crisis)|airway|resuscitation|cardiac\s+arrest)\b',
    re.IGNORECASE,
)
_ACTION_PRIORITY_TYPES = {
    "dosage_table", "treatment_protocol", "danger_signs",
    "recommendation", "imci_classification", "fluid_rehydration",
}
_ACTION_BONUS = 1.15   # 15% bonus for action-type chunks on action queries
_ACTIONABLE_TEXT_BONUS = 1.08  # small boost for protocol-like language in regular chunks

# ── Strict CDS mode: permitted content types ──────────────────────────────
# Precision-over-recall set: return nothing rather than generic blobs
_STRICT_CDS_TYPES = {"dosage_table", "treatment_protocol", "danger_signs", "recommendation"}

# ── Fix R5: Paediatric source penalty (tightened regex, no "birth") ───────
_PAEDIATRIC_PDFS = {
    "WHO_Guide_childrencare.pdf",
    # Add others confirmed as paediatric-only after inspection
}
_PAEDIATRIC_QUERY_RE = re.compile(
    r'\b(?:child(?:ren)?|infant|neonate|neonatal|paediatric|pediatric'
    r'|newborn|under.?5|under.?one)\b',
    re.IGNORECASE,
)

# ── Fix R6: Conditional evidence_profile hard-exclusion floor ─────────────
_EVIDENCE_HARD_FLOOR = 0.08   # exclude e/e2d chunks below this (after R2 demotion)

# ── Hard intent-constrained reranking ─────────────────────────────────────
# Extracts 4 intent slots from the query: condition, severity, population, task.
# For each retrieved chunk, scores slot alignment and applies:
#   - Multiplicative penalties for clear mismatch  (drives chunk down)
#   - Additive boost for confirmed slot alignment  (drives chunk up)
# Final adjusted score: raw_score * penalty * (1 + boost)
# Penalties stack but floor at 0.10 so chunks remain visible to callers.

# ── Slot 1: Condition vocabulary ──────────────────────────────────────────
# Tuples: (canonical_id, query_detect_re, chunk_match_re, chunk_exclude_re|None)
# chunk_exclude: clear tokens that place a chunk in a DIFFERENT condition.
def _cre(q, c, x=None, f=re.IGNORECASE):
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
        r'\btyphoid\b|\benteric fever\b|\bsalmonella typhi\b|\bceftriaxone\b|\bazithromycin\b',
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
        r'\bhypertensiv[e ]\b|\bhypertensive emergency\b',
        r'\bhypertensiv\b|\bblood pressure\b',
        r'\bmalaria\b|\bmeningitis\b')),
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

# ── Slot 2: Severity ──────────────────────────────────────────────────────
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

# ── Slot 3: Population ────────────────────────────────────────────────────
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

# ── Slot 4: Task ──────────────────────────────────────────────────────────
_TASK_DOSE_Q      = re.compile(r'\bdose\b|\bdosage\b|\bmg(?:/kg)?\b|\bregimen\b|\badminister\b|\binfusion\b|\bhow much\b', re.IGNORECASE)
_TASK_DIAGNOSE_Q  = re.compile(r'\bdiagnos(?:e|is|tic)\b|\bcriteria\b|\bsigns?\b|\bdifferential\b|\bidentif\b', re.IGNORECASE)
_TASK_FIRSTLINE_Q = re.compile(r'\bfirst.?line\b|\bempiric\b|\binitial treatment\b', re.IGNORECASE)
_TASK_REFERRAL_Q  = re.compile(r'\breferral\b|\bwhen to refer\b|\bbefore referral\b|\btransfer\b', re.IGNORECASE)
_TASK_PREVENT_Q   = re.compile(r'\bprevention\b|\bprophylax\b|\bprevent\b|\bvaccin\b', re.IGNORECASE)
_TREATMENT_COMPONENT_Q = re.compile(
    r'\b(?:vasopressor|norepinephrine|noradrenaline|epinephrine|adrenaline'
    r'|dopamine|dobutamine|ceftriaxone|ciprofloxacin|amoxicillin|dose|dosage'
    r'|mg(?:/kg)?|infusion|iv\b|im\b|oral)\b',
    re.IGNORECASE,
)

_TASK_PREFERRED_TYPES: Dict[str, set] = {
    "dose":       {"dosage_table", "treatment_protocol"},
    "diagnosis":  {"diagnostic_criteria", "danger_signs", "imci_classification"},
    "first_line": {"treatment_protocol", "dosage_table", "recommendation"},
    "referral":   {"treatment_protocol", "recommendation", "danger_signs"},
    "prevention": {"recommendation"},
}

# ── Annex / change-summary hard-demote ────────────────────────────────────
_ANNEX_TITLE_RE = re.compile(
    r'^\s*(?:annex|appendix|table of|list of|summary of changes|change.*summary)',
    re.IGNORECASE,
)
_GENERIC_SECTION_TITLE_RE = re.compile(
    r'^\s*(?:\d+[\.\)]\s*)?(?:chapter\s+\d+[:.]?\s*)?'
    r'(?:management|monitoring|recommendations?|evidence'
    r'|recommendation and supporting evidence|background|approach|assessment'
    r'|recommendations?\s+and\s+evidence)'
    r'(?:\s*\(page\s*\d+/\d+\))?\s*$',
    re.IGNORECASE,
)
_NOISE_ANCHOR_TITLE_RE = re.compile(
    r'^\s*(?:8\.\s*violence and injury prevention'
    r'|emergency tr(?:i| )age assessment and treatment'
    r'|4\.\s*severe dehydration|second edition)\s*$',
    re.IGNORECASE,
)


def _intent_rerank(hits: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Hard intent-constrained reranking (4-slot).

    Extracts condition / severity / population / task from the query, then
    scores each hit for slot alignment.  Returns hits sorted by:
        raw_score * penalty * (1 + boost)

    Penalties (< 1.0) stack multiplicatively for mismatching slots.
    Boosts (additive) accumulate for confirmed matching slots.
    Floor of 0.10 prevents any hit from disappearing entirely.
    """
    if not hits:
        return hits

    # ── Extract query intent slots ─────────────────────────────────────────
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

    # ── Score each hit ─────────────────────────────────────────────────────
    def _adjusted(hit: Dict[str, Any]) -> float:
        title    = (hit.get("title") or "").strip()
        content  = hit.get("content") or ""
        ct       = hit.get("content_type") or ""
        combined = title.lower() + " " + content.lower()
        chunk_on_target = False

        penalty       = 1.0
        boost         = 0.0
        matched_slots = 0

        # Annex / change-summary → hard demote regardless of other signals
        if _ANNEX_TITLE_RE.search(title):
            penalty *= 0.25

        # ── Condition ──────────────────────────────────────────────────────
        if condition and cond_c_re:
            chunk_on_target = bool(cond_c_re.search(combined))
            chunk_off_target = bool(cond_x_re.search(combined)) if cond_x_re else False
            if chunk_off_target and not chunk_on_target:
                penalty *= 0.12          # clearly wrong condition
            elif chunk_off_target and chunk_on_target:
                penalty *= 0.55          # mixed/confuser content
            elif chunk_on_target:
                boost += 0.25            # condition confirmed
                matched_slots += 1
        if condition == "snakebite" and re.search(r'\bscorpion\b', combined, re.I):
            penalty *= 0.05
            if boost >= 0.25:
                boost -= 0.25
            if matched_slots > 0:
                matched_slots -= 1
        if condition == "rabies" and re.search(r'\bmalaria\b|\bRTS,S\b|\bcasirivimab\b|\bimdevimab\b', combined, re.I):
            penalty *= 0.20
        if condition and _GENERIC_SECTION_TITLE_RE.search(title) and not chunk_on_target:
            penalty *= 0.45
        if condition and _NOISE_ANCHOR_TITLE_RE.search(title) and not chunk_on_target:
            penalty *= 0.35

        # ── Severity ───────────────────────────────────────────────────────
        if sev_light:
            if _SEV_HEAVY_CHUNK.search(combined):
                penalty *= 0.35          # severe chunk for uncomplicated query
            elif _SEV_LIGHT_CHUNK.search(combined):
                boost += 0.10
                matched_slots += 1
        elif sev_heavy:
            if _SEV_HEAVY_CHUNK.search(combined):
                boost += 0.10
                matched_slots += 1

        # DS-TB query → penalise DR-TB / MDR-TB / BPaLM chunks
        if ds_tb and _DR_TB_CHUNK.search(combined):
            penalty *= 0.25

        # ── Population ─────────────────────────────────────────────────────
        if pop_adult:
            if _POP_NEONATE_CHUNK.search(combined):
                penalty *= 0.30          # neonatal chunk for adult query: high-risk
            elif _POP_CHILD_CHUNK.search(combined) and not _POP_ADULT_CHUNK.search(combined):
                penalty *= 0.60          # paediatric-only, no adult signal
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

        # ── Task ───────────────────────────────────────────────────────────
        if task:
            if ct in _TASK_PREFERRED_TYPES.get(task, set()):
                boost += 0.10
                matched_slots += 1

        # ── Multi-slot match bonus ─────────────────────────────────────────
        if matched_slots >= 3:
            boost += 0.15
        elif matched_slots == 2:
            boost += 0.05

        # Floor: keep chunk visible even under maximum stacked penalties
        penalty = max(penalty, 0.10)

        return hit["score"] * penalty * (1.0 + boost)

    adjusted_hits: List[Tuple[Dict[str, Any], float]] = [(h, _adjusted(h)) for h in hits]

    # B4: For light-severity queries, keep best on-condition chunk visible when
    # all on-condition chunks are heavy-only matches (avoid empty off-condition top ranks).
    if sev_light and condition and cond_c_re:
        on_condition: List[Tuple[Dict[str, Any], float]] = []
        has_light_on_condition = False
        for h, sc in adjusted_hits:
            combined = ((h.get("title") or "") + " " + (h.get("content") or "")).lower()
            if cond_c_re.search(combined):
                on_condition.append((h, sc))
                if _SEV_LIGHT_CHUNK.search(combined):
                    has_light_on_condition = True
        if on_condition and not has_light_on_condition:
            best_hit, _ = max(on_condition, key=lambda item: item[1])
            relax_factor = 0.50 / 0.35  # soften severe-only suppression for one best candidate
            adjusted_hits = [
                (h, (sc * relax_factor) if h is best_hit else sc)
                for h, sc in adjusted_hits
            ]

    adjusted_hits.sort(key=lambda item: item[1], reverse=True)
    for h, adjusted in adjusted_hits:
        h["score"] = float(adjusted)
    return [h for h, _ in adjusted_hits]


# ── F1: Retrieval-time corruption filter ──────────────────────────────────
_RETRIEVAL_CORRUPTION_RE = re.compile(
    r'·i\b|recursively\b|>loc[_\s]\d+|\blocas\b'
    r'|smoker,\s+and\s+a\s+person\s+who\s+is\s+not\s+smoking'
    r'|ganges\s+river'
    r'|methanibutol|diploplohydria|hydratraemia|hypokatropaenia',
    re.IGNORECASE,
)
_RETRIEVAL_CORRUPTION_SOFT_RE = re.compile(
    r'\b(?:AND OR|OR RF|maximum\s+maximum)\b',
    re.IGNORECASE,
)

# ── F2: Query synonym expansion ───────────────────────────────────────────
_QUERY_SYNONYM_MAP: Dict[str, List[str]] = {
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
    # H2: Phase H additions
    "dysentery":         ["Shigella", "ciprofloxacin"],
    "antivenom":         ["envenomation", "snake bite"],
    "envenomation":      ["antivenom", "snake bite"],
    "SAM":               ["severe acute malnutrition", "RUTF"],
    "haloperidol":       ["antipsychotic", "psychosis"],
    "resuscitation":     ["airway breathing circulation"],
}
_SYNONYM_DETECT: Dict[str, re.Pattern] = {
    term: re.compile(r'\b' + re.escape(term) + r'\b', re.I)
    for term in _QUERY_SYNONYM_MAP
}

# ── F3: Domain coherence scoring ──────────────────────────────────────────
_CONDITION_COHERENCE = [
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
    # H1a: Dysentery — guards against PKDL/plague contamination (Q17)
    (re.compile(r'\b(?:dysentery|bloody\s+diarrh?oea|shigella)\b', re.I),
     re.compile(r'\b(?:dysentery|shigella|ciprofloxacin|azithromycin|bloody\s+stool|acute\s+diarr?hoea)\b', re.I)),
    # H1b: Cardiac arrest — prevents generic shock chunk from dominating (Q20)
    (re.compile(r'\b(?:cardiac\s+arrest|cardiopulmonary\s+resuscitation|AED)\b', re.I),
     re.compile(r'\b(?:cardiac\s+arrest|CPR|defibrillat|compressions?|AED|resuscitat|adrenaline|epinephrine|atropine|airway)\b', re.I)),
    # H1c: Snakebite envenomation (snake-specific trigger only)
    (re.compile(r'\b(?:snake.?bite|snake\s+(?:venom|envenomation)|snakebite)\b', re.I),
     re.compile(r'\b(?:antivenom|snake\s+(?:bite|venom)|polyvalent|antitoxin)\b', re.I)),
    # H1c2: Scorpion envenomation
    (re.compile(r'\b(?:scorpion(?:\s+(?:sting|envenomation))?)\b', re.I),
     re.compile(r'\b(?:scorpion|sting|envenomation|antivenom|analgesic)\b', re.I)),
    # H1d: Neonatal sepsis — prevents ophthalmia neonatorum from winning (Q04)
    # .{0,60} limits cross-term span so "neonatal ... sepsis" within a short query fires
    (re.compile(r'\b(?:neonatal|newborn)\b.{0,60}\b(?:sepsis|infection|bacteremia|antibiotic)\b', re.I),
     re.compile(r'\b(?:sepsis|septic|ampicillin|gentamicin|bacteremia|bacteraemia|neonatal\s+(?:sepsis|infection))\b', re.I)),
    # H1e: Malnutrition/SAM guard
    (re.compile(r'\b(?:severe\s+acute\s+malnutrition|SAM\b|RUTF|marasmus|kwashiorkor)\b', re.I),
     re.compile(r'\b(?:malnutrition|RUTF|kwashiorkor|marasmus|F-75|F-100|therapeutic\s+feeding|wasting|MUAC|undernutrition)\b', re.I)),
]
_DOMAIN_COHERENCE_PENALTY = 0.65
_DOMAIN_COHERENCE_PENALTY_BOOSTED_OFFTARGET = 0.20

# ── H5: Condition title exclusion — demote cross-disease title contamination ─
_CONDITION_TITLE_EXCLUSIONS = [
    # meningitis query: demote chunks explicitly titled with "pneumonia"
    (re.compile(r'\b(?:meningitis|meningococcal)\b', re.I),
     re.compile(r'\bpneumonia\b', re.I)),
    # pneumonia query: demote chunks explicitly titled with "meningitis"
    (re.compile(r'\b(?:pneumonia|community.acquired\s+pneumonia)\b', re.I),
     re.compile(r'\bmeningitis\b', re.I)),
    # NTD skin queries: demote envenomation/confuser titles
    (re.compile(r'\b(?:buruli|podoconiosis|tungiasis|noma|cancrum oris|necroti[sz]ing gingivitis)\b', re.I),
     re.compile(r'\b(?:scorpion|snake|envenomation|echinococcosis|PKDL)\b', re.I)),
    # Dysentery query: demote STI/sexual-assault treatment chunks.
    (re.compile(r'\b(?:dysentery|shigella|bloody\s+diarrh?oe?a)\b', re.I),
     re.compile(r'\b(?:sexual assault|sti|sexually transmitted|post.?exposure prophylaxis)\b', re.I)),
    # Rabies query: demote TB-specific titles
    (re.compile(r'\b(?:rabies|post.?exposure prophylaxis|PEP)\b', re.I),
     re.compile(r'\b(?:TB|tuberculosis|rifapentine|isoniazid)\b', re.I)),
    # Dengue fluid-management query: demote generic dehydration plan chunks.
    (re.compile(r'\b(?:dengue)\b', re.I),
     re.compile(r'\b(?:if intravenous treatment|plan c|severe dehydration|if there is high fever|when to stop vasopressors)\b', re.I)),
    # Burn-resuscitation query: demote generic pediatric/dehydration chunks.
    (re.compile(r'\b(?:burn(?:s|ed)?|thermal injury|TBSA|parkland)\b', re.I),
     re.compile(r'\b(?:severe dehydration|if there is high fever|children under 2 months)\b', re.I)),
    # Snakebite query: demote scorpion-specific titles.
    (re.compile(r'\b(?:snake.?bite|antivenom)\b', re.I),
     re.compile(r'\bscorpion\b', re.I)),
]
_CONDITION_TITLE_EXCLUSION_PENALTY = 0.60
_CONDITION_TITLE_EXCLUSION_OVERRIDES = [
    # Stronger pairing-specific suppression for meningitis/pneumonia bleed-through.
    (re.compile(r'\b(?:meningitis|meningococcal)\b', re.I),
     re.compile(r'\bpneumonia\b', re.I), 0.35),
    (re.compile(r'\b(?:pneumonia|community.acquired\s+pneumonia)\b', re.I),
     re.compile(r'\bmeningitis\b', re.I), 0.35),
]
_CONDITION_CONTENT_EXCLUSIONS = [
    # Dysentery queries: demote STI/sexual-assault content even under generic titles.
    (re.compile(r'\b(?:dysentery|shigella|bloody\s+diarrh?oe?a)\b', re.I),
     re.compile(r'\b(?:sexual assault|sexually transmitted|post.?exposure prophylaxis'
                r'|sex transm infect|hiv pep)\b', re.I), 0.20),
]

# ── F4: Context-type demotion on action queries ───────────────────────────
_ACTION_CONTEXT_DEMOTE = {"recommendation", "diagnostic_criteria", "programme_table"}
# NOT "regular" — clinically useful regular chunks must not be universally suppressed
_ACTION_CONTEXT_DEMOTE_FACTOR = 0.75

# ── F5: Actionability text bonus regexes ──────────────────────────────────
_DOSE_NUMBER_RE = re.compile(
    r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|mmol|mL|ml|g/kg|mg/kg|mcg/kg|IU|units?|%)\b'
    r'|\b\d+(?:\.\d+)?\s*(?:mcg|µg)/(?:kg/)?min\b',
    re.IGNORECASE,
)
_ROUTE_ADMIN_RE = re.compile(
    r'\b(?:IV\b|IM\b|PO\b|SC\b|oral(?:ly)?|intravenous|intramuscular|subcutaneous|infusion|injection)\b',
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

# ── F7: Tail precision delta filter constants ─────────────────────────────
_MIN_KEEP = 4
_MIN_TAIL_RATIO = 0.30

# ── F9: Population filter — OBG inverse ───────────────────────────────────
_OBG_PDFS = {"MSF_OBG.pdf"}   # expand after confirming which PDFs are OBG-specific
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


def _is_action_query(query: str) -> bool:
    """True when query has a condition signal and either dose or operational task intent."""
    return bool(_ACTION_CONDITION_RE.search(query) and
                (_ACTION_DOSE_RE.search(query) or _ACTION_TASK_RE.search(query)))


def _expand_query(query: str) -> str:
    """F2: Append canonical synonym aliases as a dedup suffix; no inline substitution."""
    aliases: set = set()
    for term, expansions in _QUERY_SYNONYM_MAP.items():
        if _SYNONYM_DETECT[term].search(query):
            aliases.update(expansions)
    # Remove aliases already present in query (case-insensitive)
    query_lower = query.lower()
    aliases = {a for a in aliases if a.lower() not in query_lower}
    return (query + " " + " ".join(sorted(aliases))).strip() if aliases else query


def _apply_domain_coherence(hits: List[Dict], query: str) -> List[Dict]:
    """F3: Penalize hits with zero overlap with any active query condition (multi-trigger)."""
    active_checks = [content_re for query_re, content_re in _CONDITION_COHERENCE
                     if query_re.search(query)]
    if not active_checks:
        return hits
    for h in hits:
        combined = h.get("title", "") + " " + h.get("content", "")[:1000]
        if not any(cr.search(combined) for cr in active_checks):
            if h.get("content_type") in _ACTION_PRIORITY_TYPES:
                h["score"] *= _DOMAIN_COHERENCE_PENALTY_BOOSTED_OFFTARGET
            else:
                h["score"] *= _DOMAIN_COHERENCE_PENALTY
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _apply_title_exclusions(hits: List[Dict], query: str) -> List[Dict]:
    """H5: Demote hits whose title explicitly names a different disease than query."""
    for query_re, title_re in _CONDITION_TITLE_EXCLUSIONS:
        if query_re.search(query):
            for h in hits:
                if title_re.search(h.get("title", "")):
                    h["score"] *= _CONDITION_TITLE_EXCLUSION_PENALTY
    for query_re, title_re, factor in _CONDITION_TITLE_EXCLUSION_OVERRIDES:
        if query_re.search(query):
            for h in hits:
                if title_re.search(h.get("title", "")):
                    h["score"] *= factor
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _apply_content_exclusions(hits: List[Dict], query: str) -> List[Dict]:
    """Demote hits by content patterns when title-based exclusions are insufficient."""
    for query_re, content_re, factor in _CONDITION_CONTENT_EXCLUSIONS:
        if query_re.search(query):
            for h in hits:
                combined = (h.get("title", "") + " " + h.get("content", "")[:1200])
                if content_re.search(combined):
                    h["score"] *= factor
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _apply_corruption_demote(hits: List[Dict]) -> List[Dict]:
    """Soft-demote suspicious noisy chunks that evade hard corruption filters."""
    for h in hits:
        sample = (h.get("title", "") + " " + h.get("content", "")[:1200])
        if _RETRIEVAL_CORRUPTION_SOFT_RE.search(sample):
            h["score"] *= 0.45
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _actionability_score(content: str) -> float:
    """F5: Returns 1.00–1.21× based on presence of dose/route/frequency signals."""
    sample = content[:2000]
    n = (bool(_DOSE_NUMBER_RE.search(sample)) +
         bool(_ROUTE_ADMIN_RE.search(sample)) +
         bool(_FREQUENCY_RE.search(sample)))
    return 1.0 + 0.07 * n


def _is_actionable_hit(hit: Dict[str, Any]) -> bool:
    """Actionability heuristic used for ranking and quality flags."""
    if hit.get("content_type") in _ACTION_PRIORITY_TYPES:
        return True
    text = ((hit.get("title") or "") + "\n" + (hit.get("content") or ""))[:2000]
    return bool(_PROTOCOL_TEXT_RE.search(text))


def _is_heading_fragment(hit: Dict[str, Any]) -> bool:
    """Detect heading-like chunks with too little clinical utility."""
    title = (hit.get("title") or "").strip()
    content = (hit.get("content") or "").strip()
    if not title:
        return False
    if not _TITLE_ONLY_FRAGMENT_RE.match(title):
        return False
    # Keep concise but actionable heading chunks (rare), demote otherwise.
    if _PROTOCOL_TEXT_RE.search(content[:1200]):
        return False
    return len(content) < 320


def _promote_aligned_top_hit(hits: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Guardrail: if top hit is off-condition and a strong aligned alternative exists, promote it."""
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
            on_target = bool(cond_c_re.search(combined))
            off_target = bool(cond_x_re.search(combined)) if cond_x_re else False
            return on_target and not off_target
    else:
        active_checks = [content_re for query_re, content_re in _CONDITION_COHERENCE if query_re.search(query)]
        if not active_checks:
            return hits

        def _aligned(h: Dict[str, Any]) -> bool:
            combined = (h.get("title", "") + " " + h.get("content", "")[:1200])
            return any(cr.search(combined) for cr in active_checks)

    top = hits[0]
    if _aligned(top):
        return hits
    # Promote first aligned candidate if it is not a tiny tail-score outlier.
    for idx, cand in enumerate(hits[1:], start=1):
        if _aligned(cand) and cand.get("score", 0.0) >= top.get("score", 0.0) * 0.55:
            reordered = [cand] + hits[:idx] + hits[idx + 1:]
            return reordered
    return hits


def _is_blocked_title(title: str) -> bool:
    """Strict CDS: hard-block chunks with generic, garbled, or uninformative titles."""
    if not title:
        return False
    if _RETRIEVAL_CORRUPTION_RE.search(title):
        return True
    return bool(_WEAK_TITLE_PENALTY_RE.match(title.strip()))


def _apply_population_filter(hits: List[Dict], query: str) -> List[Dict]:
    """Fix R5: Demote paediatric-source chunks for non-paediatric queries.
    Fix F9: Demote OBG source chunks for explicit adult non-OBG queries.
    """
    if _PAEDIATRIC_QUERY_RE.search(query):
        return hits   # paediatric query — no penalty for either filter
    for h in hits:
        uri = h.get("uri", "")
        pdf = uri.split("/")[-2] if "/" in uri else ""
        if pdf in _PAEDIATRIC_PDFS:
            h["score"] *= 0.55
    # OBG inverse filter: demote obstetric sources for explicit adult non-OBG queries
    # Guard: skipped if ANY OBG/pregnancy signal present in query
    if _ADULT_NONOBG_RE.search(query) and not _OBG_QUERY_RE.search(query):
        for h in hits:
            uri = h.get("uri", "")
            pdf = uri.split("/")[-2] if "/" in uri else ""
            if pdf in _OBG_PDFS:
                h["score"] *= 0.60
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _extract_hits(raw: Any) -> List[Dict[str, Any]]:
    """Normalize mem.find() return shape."""
    if isinstance(raw, dict):
        hits = raw.get("hits", [])
        return hits if isinstance(hits, list) else []
    if isinstance(raw, list):
        return raw
    return []


def _normalize_hit(hit: Dict[str, Any], source_name: str, snippet_chars: int) -> Optional[Dict[str, Any]]:
    """Convert raw memvid hit into a stable response shape."""
    score = hit.get("score", 0)
    content = hit.get("snippet", hit.get("frame", ""))
    if not content:
        return None

    # Fix 5: Strip leaked metadata lines from content
    lines = str(content).split("\n")
    clean_lines = [l for l in lines if not any(
        l.strip().startswith(p) for p in _METADATA_LINE_PREFIXES
    )]
    content = "\n".join(clean_lines).strip()

    frame_id = hit.get("frame_id")

    # Fix 2: Strip #page-N suffix from URI (pagination artifact from BM25 frames)
    uri = hit.get("uri", "")
    if "#page-" in uri:
        uri = uri.split("#")[0]

    # Fix 4: Expose content_type so boost logic and callers can read it
    content_type = hit.get("metadata", {}).get("content_type", "regular")

    out = {
        "score": float(score),
        "title": hit.get("title", ""),
        "content": content[:snippet_chars],
        "source": source_name,
        "uri": uri,
        "frame_id": str(frame_id) if frame_id is not None else "",
        "content_type": content_type,
    }
    return out


def _rrf_merge(bm25_hits: List[Dict[str, Any]], sem_hits: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion of BM25 and semantic hit lists.

    Uses frame_id as the merge key.  Score = sum of 1/(k + rank + 1) across
    both lists.  k=60 is the standard bias term that dampens rank-1 advantage.

    Fix 2: Strips #page-N suffix before merging so all BM25 pagination frames
    for the same chunk collapse into a single RRF entry.
    """
    def _parent_id(fid: str) -> str:
        # Strip #page-N suffix — treat all pages of a chunk as one unit
        return fid.split("#")[0] if fid else fid

    scores: Dict[str, Dict[str, Any]] = {}
    for rank, hit in enumerate(bm25_hits):
        fid = _parent_id(hit.get("frame_id") or f"b{rank}")
        if fid not in scores:
            scores[fid] = {"hit": hit, "rrf": 0.0}
        scores[fid]["rrf"] += 1.0 / (k + rank + 1)
    for rank, hit in enumerate(sem_hits):
        fid = _parent_id(hit.get("frame_id") or f"s{rank}")
        if fid not in scores:
            scores[fid] = {"hit": hit, "rrf": 0.0}
        scores[fid]["rrf"] += 1.0 / (k + rank + 1)
    merged = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)
    return [dict(m["hit"], score=m["rrf"]) for m in merged]


def _apply_cds_boost(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix 4: Boost actionable CDS chunks; demote GRADE/GDG methodology chunks.
    Fix R3: Apply weak-title penalty to regular-type chunks with uninformative titles.
    """
    for h in hits:
        ct = h.get("content_type", "regular")
        h["score"] = h["score"] * _CDS_CONTENT_BOOST.get(ct, 1.0)
        # R3: Weak-title penalty — only for regular content_type to avoid penalising
        # dosage_table/protocol chunks whose section title happens to be generic
        if ct == "regular":
            title = h.get("title", "")
            if title and _WEAK_TITLE_PENALTY_RE.match(title.strip()):
                h["score"] *= 0.60
        # Issue 1: suppress heading-only fragments that consume top-k slots.
        if _is_heading_fragment(h):
            h["score"] *= 0.35
    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _apply_source_diversity(hits: List[Dict[str, Any]], max_per_source: int = 2) -> List[Dict[str, Any]]:
    """Fix 6: Demote 3rd+ results from the same PDF to break hub-chunk dominance."""
    seen_pdf: Dict[str, int] = {}
    result = []
    for h in hits:
        uri = h.get("uri", "")
        # Extract PDF identifier from uri: mv2://who/<pdf_file>/<chunk_id>
        parts = uri.split("/")
        pdf = parts[-2] if len(parts) >= 2 else ""
        count = seen_pdf.get(pdf, 0)
        if count < max_per_source:
            result.append(h)
        else:
            result.append(dict(h, score=h["score"] * 0.5))
        seen_pdf[pdf] = count + 1
    return sorted(result, key=lambda x: x["score"], reverse=True)


class KBRetriever:
    """Thread-safe retriever with lazy-loaded memvid handles."""

    def __init__(self, who_index: str = WHO_INDEX) -> None:
        self.who_index = who_index
        self._who_mem = None
        self._embedder: Optional[EmbedGemmaEmbedder] = None
        self._lock = threading.Lock()

    def initialize(self, enable_vec: bool = False) -> None:
        """Load indexes once.

        enable_vec: set True when the index was built with vector support and
        semantic / RRF search modes are required.
        """
        if self._who_mem is not None:
            return
        with self._lock:
            if self._who_mem is not None:
                return
            import memvid_sdk

            self._who_mem = memvid_sdk.use(
                "basic",
                self.who_index,
                read_only=True,
                enable_vec=enable_vec,
                enable_lex=True,
            )

    def _get_embedder(self) -> Optional["EmbedGemmaEmbedder"]:
        """Lazy-init the embedder; returns None if server is unavailable."""
        if self._embedder is None:
            emb = EmbedGemmaEmbedder()
            if emb.available:
                self._embedder = emb
        return self._embedder

    def _find_lex(self, mem: Any, query: str, k: int, snippet_chars: int) -> Any:
        return mem.find(query, k=k, snippet_chars=snippet_chars, mode="lex")

    def _find_sem(self, mem: Any, query: str, k: int, snippet_chars: int) -> Any:
        embedder = self._get_embedder()
        if embedder is None:
            raise RuntimeError("Embedding server unavailable for semantic search")
        return mem.find(query, k=k, snippet_chars=snippet_chars, mode="sem", embedder=embedder)

    def _search_multi(
        self,
        mem: Any,
        source_name: str,
        query: str,
        k: int,
        snippet_chars: int,
        search_mode: str = "bm25",
        strict_cds_mode: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        errors: List[str] = []
        hits: List[Dict[str, Any]] = []

        # F8: Wider candidate pool (3× k, cap 50) for action queries; guard against latency
        action_query = _is_action_query(query)
        k_internal = min(max(k * 3, 15), 50) if action_query else k

        def _collect(raw: Any) -> List[Dict[str, Any]]:
            result = []
            for item in _extract_hits(raw):
                normalized = _normalize_hit(item, source_name=source_name, snippet_chars=snippet_chars)
                if normalized:
                    result.append(normalized)
            return result

        try:
            if search_mode == "sem":
                raw = self._find_sem(mem, query, k_internal, snippet_chars)
                hits = _collect(raw)
            elif search_mode == "rrf":
                raw_lex = self._find_lex(mem, query, k_internal, snippet_chars)
                raw_sem = self._find_sem(mem, query, k_internal, snippet_chars)
                bm25_hits = _collect(raw_lex)
                sem_hits = _collect(raw_sem)
                hits = _rrf_merge(bm25_hits, sem_hits)
            else:  # default: bm25
                raw = self._find_lex(mem, query, k_internal, snippet_chars)
                hits = _collect(raw)
        except Exception as exc:  # pragma: no cover - defensive on memvid bindings
            errors.append(f"{source_name}: {exc}")
            return [], errors

        # F1: Drop corrupted chunks — scan first 2000 chars (corruption can be deeper in chunks)
        hits = [h for h in hits
                if not _RETRIEVAL_CORRUPTION_RE.search(h.get("content", "")[:2000])]

        # R3/Fix4: Apply CDS content-type boost/demote (includes weak-title penalty)
        hits = _apply_cds_boost(hits)

        # F4: Demote policy/context types on action queries
        # recommendation: 1.30× → 0.975× effective; dosage_table stays 1.725×
        if action_query:
            for h in hits:
                if h.get("content_type") in _ACTION_CONTEXT_DEMOTE:
                    h["score"] *= _ACTION_CONTEXT_DEMOTE_FACTOR
            hits.sort(key=lambda x: x["score"], reverse=True)

        # F5: Actionability text bonus (1.00–1.21×) — applied before R4 priority bonus
        if action_query:
            for h in hits:
                h["score"] *= _actionability_score(h.get("content", ""))
            hits.sort(key=lambda x: x["score"], reverse=True)

        # R4: Apply 15% bonus to action-priority chunks on action queries
        # Stacks on top of CDS boost (e.g. dosage_table 1.50× → 1.725× net; with F5 up to 1.966×)
        if action_query:
            for h in hits:
                if h.get("content_type") in _ACTION_PRIORITY_TYPES:
                    h["score"] *= _ACTION_BONUS
            hits.sort(key=lambda x: x["score"], reverse=True)

        # Small extra lift for protocol-like language even in regular chunks.
        if action_query:
            for h in hits:
                if _is_actionable_hit(h):
                    h["score"] *= _ACTIONABLE_TEXT_BONUS
            hits.sort(key=lambda x: x["score"], reverse=True)

        # R6: Conditional hard-exclusion of evidence chunks on action queries
        # Keeps high-scoring e/e2d chunks (only source for rare queries) but drops low scorers
        if action_query:
            hits = [h for h in hits
                    if h.get("content_type") not in ("evidence_profile", "evidence_to_decision")
                    or h["score"] >= _EVIDENCE_HARD_FLOOR]

        # R5 + F9: Population filter (paediatric penalty + OBG inverse)
        hits = _apply_population_filter(hits, query)

        # F3: Domain coherence penalty (multi-trigger — penalize zero-overlap hits)
        hits = _apply_domain_coherence(hits, query)

        # H5: Condition title exclusion — demote cross-disease title contamination
        hits = _apply_title_exclusions(hits, query)
        hits = _apply_content_exclusions(hits, query)

        # Issue 3: additional soft demotion for noisy OCR/extraction artifacts.
        hits = _apply_corruption_demote(hits)

        # Fix 6: Source diversity cap (prevent hub-chunk dominance)
        hits = _apply_source_diversity(hits)
        hits.sort(key=lambda item: item["score"], reverse=True)

        # F7: Tail delta filter — trim weak tail
        # strict_cds_mode: no minimum keep (prefer empty over weak)
        # normal mode:     always keep ≥_MIN_KEEP hits if available
        if action_query and hits:
            top_score = hits[0]["score"]
            if strict_cds_mode:
                hits = [h for h in hits if h["score"] >= top_score * _MIN_TAIL_RATIO]
            elif len(hits) > _MIN_KEEP:
                hits = (hits[:_MIN_KEEP] +
                        [h for h in hits[_MIN_KEEP:]
                         if h["score"] >= top_score * _MIN_TAIL_RATIO])

        # Strict CDS: filter to action-oriented types only + hard-block generic/garbled titles
        if strict_cds_mode:
            hits = [h for h in hits if h.get("content_type") in _STRICT_CDS_TYPES]
            hits = [h for h in hits if not _is_blocked_title(h.get("title", ""))]

        if k and len(hits) > k:
            hits = hits[:k]
        return hits, errors

    def _search_single(
        self,
        mem: Any,
        source_name: str,
        query: str,
        k: int,
        snippet_chars: int,
        search_mode: str = "bm25",
        strict_cds_mode: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        hits, errors = self._search_multi(mem, source_name, query, k, snippet_chars,
                                          search_mode=search_mode, strict_cds_mode=strict_cds_mode)
        return (hits[0] if hits else None), errors

    def search(
        self,
        query: str,
        k: int = 3,
        snippet_chars: int = 15000,
        source_mode: str = "who",
        threshold: float = 0.0,
        search_mode: str = "bm25",
        strict_mode: bool = False,
        strict_cds_mode: bool = False,
        safe_top1_guardrail: bool = False,
    ) -> Dict[str, Any]:
        """
        Query KB and return stable structured response.

        source_mode:
          - who: WHO only (canonical production mode)
          - auto: accepted for backward compatibility; treated as WHO-only

        search_mode:
          - bm25: Tantivy BM25 lexical search (default, no embedding server needed)
          - sem:  Semantic vector search (requires vec-enabled index + embedding server)
          - rrf:  Reciprocal Rank Fusion of bm25 + sem results

        strict_mode:     if True, zero out hits when quality_flags are triggered (F10)
        strict_cds_mode: if True, filter to action-oriented types only; no minimum-keep;
                         hard-block generic/garbled titles (precision over recall)
        """
        start = time.time()
        self.initialize()

        mode = (source_mode or "who").lower()
        errors: List[str] = []
        hit: Optional[Dict[str, Any]] = None
        hits: List[Dict[str, Any]] = []

        sm = (search_mode or "bm25").lower()
        # F2: Expand query with synonym aliases (dedup suffix, no inline substitution)
        query = _expand_query(query)

        scds = strict_cds_mode  # shorthand for all _search_multi calls below

        if mode not in {"who", "auto"}:
            errors.append(f"unsupported source_mode '{mode}'; using who")
        who_hits, errs = self._search_multi(
            self._who_mem, SOURCE_WHO, query, k, snippet_chars, search_mode=sm, strict_cds_mode=scds
        )
        errors.extend(errs)
        hits = who_hits
        hit = who_hits[0] if who_hits else None

        # Apply caller-provided minimum threshold only.
        # No hardcoded floor for action queries.
        effective_min_score = threshold
        if effective_min_score > 0:
            hits = [h for h in hits if h["score"] >= effective_min_score]
            if hit and hit["score"] < effective_min_score:
                hit = None

        # Intent-constrained reranking: penalise hits that contradict query intent
        hits = _intent_rerank(hits, query)
        hits = _promote_aligned_top_hit(hits, query)

        top1_swapped = False
        top1_swap_reason: Optional[str] = None

        # Optional safety guardrail: only intervene for clear off-condition top results.
        if safe_top1_guardrail and len(hits) > 1:
            condition_match_re: Optional[re.Pattern] = None
            active_coherence: List[re.Pattern] = []
            for _cid, q_re, c_re, _x_re in _CONDITION_VOCAB:
                if q_re.search(query):
                    condition_match_re = c_re
                    break
            if condition_match_re is not None:
                active_coherence = [content_re for query_re, content_re in _CONDITION_COHERENCE
                                    if query_re.search(query)]
                top = hits[0]
                top_combined = ((top.get("title") or "") + " " + (top.get("content") or "")[:1200])
                top_coherence_miss = (not active_coherence) or (not any(cr.search(top_combined) for cr in active_coherence))
                top_component_query = bool(_TREATMENT_COMPONENT_Q.search(query))
                top_actionable = top.get("content_type") in {"dosage_table", "treatment_protocol"}
                top_off_condition = ((not condition_match_re.search(top_combined))
                                     and top_coherence_miss
                                     and not (top_component_query and top_actionable))
                if top_off_condition:
                    action_query = _is_action_query(query)
                    top_score = float(top.get("score", 0.0))
                    replacement_idx: Optional[int] = None
                    for idx, cand in enumerate(hits[1:], start=1):
                        if cand.get("source") != top.get("source"):
                            continue
                        cand_combined = ((cand.get("title") or "") + " " + (cand.get("content") or "")[:1200])
                        if not condition_match_re.search(cand_combined):
                            continue
                        if active_coherence and not any(cr.search(cand_combined) for cr in active_coherence):
                            continue
                        if action_query and cand.get("content_type") not in _ACTION_PRIORITY_TYPES:
                            continue
                        if float(cand.get("score", 0.0)) < (top_score * 0.70):
                            continue
                        replacement_idx = idx
                        break
                    if replacement_idx is not None:
                        replacement = hits[replacement_idx]
                        hits = [replacement] + hits[:replacement_idx] + hits[replacement_idx + 1:]
                        top1_swapped = True
                        top1_swap_reason = "off_condition_top_replaced_by_aligned_candidate"

        # F10: Compute quality flags for observability and optional strict_mode
        quality_flags: List[str] = []
        if any(_RETRIEVAL_CORRUPTION_RE.search(h.get("content", "")[:300]) for h in hits[:3]):
            quality_flags.append("corruption_in_top3")
        if hits and hits[0].get("score", 0.0) < 0.012:
            quality_flags.append("low_confidence_all_results")
        # Flag silent contamination: confident score but top result not aligned to detected condition.
        top = hits[0] if hits else None
        if top:
            condition_match_re: Optional[re.Pattern] = None
            for _cid, q_re, c_re, _x_re in _CONDITION_VOCAB:
                if q_re.search(query):
                    condition_match_re = c_re
                    break
            if condition_match_re is not None:
                top_combined = ((top.get("title") or "") + " " + (top.get("content") or "")[:1200])
                active_coherence = [content_re for query_re, content_re in _CONDITION_COHERENCE
                                    if query_re.search(query)]
                coherence_miss = (not active_coherence) or (not any(cr.search(top_combined) for cr in active_coherence))
                component_query = bool(_TREATMENT_COMPONENT_Q.search(query))
                top_actionable = top.get("content_type") in {"dosage_table", "treatment_protocol"}
                if (not condition_match_re.search(top_combined)) and coherence_miss and not (component_query and top_actionable):
                    quality_flags.append("off_condition_top_result")
        if _is_action_query(query) and hits and not any(_is_actionable_hit(h) for h in hits[:3]):
            quality_flags.append("no_protocol_in_top3")
        # H3: surface no-hit case for strict_cds_mode — do NOT relax type filter
        if strict_cds_mode and not hits and _is_action_query(query):
            quality_flags.append("strict_cds_no_hit")

        # F10: strict_mode — zero out results when quality flags are triggered
        if strict_mode and quality_flags:
            hits = []
            hit = None

        # Ensure .hit is always consistent with .hits ordering.
        hit = hits[0] if hits else None

        latency_ms = (time.time() - start) * 1000.0
        return {
            "query": query,
            "hit": hit,
            "source_used": hit["source"] if hit else "none",
            "hits": hits,
            "latency_ms": latency_ms,
            "errors": errors,
            "quality_flags": quality_flags,
            "top1_swapped": top1_swapped,
            "top1_swap_reason": top1_swap_reason,
        }

    def stats(self) -> Dict[str, Any]:
        """Expose KB stats for diagnostics."""
        self.initialize()
        out = {"who_loaded": self._who_mem is not None}
        if self._who_mem is not None:
            try:
                out["who_stats"] = self._who_mem.stats()
            except Exception:
                out["who_stats"] = {}
        return out
