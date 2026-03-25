# =============================================================================
# TIU Inpatient Note Classifier
# Databricks Notebook — PySpark + BGE-large-en-v1.5 (on-prem, no PHI egress)
#
# Architecture:
#   Stage 1 : Regex/keyword rule engine  → confidence-weighted multi-label
#   Stage 2 : BGE cosine similarity      → fallback for low-confidence titles
#
# Output schema per note title:
#   role            : List[str]   e.g. ["NURSING"]
#   note_type       : List[str]   e.g. ["ASSESSMENT", "PLAN"]
#   classification_source : str   "RULES" | "EMBEDDING" | "RULES+EMBEDDING"
#   role_confidence       : float
#   note_type_confidence  : float
#   rule_hits             : List[str]  (audit trail — which patterns fired)
# =============================================================================

# -----------------------------------------------------------------------------
# CELL 1 — Imports & Config
# -----------------------------------------------------------------------------
import re
import json
import numpy as np
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

import torch
from sentence_transformers import SentenceTransformer

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType,
    ArrayType, MapType
)

spark = SparkSession.builder.getOrCreate()

# --- Paths & thresholds -------------------------------------------------------
BGE_MODEL_PATH   = "dbfs:/models/bge-large-en-v1.5"   # adjust to your DBFS path
MLFLOW_EXP_NAME  = "/Users/you/tiu-note-classifier"
RULE_CONF_THRESH = 0.72   # below this, fall through to embedding stage
EMB_CONF_THRESH  = 0.45   # below this, emit UNCERTAIN labels


# =============================================================================
# CELL 2 — Label Taxonomy
# =============================================================================

# ---- Role labels -------------------------------------------------------------
ROLE_LABELS = [
    "CLINICAL",        # MD / DO / NP / PA / Resident / Fellow
    "NURSING",         # RN / LPN / CNA
    "SOCIAL_WORK",     # Social worker, case manager, discharge planner
    "ALLIED_HEALTH",   # PT, OT, RT, Nutrition, Pharmacy, Audiology, etc.
    "ADMINISTRATIVE",  # HIM, MRT, transcriptionist, admin
]

# ---- Note-type labels --------------------------------------------------------
NOTE_TYPE_LABELS = [
    "ADMISSION",       # H&P, admission assessment, intake
    "PROGRESS",        # Daily / interval progress notes
    "ASSESSMENT",      # Stand-alone assessments, evaluations, screens
    "PLAN",            # Care plans, treatment plans
    "PROCEDURE",       # Procedure notes, op notes, intervention notes
    "CONSULT",         # Consult requests and responses
    "HANDOFF",         # Shift notes, sign-out, transfer notes, SBAR
    "DISCHARGE",       # DC summaries, DC instructions, DC planning notes
    "ADDENDUM",        # Addenda to any parent note
    "ORDER",           # Verbal order confirmations, telephone orders
    "EDUCATION",       # Patient/family education
    "ADMINISTRATIVE",  # Consent, legal, HIM, coding
]


# =============================================================================
# CELL 3 — Stage 1: Rule Engine
# =============================================================================
#
# Each rule is a dict:
#   pattern  : compiled regex (applied to uppercased title)
#   role     : List[str] | None
#   note_type: List[str] | None
#   weight   : float  — contribution to confidence when this rule fires
#              Multiple rules can fire; confidence = min(1.0, sum(weights))
#   tag      : str    — human-readable audit label
# -----------------------------------------------------------------------------

def _r(pattern: str) -> re.Pattern:
    """Compile case-insensitive regex."""
    return re.compile(pattern, re.IGNORECASE)


RULES = [

    # =========================================================================
    # ROLE — CLINICAL
    # =========================================================================
    {"pattern": _r(r"\b(PHYSICIAN|ATTENDING|RESIDENT|FELLOW|INTERN)\b"),
     "role": ["CLINICAL"], "note_type": None, "weight": 0.80,
     "tag": "clinical:title_physician"},

    {"pattern": _r(r"\b(MD|DO|NP|PA|ARNP|CRNA)\b"),
     "role": ["CLINICAL"], "note_type": None, "weight": 0.75,
     "tag": "clinical:credential_suffix"},

    {"pattern": _r(r"\bMEDICINE\b(?!.*MEDICATION)"),
     "role": ["CLINICAL"], "note_type": None, "weight": 0.55,
     "tag": "clinical:medicine_service"},

    {"pattern": _r(r"\b(SURGERY|SURGICAL|OPERATIVE|ANESTHESIA|ANESTHESIOLOGY)\b"),
     "role": ["CLINICAL"], "note_type": None, "weight": 0.75,
     "tag": "clinical:surgery"},

    {"pattern": _r(r"\b(PSYCHIATRY|PSYCHIATRIC|PSYCH(?!OLOGY))\b"),
     "role": ["CLINICAL"], "note_type": None, "weight": 0.65,
     "tag": "clinical:psychiatry"},

    {"pattern": _r(r"\b(CARDIOLOGY|PULMONOLOGY|NEPHROLOGY|NEUROLOGY|ONCOLOGY"
                   r"|HEMATOLOGY|GASTROENTEROLOGY|GI|ENDOCRINOLOGY|RHEUMATOLOGY"
                   r"|DERMATOLOGY|UROLOGY|ORTHOPEDIC|ORTHOPAEDIC|OPHTHALMOLOGY"
                   r"|OTOLARYNGOLOGY|ENT|VASCULAR|THORACIC|TRAUMA)\b"),
     "role": ["CLINICAL"], "note_type": None, "weight": 0.80,
     "tag": "clinical:specialty_service"},

    {"pattern": _r(r"\b(PALLIATIVE|HOSPICE|PAIN MANAGEMENT|PAIN CLINIC)\b"),
     "role": ["CLINICAL"], "note_type": None, "weight": 0.70,
     "tag": "clinical:palliative"},

    {"pattern": _r(r"\bICU\b|\bCCU\b|\bMICU\b|\bSICU\b|\bNICU\b|\bPICU\b"),
     "role": ["CLINICAL"], "note_type": ["PROGRESS"], "weight": 0.60,
     "tag": "clinical:icu_unit"},

    {"pattern": _r(r"\b(HISTORY\s*(AND|&)\s*PHYSICAL|H\s*&\s*P|H\/P)\b"),
     "role": ["CLINICAL"], "note_type": ["ADMISSION"], "weight": 0.90,
     "tag": "clinical:hp"},

    # =========================================================================
    # ROLE — NURSING
    # =========================================================================
    {"pattern": _r(r"\bNURSING\b"),
     "role": ["NURSING"], "note_type": None, "weight": 0.90,
     "tag": "nursing:keyword_nursing"},

    {"pattern": _r(r"\b(RN|LPN|CNA|LVN)\b"),
     "role": ["NURSING"], "note_type": None, "weight": 0.80,
     "tag": "nursing:credential"},

    {"pattern": _r(r"\b(NURSE|NURSES)\b"),
     "role": ["NURSING"], "note_type": None, "weight": 0.85,
     "tag": "nursing:keyword_nurse"},

    {"pattern": _r(r"\b(FLOW\s*SHEET|FLOWSHEET|MAR|MEDICATION\s*ADMINISTRATION)\b"),
     "role": ["NURSING"], "note_type": ["PROGRESS"], "weight": 0.70,
     "tag": "nursing:flowsheet"},

    {"pattern": _r(r"\b(SHIFT\s*(NOTE|SUMMARY|REPORT)|END\s*OF\s*SHIFT)\b"),
     "role": ["NURSING"], "note_type": ["HANDOFF"], "weight": 0.85,
     "tag": "nursing:shift_note"},

    {"pattern": _r(r"\b(FALL\s*RISK|FALL\s*PREVENTION|BRADEN|MORSE)\b"),
     "role": ["NURSING"], "note_type": ["ASSESSMENT"], "weight": 0.75,
     "tag": "nursing:fall_risk"},

    {"pattern": _r(r"\b(WOUND\s*CARE|WOUND\s*ASSESSMENT|OSTOMY|SKIN\s*CARE"
                   r"|PRESSURE\s*(ULCER|INJURY|WOUND))\b"),
     "role": ["NURSING"], "note_type": ["ASSESSMENT"], "weight": 0.70,
     "tag": "nursing:wound"},

    {"pattern": _r(r"\b(IV|INTRAVENOUS)\s*(ACCESS|LINE|CARE|NOTE|ASSESSMENT)\b"),
     "role": ["NURSING"], "note_type": ["PROCEDURE"], "weight": 0.72,
     "tag": "nursing:iv_access"},

    # =========================================================================
    # ROLE — SOCIAL WORK / CARE MANAGEMENT
    # =========================================================================
    {"pattern": _r(r"\b(SOCIAL\s*WORK(ER)?|MSW|LCSW|LMSW)\b"),
     "role": ["SOCIAL_WORK"], "note_type": None, "weight": 0.90,
     "tag": "sw:keyword_sw"},

    {"pattern": _r(r"\b(CASE\s*MANAG(ER|EMENT)|CARE\s*MANAG(ER|EMENT))\b"),
     "role": ["SOCIAL_WORK"], "note_type": None, "weight": 0.85,
     "tag": "sw:case_manager"},

    {"pattern": _r(r"\b(DISCHARGE\s*PLAN(NING)?|DC\s*PLAN(NING)?)\b"),
     "role": ["SOCIAL_WORK"], "note_type": ["DISCHARGE", "PLAN"], "weight": 0.75,
     "tag": "sw:dc_planning"},

    {"pattern": _r(r"\b(HOMELESSNESS|HOUSING\s*INSTAB|HOUSING\s*ASSESS)\b"),
     "role": ["SOCIAL_WORK"], "note_type": ["ASSESSMENT"], "weight": 0.80,
     "tag": "sw:housing"},

    {"pattern": _r(r"\b(PSYCHOSOCIAL|SOCIAL\s*HISTORY|SOCIAL\s*ASSESSMENT)\b"),
     "role": ["SOCIAL_WORK"], "note_type": ["ASSESSMENT"], "weight": 0.78,
     "tag": "sw:psychosocial"},

    {"pattern": _r(r"\b(ADVANCE\s*DIRECTIVE|POWER\s*OF\s*ATTORNEY|HEALTHCARE\s*PROXY"
                   r"|SURROGATE|GUARDIAN)\b"),
     "role": ["SOCIAL_WORK", "ADMINISTRATIVE"], "note_type": ["ADMINISTRATIVE"], "weight": 0.70,
     "tag": "sw:advance_directive"},

    {"pattern": _r(r"\b(SUBSTANCE\s*ABUSE|ALCOHOL|ADDICTION|SATP|BSAS|DAST|AUDIT\b)\b"),
     "role": ["SOCIAL_WORK"], "note_type": ["ASSESSMENT"], "weight": 0.65,
     "tag": "sw:substance"},

    # =========================================================================
    # ROLE — ALLIED HEALTH
    # =========================================================================

    # Physical Therapy
    {"pattern": _r(r"\b(PHYSICAL\s*THERAP(Y|IST)|PT\s*EVAL|PT\s*NOTE|P\.T\.)\b"),
     "role": ["ALLIED_HEALTH"], "note_type": None, "weight": 0.90,
     "tag": "allied:pt"},

    # Occupational Therapy
    {"pattern": _r(r"\b(OCCUPATIONAL\s*THERAP(Y|IST)|OT\s*EVAL|OT\s*NOTE|O\.T\.)\b"),
     "role": ["ALLIED_HEALTH"], "note_type": None, "weight": 0.90,
     "tag": "allied:ot"},

    # Respiratory Therapy
    {"pattern": _r(r"\b(RESPIRATORY\s*THERAP(Y|IST)|RT\s*NOTE|RESP\s*CARE"
                   r"|VENTILATOR\s*NOTE|VENT\s*MANAGEMENT)\b"),
     "role": ["ALLIED_HEALTH"], "note_type": None, "weight": 0.88,
     "tag": "allied:rt"},

    # Nutrition / Dietetics
    {"pattern": _r(r"\b(NUTRITION|DIETITIAN|DIETETICS|DIETARY|RD\s*NOTE"
                   r"|NUTRITIONAL\s*ASSESS)\b"),
     "role": ["ALLIED_HEALTH"], "note_type": None, "weight": 0.88,
     "tag": "allied:nutrition"},

    # Pharmacy
    {"pattern": _r(r"\b(PHARMAC(Y|IST)|PHARM\s*NOTE|MEDICATION\s*RECONCIL"
                   r"|MED\s*REC|CLINICAL\s*PHARM)\b"),
     "role": ["ALLIED_HEALTH"], "note_type": None, "weight": 0.85,
     "tag": "allied:pharmacy"},

    # Speech / Language
    {"pattern": _r(r"\b(SPEECH|LANGUAGE\s*PATH|SLP|DYSPHAGIA|SWALLOW(ING)?)\b"),
     "role": ["ALLIED_HEALTH"], "note_type": None, "weight": 0.88,
     "tag": "allied:slp"},

    # Audiology
    {"pattern": _r(r"\b(AUDIOLOGY|AUDIOLOGIST|HEARING\s*(TEST|EVAL|ASSESS))\b"),
     "role": ["ALLIED_HEALTH"], "note_type": None, "weight": 0.88,
     "tag": "allied:audiology"},

    # Chaplain / Spiritual Care
    {"pattern": _r(r"\b(CHAPLAIN|SPIRITUAL\s*CARE|PASTORAL\s*CARE)\b"),
     "role": ["ALLIED_HEALTH"], "note_type": ["ASSESSMENT"], "weight": 0.85,
     "tag": "allied:chaplain"},

    # Kinesiotherapy / Recreation Therapy
    {"pattern": _r(r"\b(KINESIOTHERAPY|KINESIO|RECREATION\s*THERAP|RECREATIONAL\s*THERAP)\b"),
     "role": ["ALLIED_HEALTH"], "note_type": None, "weight": 0.88,
     "tag": "allied:kinesiotherapy"},

    # =========================================================================
    # ROLE — ADMINISTRATIVE / HIM
    # =========================================================================
    {"pattern": _r(r"\b(HIM|HEALTH\s*INFORMATION|MEDICAL\s*RECORD(S)?|MRT)\b"),
     "role": ["ADMINISTRATIVE"], "note_type": None, "weight": 0.85,
     "tag": "admin:him"},

    {"pattern": _r(r"\b(TRANSCRIPTION|TRANSCRIBED|DICTATED\s*BY)\b"),
     "role": ["ADMINISTRATIVE"], "note_type": None, "weight": 0.75,
     "tag": "admin:transcription"},

    {"pattern": _r(r"\b(CONSENT\s*FORM|INFORMED\s*CONSENT)\b"),
     "role": ["ADMINISTRATIVE"], "note_type": ["ADMINISTRATIVE"], "weight": 0.80,
     "tag": "admin:consent"},

    {"pattern": _r(r"\b(CODING|ICD|CPT|DRG|BILLING)\b"),
     "role": ["ADMINISTRATIVE"], "note_type": ["ADMINISTRATIVE"], "weight": 0.85,
     "tag": "admin:coding"},

    # =========================================================================
    # NOTE TYPE — ADMISSION
    # =========================================================================
    {"pattern": _r(r"\b(ADMISSION|ADMIT|ADMITTING|INTAKE)\b"),
     "role": None, "note_type": ["ADMISSION"], "weight": 0.85,
     "tag": "type:admission"},

    {"pattern": _r(r"\b(HISTORY\s*(AND|&)\s*PHYSICAL|H\s*&\s*P)\b"),
     "role": None, "note_type": ["ADMISSION"], "weight": 0.90,
     "tag": "type:hp"},

    {"pattern": _r(r"\bINITIAL\s*(EVAL|ASSESSMENT|NOTE|CONSULT)\b"),
     "role": None, "note_type": ["ADMISSION"], "weight": 0.70,
     "tag": "type:initial_eval"},

    # =========================================================================
    # NOTE TYPE — PROGRESS
    # =========================================================================
    {"pattern": _r(r"\b(PROGRESS\s*NOTE|PN|DAILY\s*NOTE|INTERVAL\s*NOTE"
                   r"|DAILY\s*PROGRESS)\b"),
     "role": None, "note_type": ["PROGRESS"], "weight": 0.90,
     "tag": "type:progress"},

    {"pattern": _r(r"\bSOAP\b"),
     "role": None, "note_type": ["PROGRESS"], "weight": 0.85,
     "tag": "type:soap"},

    # =========================================================================
    # NOTE TYPE — ASSESSMENT
    # =========================================================================
    {"pattern": _r(r"\b(ASSESSMENT|EVAL(UATION)?|SCREEN(ING)?)\b"),
     "role": None, "note_type": ["ASSESSMENT"], "weight": 0.70,
     "tag": "type:assessment"},

    {"pattern": _r(r"\b(PHQS?[\s-]?\d|GAD[\s-]?\d|MMSE|MOCA|CAGE|AUDIT|CIWA"
                   r"|CPRS\s*SCREEN|MINI[\s-]?MENTAL)\b"),
     "role": None, "note_type": ["ASSESSMENT"], "weight": 0.90,
     "tag": "type:standardized_screen"},

    {"pattern": _r(r"\b(RISK\s*ASSESS|SAFETY\s*ASSESS|SUICIDE\s*RISK|VIOLENCE\s*RISK)\b"),
     "role": None, "note_type": ["ASSESSMENT"], "weight": 0.88,
     "tag": "type:risk_assessment"},

    # =========================================================================
    # NOTE TYPE — PLAN
    # =========================================================================
    {"pattern": _r(r"\b(CARE\s*PLAN|TREATMENT\s*PLAN|PLAN\s*OF\s*CARE"
                   r"|INTERDISCIPLINARY\s*PLAN|IDT\s*PLAN)\b"),
     "role": None, "note_type": ["PLAN"], "weight": 0.90,
     "tag": "type:care_plan"},

    # =========================================================================
    # NOTE TYPE — PROCEDURE
    # =========================================================================
    {"pattern": _r(r"\b(PROCEDURE|PROCEDURAL|OPERATIVE|OPERATION|OP\s*NOTE"
                   r"|OPERATIVE\s*REPORT)\b"),
     "role": None, "note_type": ["PROCEDURE"], "weight": 0.88,
     "tag": "type:procedure"},

    {"pattern": _r(r"\b(BRONCHOSCOPY|ENDOSCOPY|COLONOSCOPY|BIOPSY|PARACENTESIS"
                   r"|THORACENTESIS|LUMBAR\s*PUNCTURE|LP\b|INTUBATION|LINE\s*PLACE"
                   r"|CENTRAL\s*LINE|ARTERIAL\s*LINE|CATHETER(IZATION)?)\b"),
     "role": None, "note_type": ["PROCEDURE"], "weight": 0.90,
     "tag": "type:named_procedure"},

    # =========================================================================
    # NOTE TYPE — CONSULT
    # =========================================================================
    {"pattern": _r(r"\b(CONSULT(ATION)?|CONSULT\s*REQUEST|CONSULT\s*RESPONSE"
                   r"|CONSULT\s*NOTE|CONSULTATION\s*NOTE)\b"),
     "role": None, "note_type": ["CONSULT"], "weight": 0.90,
     "tag": "type:consult"},

    {"pattern": _r(r"\b(REQUEST\s*FOR\s*CONSULT|REFERRAL)\b"),
     "role": None, "note_type": ["CONSULT"], "weight": 0.80,
     "tag": "type:referral"},

    # =========================================================================
    # NOTE TYPE — HANDOFF
    # =========================================================================
    {"pattern": _r(r"\b(HANDOFF|HAND[\s-]OFF|SIGN[\s-]OUT|SIGNOUT|SBAR"
                   r"|TRANSFER\s*NOTE|TRANSFER\s*SUMMARY)\b"),
     "role": None, "note_type": ["HANDOFF"], "weight": 0.90,
     "tag": "type:handoff"},

    {"pattern": _r(r"\b(SHIFT\s*(NOTE|REPORT|SUMMARY)|END[\s-]OF[\s-]SHIFT)\b"),
     "role": None, "note_type": ["HANDOFF"], "weight": 0.88,
     "tag": "type:shift"},

    # =========================================================================
    # NOTE TYPE — DISCHARGE
    # =========================================================================
    {"pattern": _r(r"\b(DISCHARGE\s*(SUMMARY|SUMM|NOTE|INSTRUCTION"
                   r"|PLANNING|PLAN|DISPOSITION)|DC\s*SUMM(ARY)?)\b"),
     "role": None, "note_type": ["DISCHARGE"], "weight": 0.92,
     "tag": "type:discharge"},

    {"pattern": _r(r"\b(AFTER\s*VISIT\s*SUMMARY|AVS)\b"),
     "role": None, "note_type": ["DISCHARGE", "EDUCATION"], "weight": 0.85,
     "tag": "type:avs"},

    # =========================================================================
    # NOTE TYPE — ADDENDUM
    # =========================================================================
    {"pattern": _r(r"\b(ADDENDUM|ADDENDA|AMENDMENT|AMENDED\s*NOTE)\b"),
     "role": None, "note_type": ["ADDENDUM"], "weight": 0.95,
     "tag": "type:addendum"},

    {"pattern": _r(r"\b(LATE\s*ENTRY|DELAYED\s*ENTRY|CORRECTION)\b"),
     "role": None, "note_type": ["ADDENDUM"], "weight": 0.85,
     "tag": "type:late_entry"},

    # =========================================================================
    # NOTE TYPE — ORDER
    # =========================================================================
    {"pattern": _r(r"\b(VERBAL\s*ORDER|TELEPHONE\s*ORDER|T\.O\.|V\.O\.|READ[\s-]BACK)\b"),
     "role": None, "note_type": ["ORDER"], "weight": 0.92,
     "tag": "type:verbal_order"},

    # =========================================================================
    # NOTE TYPE — EDUCATION
    # =========================================================================
    {"pattern": _r(r"\b(PATIENT\s*EDUCATION|FAMILY\s*EDUCATION|TEACHING\s*NOTE"
                   r"|DISCHARGE\s*INSTRUCTION|HOME\s*CARE\s*INSTRUCTION)\b"),
     "role": None, "note_type": ["EDUCATION"], "weight": 0.88,
     "tag": "type:education"},

    # =========================================================================
    # NOTE TYPE — ADMINISTRATIVE
    # =========================================================================
    {"pattern": _r(r"\b(CONSENT|AUTHORIZATION|RELEASE\s*OF\s*INFORMATION"
                   r"|ROI|HIPAA)\b"),
     "role": None, "note_type": ["ADMINISTRATIVE"], "weight": 0.85,
     "tag": "type:admin_consent"},

    {"pattern": _r(r"\b(INCIDENT\s*REPORT|OCCURRENCE\s*REPORT|UNUSUAL\s*INCIDENT)\b"),
     "role": None, "note_type": ["ADMINISTRATIVE"], "weight": 0.88,
     "tag": "type:incident"},

]


def apply_rules(title: str) -> dict:
    """
    Apply all rules to a single title string.

    Returns:
        {
            "roles"           : List[str],
            "note_types"      : List[str],
            "role_confidence" : float,
            "type_confidence" : float,
            "rule_hits"       : List[str],
        }
    """
    title_upper = title.upper()

    role_weights   = {}   # role_label -> cumulative weight
    type_weights   = {}   # type_label -> cumulative weight
    hits           = []

    for rule in RULES:
        if rule["pattern"].search(title_upper):
            hits.append(rule["tag"])
            if rule["role"]:
                for r in rule["role"]:
                    role_weights[r] = role_weights.get(r, 0.0) + rule["weight"]
            if rule["note_type"]:
                for t in rule["note_type"]:
                    type_weights[t] = type_weights.get(t, 0.0) + rule["weight"]

    # Confidence = highest single-label weight, clipped to [0, 1]
    role_conf = min(max(role_weights.values(), default=0.0), 1.0)
    type_conf = min(max(type_weights.values(), default=0.0), 1.0)

    # Emit only labels whose weight >= 60% of the top-scoring label
    def threshold_labels(weight_dict: dict) -> list:
        if not weight_dict:
            return []
        top = max(weight_dict.values())
        return sorted(
            [k for k, v in weight_dict.items() if v >= 0.60 * top]
        )

    return {
        "roles"           : threshold_labels(role_weights),
        "note_types"      : threshold_labels(type_weights),
        "role_confidence" : round(role_conf, 4),
        "type_confidence" : round(type_conf, 4),
        "rule_hits"       : hits,
    }


# =============================================================================
# CELL 4 — Stage 2: Embedding Classifier
# =============================================================================

# ---- Label centroid library --------------------------------------------------
# Each entry is a list of natural-language anchor phrases for that label.
# The centroid is the mean embedding of all anchors.
# Add domain-specific phrasing your VA uses to improve recall.

ROLE_ANCHORS = {
    "CLINICAL": [
        "physician progress note", "attending note", "medical doctor assessment",
        "resident note", "fellow progress note", "internal medicine note",
        "surgical note", "hospitalist note", "intensivist note",
        "provider assessment and plan", "MD note", "doctor note",
        "intern progress note", "NP progress note", "PA note",
        "nurse practitioner assessment", "physician assistant note",
    ],
    "NURSING": [
        "nursing note", "RN note", "registered nurse assessment",
        "nursing assessment", "bedside nurse note", "floor nurse note",
        "nursing shift report", "nursing flowsheet", "medication administration note",
        "LPN note", "licensed practical nurse", "charge nurse note",
        "nursing care documentation", "nursing progress note",
    ],
    "SOCIAL_WORK": [
        "social work note", "social worker assessment", "MSW note",
        "case management note", "care coordination note",
        "discharge planning note", "psychosocial assessment",
        "LCSW note", "social services note", "community resource referral",
        "housing assessment", "social history note",
    ],
    "ALLIED_HEALTH": [
        "physical therapy evaluation", "PT note", "occupational therapy note",
        "OT evaluation", "respiratory therapy note", "RT note",
        "nutrition assessment", "dietitian note", "pharmacy note",
        "clinical pharmacist review", "speech language pathology note",
        "SLP swallow evaluation", "chaplain spiritual care note",
        "kinesiotherapy note", "recreation therapy note",
        "audiology evaluation", "audiologist note",
    ],
    "ADMINISTRATIVE": [
        "health information management note", "HIM note",
        "medical records technician", "transcription note",
        "administrative documentation", "coding note", "billing note",
        "consent documentation", "release of information",
    ],
}

NOTE_TYPE_ANCHORS = {
    "ADMISSION": [
        "admission note", "history and physical", "H&P note",
        "admitting assessment", "intake evaluation", "admission history",
        "initial assessment on admission", "admit note",
    ],
    "PROGRESS": [
        "daily progress note", "interval progress note", "SOAP note",
        "progress note", "daily note", "clinical progress update",
        "inpatient daily note",
    ],
    "ASSESSMENT": [
        "clinical assessment", "evaluation note", "screening note",
        "risk assessment", "suicide risk assessment", "fall risk screening",
        "mental status exam", "cognitive screening", "PHQ depression screen",
        "pain assessment", "functional assessment",
    ],
    "PLAN": [
        "care plan", "treatment plan", "plan of care",
        "interdisciplinary care plan", "individualized treatment plan",
        "clinical management plan",
    ],
    "PROCEDURE": [
        "procedure note", "operative note", "procedural documentation",
        "bronchoscopy note", "endoscopy note", "lumbar puncture note",
        "central line placement note", "bedside procedure note",
        "intervention note", "surgical procedure note",
    ],
    "CONSULT": [
        "consultation note", "consult request", "consult response",
        "specialist consultation", "referral note", "consult documentation",
        "consulting service note",
    ],
    "HANDOFF": [
        "shift handoff note", "sign out note", "transfer note",
        "SBAR communication", "end of shift summary",
        "nursing shift report", "hand off communication",
        "patient transfer summary",
    ],
    "DISCHARGE": [
        "discharge summary", "discharge note", "DC summary",
        "discharge planning documentation", "discharge instructions",
        "hospital discharge note", "after visit summary",
    ],
    "ADDENDUM": [
        "addendum to note", "late entry addendum", "correction addendum",
        "amendment to clinical note", "delayed entry",
    ],
    "ORDER": [
        "verbal order confirmation", "telephone order read back",
        "verbal order documentation", "telephone order note",
    ],
    "EDUCATION": [
        "patient education note", "family education documentation",
        "discharge teaching", "patient teaching note",
        "home care instruction", "medication education note",
    ],
    "ADMINISTRATIVE": [
        "consent form documentation", "informed consent note",
        "release of information note", "HIPAA documentation",
        "incident report", "administrative note", "HIM documentation",
    ],
}


class EmbeddingClassifier:
    """
    Loads BGE-large-en-v1.5 from DBFS, encodes label centroids,
    and classifies title strings via cosine similarity.
    """

    def __init__(self, model_path: str):
        local_path = model_path.replace("dbfs:", "/dbfs")
        self.model = SentenceTransformer(local_path)
        self.role_centroids   = self._build_centroids(ROLE_ANCHORS)
        self.type_centroids   = self._build_centroids(NOTE_TYPE_ANCHORS)

    def _build_centroids(self, anchor_dict: dict) -> dict:
        centroids = {}
        for label, phrases in anchor_dict.items():
            embeddings = self.model.encode(
                phrases,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=64,
            )
            centroids[label] = embeddings.mean(axis=0)
            # Re-normalize the centroid
            norm = np.linalg.norm(centroids[label])
            if norm > 0:
                centroids[label] = centroids[label] / norm
        return centroids

    def _cosine_scores(self, title: str, centroid_dict: dict) -> dict:
        vec = self.model.encode(
            [title],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return {
            label: float(np.dot(vec, centroid))
            for label, centroid in centroid_dict.items()
        }

    def classify(
        self,
        title: str,
        top_k_role: int = 2,
        top_k_type: int = 3,
    ) -> dict:
        role_scores = self._cosine_scores(title, self.role_centroids)
        type_scores = self._cosine_scores(title, self.type_centroids)

        def top_k_above_thresh(scores, k, thresh):
            sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            results  = [(lbl, sc) for lbl, sc in sorted_s[:k] if sc >= thresh]
            return results

        role_results = top_k_above_thresh(role_scores, top_k_role, EMB_CONF_THRESH)
        type_results = top_k_above_thresh(type_scores, top_k_type, EMB_CONF_THRESH)

        return {
            "roles"           : [r[0] for r in role_results],
            "note_types"      : [t[0] for t in type_results],
            "role_confidence" : round(role_results[0][1], 4) if role_results else 0.0,
            "type_confidence" : round(type_results[0][1], 4) if type_results else 0.0,
            "role_scores"     : {k: round(v, 4) for k, v in role_scores.items()},
            "type_scores"     : {k: round(v, 4) for k, v in type_scores.items()},
        }


# =============================================================================
# CELL 5 — Hybrid Classifier
# =============================================================================

class TIUNoteClassifier:
    """
    Two-stage hybrid classifier.
      Stage 1: Rule engine  → if both role_conf and type_conf >= RULE_CONF_THRESH,
                              emit directly with source="RULES"
      Stage 2: Embedding    → fallback for any dimension that didn't clear the
                              threshold; source="EMBEDDING" or "RULES+EMBEDDING"
    """

    def __init__(self, model_path: str = BGE_MODEL_PATH):
        self.emb_classifier = EmbeddingClassifier(model_path)

    def classify(self, title: str) -> dict:
        if not title or not title.strip():
            return self._empty_result(title)

        # ---- Stage 1 --------------------------------------------------------
        rule_out = apply_rules(title)
        role_done = rule_out["role_confidence"] >= RULE_CONF_THRESH
        type_done = rule_out["type_confidence"] >= RULE_CONF_THRESH

        if role_done and type_done:
            return {
                "title"                  : title,
                "roles"                  : rule_out["roles"],
                "note_types"             : rule_out["note_types"],
                "role_confidence"        : rule_out["role_confidence"],
                "note_type_confidence"   : rule_out["type_confidence"],
                "classification_source"  : "RULES",
                "rule_hits"              : rule_out["rule_hits"],
                "embedding_role_scores"  : {},
                "embedding_type_scores"  : {},
            }

        # ---- Stage 2 --------------------------------------------------------
        emb_out = self.emb_classifier.classify(title)

        # Merge: prefer rule output for any dimension that cleared threshold
        final_roles = (
            rule_out["roles"] if role_done
            else (emb_out["roles"] or rule_out["roles"])
        )
        final_types = (
            rule_out["note_types"] if type_done
            else (emb_out["note_types"] or rule_out["note_types"])
        )

        final_role_conf = (
            rule_out["role_confidence"] if role_done
            else emb_out["role_confidence"]
        )
        final_type_conf = (
            rule_out["type_confidence"] if type_done
            else emb_out["type_confidence"]
        )

        source = (
            "RULES+EMBEDDING" if (role_done or type_done)
            else "EMBEDDING"
        )

        return {
            "title"                  : title,
            "roles"                  : final_roles,
            "note_types"             : final_types,
            "role_confidence"        : final_role_conf,
            "note_type_confidence"   : final_type_conf,
            "classification_source"  : source,
            "rule_hits"              : rule_out["rule_hits"],
            "embedding_role_scores"  : emb_out.get("role_scores", {}),
            "embedding_type_scores"  : emb_out.get("type_scores", {}),
        }

    @staticmethod
    def _empty_result(title):
        return {
            "title"                  : title,
            "roles"                  : ["UNKNOWN"],
            "note_types"             : ["UNKNOWN"],
            "role_confidence"        : 0.0,
            "note_type_confidence"   : 0.0,
            "classification_source"  : "EMPTY",
            "rule_hits"              : [],
            "embedding_role_scores"  : {},
            "embedding_type_scores"  : {},
        }


# =============================================================================
# CELL 6 — MLflow Model Wrapper & Registration
# =============================================================================

class TIUClassifierWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper so the classifier can be versioned,
    logged, and served from Databricks Model Serving.
    """

    def load_context(self, context):
        model_path = context.artifacts["bge_model_path"]
        self.classifier = TIUNoteClassifier(model_path=model_path)

    def predict(self, context, model_input):
        import pandas as pd
        titles = model_input["title"].tolist()
        results = [self.classifier.classify(t) for t in titles]
        return pd.DataFrame(results)


def log_classifier_to_mlflow(
    model_path: str = BGE_MODEL_PATH,
    experiment_name: str = MLFLOW_EXP_NAME,
):
    mlflow.set_experiment(experiment_name)

    # Quick smoke-test inputs for signature inference
    sample_titles = [
        "NURSING PROGRESS NOTE",
        "DISCHARGE SUMMARY",
        "PHYSICAL THERAPY EVALUATION",
        "SOCIAL WORK CONSULTATION",
        "PHYSICIAN H&P",
    ]

    import pandas as pd
    sample_input  = pd.DataFrame({"title": sample_titles})
    clf           = TIUNoteClassifier(model_path=model_path)
    sample_output = pd.DataFrame([clf.classify(t) for t in sample_titles])
    signature     = infer_signature(sample_input, sample_output)

    artifacts = {"bge_model_path": model_path.replace("dbfs:", "/dbfs")}

    with mlflow.start_run(run_name="tiu-note-classifier-v1"):
        mlflow.log_params({
            "rule_conf_thresh" : RULE_CONF_THRESH,
            "emb_conf_thresh"  : EMB_CONF_THRESH,
            "n_rules"          : len(RULES),
            "n_role_labels"    : len(ROLE_LABELS),
            "n_type_labels"    : len(NOTE_TYPE_LABELS),
            "bge_model"        : "bge-large-en-v1.5",
        })
        mlflow.log_dict(
            {lbl: anchors for lbl, anchors in ROLE_ANCHORS.items()},
            "role_anchors.json",
        )
        mlflow.log_dict(
            {lbl: anchors for lbl, anchors in NOTE_TYPE_ANCHORS.items()},
            "type_anchors.json",
        )

        model_info = mlflow.pyfunc.log_model(
            artifact_path   = "tiu_note_classifier",
            python_model    = TIUClassifierWrapper(),
            artifacts       = artifacts,
            signature       = signature,
            registered_model_name = "tiu_note_classifier",
        )

    print(f"Model logged: {model_info.model_uri}")
    return model_info


# =============================================================================
# CELL 7 — Spark UDF & DataFrame Classification
# =============================================================================

OUTPUT_SCHEMA = StructType([
    StructField("roles",                  ArrayType(StringType()), True),
    StructField("note_types",             ArrayType(StringType()), True),
    StructField("role_confidence",        FloatType(),             True),
    StructField("note_type_confidence",   FloatType(),             True),
    StructField("classification_source",  StringType(),            True),
    StructField("rule_hits",              ArrayType(StringType()), True),
    StructField("embedding_role_scores",  MapType(StringType(), FloatType()), True),
    StructField("embedding_type_scores",  MapType(StringType(), FloatType()), True),
])


def make_classify_udf(model_path: str = BGE_MODEL_PATH):
    """
    Returns a Spark UDF.  The classifier is instantiated once per executor
    via broadcast to avoid re-loading the model for every row.

    Usage:
        classify_udf = make_classify_udf()
        df_out = df.withColumn("classification", classify_udf(F.col("note_title")))
    """
    # Broadcast the model path string; model is loaded lazily on each executor
    model_path_bc = spark.sparkContext.broadcast(model_path)

    # We use a closure so each executor gets exactly one model instance
    _classifier_ref = {}

    def _get_classifier():
        if "clf" not in _classifier_ref:
            _classifier_ref["clf"] = TIUNoteClassifier(
                model_path=model_path_bc.value
            )
        return _classifier_ref["clf"]

    def classify_title(title: Optional[str]):
        if title is None:
            return TIUNoteClassifier._empty_result(title)
        clf = _get_classifier()
        result = clf.classify(title)
        return (
            result["roles"],
            result["note_types"],
            result["role_confidence"],
            result["note_type_confidence"],
            result["classification_source"],
            result["rule_hits"],
            result["embedding_role_scores"],
            result["embedding_type_scores"],
        )

    return F.udf(classify_title, OUTPUT_SCHEMA)


def classify_tiu_notes(
    df: DataFrame,
    title_col: str = "note_title",
    model_path: str = BGE_MODEL_PATH,
) -> DataFrame:
    """
    Main entry point.  Accepts a Spark DataFrame with a note title column,
    returns the same DataFrame with a nested 'classification' struct column
    plus convenience flat columns for role and note_type.

    Example:
        df_classified = classify_tiu_notes(df_notes, title_col="tiu_document_definition")
    """
    classify_udf = make_classify_udf(model_path)

    df_out = (
        df
        .withColumn("classification", classify_udf(F.col(title_col)))
        # Convenience flat columns for easy filtering/joins
        .withColumn("note_roles",     F.col("classification.roles"))
        .withColumn("note_types",     F.col("classification.note_types"))
        .withColumn("role_conf",      F.col("classification.role_confidence"))
        .withColumn("type_conf",      F.col("classification.note_type_confidence"))
        .withColumn("clf_source",     F.col("classification.classification_source"))
        .withColumn("rule_hits",      F.col("classification.rule_hits"))
    )

    return df_out


# =============================================================================
# CELL 8 — Quick Smoke Test (run interactively in Databricks)
# =============================================================================

def smoke_test():
    test_titles = [
        # Should hit rules cleanly
        "NURSING PROGRESS NOTE",
        "PHYSICIAN H&P",
        "PHYSICAL THERAPY EVALUATION",
        "SOCIAL WORK DISCHARGE PLANNING NOTE",
        "DISCHARGE SUMMARY",
        "RESPIRATORY THERAPY NOTE",
        "NUTRITION ASSESSMENT",
        "PHARMACY MEDICATION RECONCILIATION",
        "CONSULT NOTE - CARDIOLOGY",
        "VERBAL ORDER CONFIRMATION",
        "PATIENT EDUCATION - DIABETES",
        "ADDENDUM TO PROGRESS NOTE",
        "SURGICAL OPERATIVE REPORT",

        # Ambiguous — should fall through to embedding
        "INTERDISCIPLINARY ROUNDING NOTE",
        "CRITICAL CARE DAILY",
        "BEHAVIORAL HEALTH NOTE",
        "COMMUNITY LIVING CENTER NOTE",
        "WOMEN'S HEALTH NOTE",
        "TELEHEALTH VISIT",
        "PEER SUPPORT SPECIALIST NOTE",

        # Edge cases
        "",
        "NOTE",
        "MISC CLINICAL DOCUMENTATION",
    ]

    clf = TIUNoteClassifier(BGE_MODEL_PATH)

    print(f"{'TITLE':<45} {'ROLES':<30} {'TYPES':<40} {'SOURCE':<18} "
          f"{'R_CONF':>6} {'T_CONF':>6}")
    print("-" * 155)

    for title in test_titles:
        r = clf.classify(title)
        print(
            f"{r['title'][:44]:<45} "
            f"{str(r['roles']):<30} "
            f"{str(r['note_types']):<40} "
            f"{r['classification_source']:<18} "
            f"{r['role_confidence']:>6.3f} "
            f"{r['note_type_confidence']:>6.3f}"
        )


# smoke_test()   # Uncomment in notebook to run

# =============================================================================
# CELL 9 — Example end-to-end Databricks pipeline
# =============================================================================

# ---- Uncomment and adapt to your actual table names -------------------------
#
# df_tiu = spark.table("cdw.tiu_document") \
#     .filter(
#         F.col("status").isin(["COMPLETED", "AMENDED", "UNCOSIGNED"]) &
#         F.col("patient_class").isin(["INPATIENT"])
#     ) \
#     .select("sta3n", "tiu_document_ien", "document_definition_name",
#             "reference_date", "status", "author_name")
#
# df_classified = classify_tiu_notes(
#     df_tiu,
#     title_col = "document_definition_name",
# )
#
# # Write to Delta with Z-ordering on role/type for fast downstream queries
# (
#     df_classified
#     .write
#     .format("delta")
#     .mode("overwrite")
#     .option("overwriteSchema", "true")
#     .saveAsTable("cdw.tiu_document_classified")
# )
#
# spark.sql("""
#     OPTIMIZE cdw.tiu_document_classified
#     ZORDER BY (note_roles, note_types, clf_source)
# """)