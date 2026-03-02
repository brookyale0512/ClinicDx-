"""FHIR resource builder for Voice Scribe.

Takes parsed model output (label: value) + manifest lookup
and constructs ready-to-POST FHIR R4 resources.
"""

import re
from datetime import datetime, timezone
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def build_observation(
    concept_meta: dict,
    value: str,
    patient_uuid: str,
    encounter_uuid: str,
) -> dict:
    """Build a FHIR R4 Observation resource."""
    timestamp = _now_iso()
    local_uuid = concept_meta["local_uuid"]
    ciel_code = concept_meta["ciel_code"]
    unit = concept_meta.get("unit")
    category = concept_meta.get("category", "exam")
    value_type = concept_meta.get("value_type", "Quantity")

    resource: dict[str, Any] = {
        "resourceType": "Observation",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": category,
                        "display": category.title(),
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {"code": local_uuid, "display": concept_meta.get("display_name", "")},
                {"system": "https://cielterminology.org", "code": ciel_code},
            ],
            "text": concept_meta.get("display_name", ""),
        },
        "subject": {"reference": f"Patient/{patient_uuid}", "type": "Patient"},
        "encounter": {"reference": f"Encounter/{encounter_uuid}", "type": "Encounter"},
        "effectiveDateTime": timestamp,
    }

    # Attach value
    if value_type == "Quantity":
        cleaned = re.sub(r"[^\d.\-]", "", value)
        if not cleaned or cleaned in (".", "-", "-."):
            resource["valueString"] = value
            return resource
        try:
            numeric = float(cleaned)
            resource["valueQuantity"] = {
                "value": numeric,
                "unit": unit or "",
            }
        except ValueError:
            resource["valueString"] = value
            return resource
    elif value_type == "CodeableConcept":
        resource["valueCodeableConcept"] = {
            "coding": [{"display": value}],
            "text": value,
        }
    else:
        resource["valueString"] = value

    return resource


def build_condition(
    concept_meta: dict,
    value: str,
    patient_uuid: str,
    encounter_uuid: str,
) -> dict:
    """Build a FHIR R4 Condition resource."""
    timestamp = _now_iso()
    local_uuid = concept_meta["local_uuid"]
    ciel_code = concept_meta["ciel_code"]

    # Map value to FHIR verification status
    verification_map = {
        "confirmed": "confirmed",
        "unconfirmed": "unconfirmed",
        "absent": "refuted",
        "ruled_out": "refuted",
    }
    verification = verification_map.get(value.lower().strip(), "confirmed")
    clinical_status = "inactive" if verification == "refuted" else "active"

    return {
        "resourceType": "Condition",
        "clinicalStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": clinical_status,
                }
            ]
        },
        "verificationStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": verification,
                }
            ]
        },
        "code": {
            "coding": [
                {"code": local_uuid, "display": concept_meta.get("display_name", "")},
                {"system": "https://cielterminology.org", "code": ciel_code},
            ],
            "text": concept_meta.get("display_name", ""),
        },
        "subject": {"reference": f"Patient/{patient_uuid}", "type": "Patient"},
        "encounter": {"reference": f"Encounter/{encounter_uuid}", "type": "Encounter"},
        "recordedDate": timestamp,
    }


def build_fhir_payload(
    label: str,
    value: str,
    concept_meta: dict,
    patient_uuid: str,
    encounter_uuid: str,
) -> Optional[dict]:
    """Route to the correct FHIR builder based on concept type."""
    fhir_type = concept_meta.get("fhir_type", "Observation")

    if fhir_type == "Observation":
        return build_observation(concept_meta, value, patient_uuid, encounter_uuid)
    elif fhir_type == "Condition":
        return build_condition(concept_meta, value, patient_uuid, encounter_uuid)

    return None


def human_readable(label: str, value: str, concept_meta: dict) -> str:
    """Generate a human-readable string for UI display."""
    display = concept_meta.get("display_name", label.replace("_", " ").title())
    unit = concept_meta.get("unit", "")
    fhir_type = concept_meta.get("fhir_type", "Observation")

    if fhir_type == "Condition":
        status_map = {
            "confirmed": "Confirmed",
            "unconfirmed": "Suspected",
            "absent": "Absent / Ruled out",
        }
        status = status_map.get(value.lower(), value.title())
        return f"{display} — {status}"

    if unit:
        return f"{display} — {value} {unit}"
    return f"{display} — {value}"
