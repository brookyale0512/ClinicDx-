"""Manifest generation for Voice Scribe.

Queries the live OpenMRS instance to build a stripped-down concept manifest
for a given encounter type. The manifest is injected into the model context
at inference time so the model selects only from what's available.
"""

import json
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OPENMRS_BASE = os.environ.get("OPENMRS_URL", "http://localhost:8080/openmrs")
OPENMRS_USER = os.environ.get("OPENMRS_USER", "admin")
OPENMRS_PASS = os.environ.get("OPENMRS_PASSWORD", "Admin123")

# Maps encounter type UUID → list of CIEL concept codes to include in manifest.
# Derived from the live OpenMRS instance observation data.
ENCOUNTER_CONCEPT_MAP: dict[str, list[str]] = {
    # Vitals encounter
    "67a71486-1a54-468f-ac3e-7091a9a79584": [
        "5088", "5085", "5086", "5087", "5092", "5089", "5090", "5242"
    ],
    # Visit Note / Consultation
    "d7151f82-c1f3-4152-a605-2f9ea7414a79": [
        "5088", "5085", "5086", "5087", "5092", "5089", "5090", "5242",
        "21", "1006", "678", "679",
    ],
    "dd528487-82a5-4082-9c72-ed246bd49591": [
        "5088", "5085", "5086", "5087", "5092", "5089", "5090", "5242",
        "21", "1006", "678", "679",
    ],
    # Lab Results
    "3596fafb-6f6f-4396-8c87-6e63a0f1bd71": [
        "21", "1006", "678", "679", "1015", "1336", "1338", "1339", "1340", "1341",
        "160913", "160232", "300", "5475",
    ],
    # Adult Visit
    "0e8230ce-bd1d-43f5-a863-cf44344fa4b0": [
        "5088", "5085", "5086", "5087", "5092", "5089", "5090", "5242",
        "21", "1006",
    ],
}

# Fallback concept set for unknown encounter types
DEFAULT_CONCEPT_CODES = [
    "5088", "5085", "5086", "5087", "5092", "5089", "5090", "5242",
]

# FHIR resource type per CIEL concept category
CONCEPT_FHIR_TYPE = {
    "exam": "Observation",
    "laboratory": "Observation",
    "condition": "Condition",
}

# Labels must match training data exactly (clips.jsonl manifest_line field).
# The model was trained on these exact strings; any deviation breaks extraction.
CIEL_LABELS: dict[str, dict] = {
    "5088": {"label": "temperature_c", "manifest_line": "temperature_c (number, C)", "unit": "C", "fhir_type": "Observation", "category": "exam", "value_type": "Quantity"},
    "5085": {"label": "systolic_blood_pressure", "manifest_line": "systolic_blood_pressure (number, mmHg)", "unit": "mmHg", "fhir_type": "Observation", "category": "exam", "value_type": "Quantity"},
    "5086": {"label": "diastolic_blood_pressure", "manifest_line": "diastolic_blood_pressure (number, mmHg)", "unit": "mmHg", "fhir_type": "Observation", "category": "exam", "value_type": "Quantity"},
    "5087": {"label": "pulse", "manifest_line": "pulse (number, beats/min)", "unit": "beats/min", "fhir_type": "Observation", "category": "exam", "value_type": "Quantity"},
    "5092": {"label": "arterial_blood_oxygen_saturation_pulse_oximeter", "manifest_line": "arterial_blood_oxygen_saturation_pulse_oximeter (number)", "unit": "%", "fhir_type": "Observation", "category": "exam", "value_type": "Quantity"},
    "5089": {"label": "weight_kg", "manifest_line": "weight_kg (number, kg)", "unit": "kg", "fhir_type": "Observation", "category": "exam", "value_type": "Quantity"},
    "5090": {"label": "height_cm", "manifest_line": "height_cm (number, cm)", "unit": "cm", "fhir_type": "Observation", "category": "exam", "value_type": "Quantity"},
    "5242": {"label": "respiratory_rate", "manifest_line": "respiratory_rate (number)", "unit": "breaths/min", "fhir_type": "Observation", "category": "exam", "value_type": "Quantity"},
    "21":   {"label": "haemoglobin", "manifest_line": "[test] haemoglobin", "unit": "g/dL", "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "1006": {"label": "total_cholesterol_mmol_l", "manifest_line": "[test] total_cholesterol_mmol_l", "unit": "mmol/L", "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "678":  {"label": "white_blood_cells", "manifest_line": "[test] white_blood_cells", "unit": "10^3/uL", "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "679":  {"label": "red_blood_cells", "manifest_line": "[test] red_blood_cells", "unit": "10^6/uL", "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "1015": {"label": "hematocrit", "manifest_line": "[test] hematocrit", "unit": "%", "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "1336": {"label": "neutrophils_microscopic_exam", "manifest_line": "[test] neutrophils_microscopic_exam", "unit": "%", "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "1338": {"label": "lymphocytes_microscopic_exam", "manifest_line": "[test] lymphocytes_microscopic_exam", "unit": "%", "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "1339": {"label": "monocytes_microscopic_exam", "manifest_line": "[test] monocytes_microscopic_exam", "unit": "%", "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "1340": {"label": "eosinophils_microscopic_exam", "manifest_line": "[test] eosinophils_microscopic_exam", "unit": "%", "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "1341": {"label": "basophils_microscopic_exam", "manifest_line": "[test] basophils_microscopic_exam", "unit": "%", "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "160913": {"label": "prostate_specific_antigen_psa_measurement_ng_ml", "manifest_line": "[test] prostate_specific_antigen_psa_measurement_ng_ml", "unit": "ng/mL", "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "160232": {"label": "rhesus_blood_grouping_test", "manifest_line": "[test] rhesus_blood_grouping_test", "unit": None, "fhir_type": "Observation", "category": "laboratory", "value_type": "CodeableConcept"},
    "300":  {"label": "blood_typing", "manifest_line": "[test] blood_typing", "unit": None, "fhir_type": "Observation", "category": "laboratory", "value_type": "CodeableConcept"},
    "5475": {"label": "tuberculin_skin_test_mm", "manifest_line": "[test] tuberculin_skin_test_mm", "unit": "mm", "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
}


class ManifestBuilder:
    """Builds concept manifests from the live OpenMRS instance."""

    def __init__(self):
        self._auth = (OPENMRS_USER, OPENMRS_PASS)
        self._concept_cache: dict[str, dict] = {}

    async def get_encounter_context(self, encounter_uuid: str) -> dict:
        """Fetch encounter metadata: patient_uuid, encounter_type_uuid, location."""
        url = f"{OPENMRS_BASE}/ws/rest/v1/encounter/{encounter_uuid}"
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, verify=False) as client:
            resp = await client.get(url, auth=self._auth)
            resp.raise_for_status()
            data = resp.json()
            return {
                "encounter_uuid": encounter_uuid,
                "patient_uuid": data.get("patient", {}).get("uuid"),
                "encounter_type_uuid": data.get("encounterType", {}).get("uuid"),
                "encounter_type_name": data.get("encounterType", {}).get("display"),
                "location_uuid": data.get("location", {}).get("uuid"),
                "location_name": data.get("location", {}).get("display"),
                "provider_uuid": (
                    data.get("encounterProviders", [{}])[0]
                    .get("provider", {})
                    .get("uuid")
                    if data.get("encounterProviders")
                    else None
                ),
            }

    async def resolve_concept_uuid(self, ciel_code: str) -> Optional[str]:
        """Resolve a CIEL code to a local OpenMRS concept UUID."""
        if ciel_code in self._concept_cache:
            return self._concept_cache[ciel_code].get("uuid")

        url = f"{OPENMRS_BASE}/ws/rest/v1/concept?source=CIEL&code={ciel_code}"
        try:
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True, verify=False) as client:
                resp = await client.get(url, auth=self._auth)
                resp.raise_for_status()
                results = resp.json().get("results", [])
                if results:
                    uuid = results[0].get("uuid")
                    self._concept_cache[ciel_code] = {"uuid": uuid}
                    return uuid
        except Exception as e:
            logger.warning("Could not resolve CIEL:%s → %s", ciel_code, e)
        return None

    async def build_manifest(self, encounter_uuid: str) -> dict:
        """Build the full manifest for an encounter.

        Returns:
            {
                "context": { encounter/patient/location metadata },
                "manifest_string": "AVAILABLE:\ntemperature (number, celsius)\n...",
                "lookup": { "temperature": { ciel, uuid, fhir_type, unit, ... } }
            }
        """
        context = await self.get_encounter_context(encounter_uuid)
        encounter_type_uuid = context.get("encounter_type_uuid", "")

        ciel_codes = ENCOUNTER_CONCEPT_MAP.get(encounter_type_uuid, DEFAULT_CONCEPT_CODES)

        manifest_lines: list[str] = ["CONCEPTS:"]
        lookup: dict[str, dict] = {}

        for code in ciel_codes:
            meta = CIEL_LABELS.get(code)
            if not meta:
                continue

            label = meta["label"]
            manifest_line = meta.get("manifest_line", label)
            unit = meta["unit"]
            value_type = meta["value_type"]
            fhir_type = meta["fhir_type"]
            category = meta["category"]

            local_uuid = await self.resolve_concept_uuid(code)
            ciel_uuid_full = f"{code}AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"[:36]

            manifest_lines.append(manifest_line)

            lookup[label] = {
                "ciel_code": code,
                "ciel_uuid_full": ciel_uuid_full,
                "local_uuid": local_uuid or ciel_uuid_full,
                "fhir_type": fhir_type,
                "category": category,
                "unit": unit,
                "value_type": value_type,
                "display_name": label.replace("_", " ").title(),
            }

        return {
            "context": context,
            "manifest_string": "\n".join(manifest_lines),
            "lookup": lookup,
        }


# Singleton
_builder: Optional[ManifestBuilder] = None


def get_builder() -> ManifestBuilder:
    global _builder
    if _builder is None:
        _builder = ManifestBuilder()
    return _builder
