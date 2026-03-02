"""MedGemma-powered concept extractor for OpenMRS.

Extracts structured medical concepts (CIEL-coded observations) from
transcribed clinical phrases. Supports two backends:
  - vLLM HTTP API (preferred): set VLLM_BASE_URL env var
  - Direct model loading: loads MedGemma 4-bit via transformers
Falls back to rule-based extraction when LLM is unavailable.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CIEL_MAPPINGS_PATH = Path(__file__).parent / "ciel_mappings.json"

# System prompt template for MedGemma concept extraction
SYSTEM_PROMPT = """\
You are a medical concept extractor for an OpenMRS clinic in Africa.
Your task: extract structured medical data from short spoken clinical phrases.
Return ONLY valid JSON — no explanation, no markdown.

RULES:
1. Map each phrase to one or more coded OpenMRS observations.
2. Use ONLY concepts from the provided CIEL dictionary below.
3. For numeric values, extract the number and map to the correct concept.
4. For diagnoses, include ICD-10 code and certainty (confirmed/provisional/presumed).
5. For medications, extract drug, dose, unit, frequency, duration, and route.
6. For lab/imaging orders, identify the test type.
7. If the phrase is ambiguous, pick the most likely clinical interpretation.
8. If no concept matches, return {"observations": [], "unmatched": "<original phrase>"}.

OUTPUT FORMAT:
{
  "observations": [
    {
      "concept_id": <CIEL integer ID>,
      "concept_uuid": "<UUID if known>",
      "label": "<human-readable name>",
      "value": <extracted value — number, string, or coded concept>,
      "datatype": "numeric|coded|text|datetime",
      "units": "<unit if numeric>",
      "confidence": <0.0-1.0>
    }
  ],
  "cds_alerts": [
    {
      "type": "warning|info|critical",
      "message": "<clinical decision support alert if any>"
    }
  ]
}

For medication orders, use this observation format:
{
  "concept_id": <drug CIEL ID>,
  "label": "<drug name>",
  "value": {
    "drug_concept_id": <CIEL ID>,
    "dose": <number>,
    "dose_unit": "mg|ml|tablet|capsule",
    "frequency": "OD|BD|TDS|TID|QID|STAT|PRN|nocte",
    "duration": <number>,
    "duration_unit": "days|weeks|months",
    "route": "oral|IV|IM|SC|topical|inhaled"
  },
  "datatype": "drug_order",
  "confidence": <0.0-1.0>
}

For diagnoses, use:
{
  "concept_id": <diagnosis CIEL ID>,
  "label": "<diagnosis name>",
  "value": {
    "icd10": "<ICD-10 code>",
    "certainty": "confirmed|provisional|presumed"
  },
  "datatype": "coded",
  "confidence": <0.0-1.0>
}

CIEL CONCEPT DICTIONARY:
{ciel_concepts}
"""

# Frequency aliases used by clinicians
FREQUENCY_MAP = {
    "once daily": "OD", "once a day": "OD", "daily": "OD", "od": "OD",
    "twice daily": "BD", "twice a day": "BD", "bd": "BD", "bid": "BD",
    "three times daily": "TDS", "three times a day": "TDS", "tds": "TDS",
    "tid": "TDS", "three times": "TDS",
    "four times daily": "QID", "four times a day": "QID", "qid": "QID",
    "immediately": "STAT", "stat": "STAT",
    "as needed": "PRN", "prn": "PRN", "when needed": "PRN",
    "at night": "nocte", "at bedtime": "nocte", "nocte": "nocte",
}


class ConceptExtractor:
    """Extracts structured CIEL-coded observations from clinical phrases."""

    def __init__(
        self,
        model_name: str = "google/medgemma2-4b-it",
        ciel_mappings_path: str = str(CIEL_MAPPINGS_PATH),
        device: Optional[str] = None,
        quantize: bool = True,
        vllm_base_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.ciel_mappings_path = ciel_mappings_path
        self.quantize = quantize
        self.device = device
        self.vllm_base_url = vllm_base_url or os.environ.get("VLLM_BASE_URL")

        self._model = None
        self._tokenizer = None
        self._ciel_data = None
        self._system_prompt = None
        self._http_client = None

    def load(self) -> None:
        """Load MedGemma model and CIEL mappings."""
        self._load_ciel_mappings()
        if self.vllm_base_url:
            self._load_vllm_client()
        else:
            self._load_model()

    def _load_ciel_mappings(self) -> None:
        """Load and format CIEL concept dictionary for the system prompt."""
        logger.info("Loading CIEL mappings from %s", self.ciel_mappings_path)

        with open(self.ciel_mappings_path) as f:
            self._ciel_data = json.load(f)

        # Format concepts for the system prompt
        lines = []
        for category_name, category in self._ciel_data.get("categories", {}).items():
            lines.append(f"\n## {category_name.upper()}")
            for concept in category.get("concepts", []):
                ciel_id = concept["ciel_id"]
                name = concept["name"]
                datatype = concept["datatype"]
                synonyms = ", ".join(concept.get("synonyms", []))
                units = concept.get("units", "")
                icd10 = concept.get("icd10", "")

                line = f"- {name}: CIEL:{ciel_id}, {datatype}"
                if units:
                    line += f", units: {units}"
                if icd10:
                    line += f", ICD-10: {icd10}"
                if synonyms:
                    line += f" (synonyms: {synonyms})"
                lines.append(line)

        ciel_text = "\n".join(lines)
        self._system_prompt = SYSTEM_PROMPT.replace("{ciel_concepts}", ciel_text)
        logger.info("CIEL mappings loaded: %d categories", len(self._ciel_data.get("categories", {})))

    def _load_vllm_client(self) -> None:
        """Set up HTTP client for vLLM server."""
        import httpx

        logger.info("Using vLLM backend at %s", self.vllm_base_url)
        self._http_client = httpx.Client(
            base_url=self.vllm_base_url,
            timeout=60.0,
        )
        # Mark _model as a sentinel so the rest of the code knows LLM is available
        self._model = "vllm"
        logger.info("vLLM client configured")

    def _load_model(self) -> None:
        """Load MedGemma model with optional 4-bit quantization."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading MedGemma model: %s (quantize=%s)", self.model_name, self.quantize)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        load_kwargs = {}
        if self.quantize:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        else:
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            load_kwargs["device_map"] = device
            load_kwargs["torch_dtype"] = torch.bfloat16

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        self._model.eval()
        logger.info("MedGemma model loaded successfully")

    def extract(
        self,
        text: str,
        form_context: Optional[str] = None,
        encounter_history: Optional[list[dict]] = None,
    ) -> dict:
        """Extract structured observations from a transcribed clinical phrase.

        Args:
            text: Transcribed clinical phrase (e.g., "temperature thirty eight point five").
            form_context: Optional context about which form section is active
                         (e.g., "vitals", "diagnosis", "medications").
            encounter_history: Optional list of previously extracted observations
                              in this encounter (for CDS alerts).

        Returns:
            dict with "observations" (list of coded obs) and "cds_alerts" (list).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build the user message
        user_msg = f'Extract structured medical data from this spoken phrase: "{text}"'
        if form_context:
            user_msg += f"\nCurrent form section: {form_context}"
        if encounter_history:
            user_msg += f"\nPrevious observations in this encounter: {json.dumps(encounter_history)}"

        if self._http_client is not None:
            return self._extract_via_vllm(user_msg, text)
        return self._extract_via_local(user_msg, text)

    def _extract_via_vllm(self, user_msg: str, original_phrase: str) -> dict:
        """Extract concepts via vLLM HTTP API."""
        messages = [
            {"role": "user", "content": self._system_prompt + "\n\n" + user_msg},
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9,
        }

        try:
            resp = self._http_client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            response_text = data["choices"][0]["message"]["content"]
            return self._parse_response(response_text, original_phrase)
        except Exception as e:
            logger.warning("vLLM extraction failed: %s, using rule-based fallback", e)
            return self._rule_based_fallback(original_phrase)

    def _extract_via_local(self, user_msg: str, original_phrase: str) -> dict:
        """Extract concepts via locally loaded model."""
        messages = [
            {"role": "user", "content": self._system_prompt + "\n\n" + user_msg},
        ]

        # Tokenize
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        input_ids = input_ids.to(self._model.device)

        # Generate
        import torch
        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0, input_ids.shape[1]:]
        response_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Parse JSON from response
        return self._parse_response(response_text, original_phrase)

    def _parse_response(self, response_text: str, original_phrase: str) -> dict:
        """Parse MedGemma's JSON response, handling common formatting issues."""
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                result = json.loads(json_match.group())
                # Ensure required keys exist
                if "observations" not in result:
                    result["observations"] = []
                if "cds_alerts" not in result:
                    result["cds_alerts"] = []
                return result
            except json.JSONDecodeError:
                pass

        # If JSON parsing fails, try the rule-based fallback
        logger.warning("Failed to parse MedGemma response as JSON, trying fallback. Response: %s", response_text[:200])
        return self._rule_based_fallback(original_phrase)

    def _rule_based_fallback(self, text: str) -> dict:
        """Rule-based concept extraction fallback when LLM fails.

        Handles common vital sign patterns that are unambiguous.
        """
        if self._ciel_data is None:
            return {"observations": [], "cds_alerts": [], "fallback": True}

        text_lower = text.lower().strip()
        observations = []

        # Build synonym lookup from CIEL data
        synonym_map = {}
        for category in self._ciel_data.get("categories", {}).values():
            for concept in category.get("concepts", []):
                for synonym in concept.get("synonyms", []):
                    synonym_map[synonym.lower()] = concept

        # Try to match vitals patterns: "temperature 38.5", "bp 120 over 80", etc.
        vitals_patterns = [
            # Temperature
            (r'(?:temperature|temp)\s+(\d+\.?\d*)', 5088, "Temperature (C)", "numeric", "DEG C"),
            # Systolic BP
            (r'(?:bp|blood\s*pressure)\s+(\d+)\s*(?:over|/)', 5085, "Systolic Blood Pressure", "numeric", "mmHg"),
            # Diastolic BP
            (r'(?:bp|blood\s*pressure)\s+\d+\s*(?:over|/)\s*(\d+)', 5086, "Diastolic Blood Pressure", "numeric", "mmHg"),
            # Heart rate
            (r'(?:heart\s*rate|pulse|hr)\s+(\d+)', 5087, "Heart Rate", "numeric", "bpm"),
            # SpO2
            (r'(?:oxygen\s*sat|spo2|sp\s*o\s*2|o2\s*sat|saturation|sats?)\s+(\d+)', 5092, "SpO2", "numeric", "%"),
            # Weight
            (r'(?:weight|wt)\s+(\d+\.?\d*)', 5089, "Weight (kg)", "numeric", "kg"),
            # Height
            (r'(?:height|ht)\s+(\d+\.?\d*)', 5090, "Height (cm)", "numeric", "cm"),
            # Respiratory rate
            (r'(?:respiratory\s*rate|resp\s*rate|rr|respirations)\s+(\d+)', 5242, "Respiratory Rate", "numeric", "breaths/min"),
            # Blood glucose
            (r'(?:blood\s*(?:sugar|glucose)|glucose|bsl|rbs)\s+(\d+\.?\d*)', 887, "Blood Glucose", "numeric", "mg/dL"),
        ]

        for pattern, ciel_id, label, datatype, units in vitals_patterns:
            match = re.search(pattern, text_lower)
            if match:
                value = float(match.group(1))
                observations.append({
                    "concept_id": ciel_id,
                    "label": label,
                    "value": value,
                    "datatype": datatype,
                    "units": units,
                    "confidence": 0.85,
                })

        # Try to match diagnosis/symptom by synonym lookup
        if not observations:
            for synonym, concept in synonym_map.items():
                if synonym in text_lower and len(synonym) > 2:
                    obs = {
                        "concept_id": concept["ciel_id"],
                        "label": concept["name"],
                        "datatype": concept["datatype"],
                        "confidence": 0.7,
                    }
                    if concept.get("icd10"):
                        obs["value"] = {"icd10": concept["icd10"], "certainty": "provisional"}
                    else:
                        obs["value"] = True
                    observations.append(obs)
                    break

        return {"observations": observations, "cds_alerts": [], "fallback": True}

    def get_ciel_concepts_summary(self) -> dict:
        """Return a summary of loaded CIEL concepts for debugging."""
        if self._ciel_data is None:
            return {}
        summary = {}
        for cat_name, cat in self._ciel_data.get("categories", {}).items():
            summary[cat_name] = len(cat.get("concepts", []))
        return summary
