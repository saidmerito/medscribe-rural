# ai/structurer.py
"""
Core AI module: uses Gemma 4 E4B via Ollama to transform raw OCR text
into structured medical records with ICD-10 codes.

Gemma 4's native function calling enforces strict JSON schema compliance.
"""

import json
import re
import httpx
from datetime import datetime
from config import OLLAMA_BASE_URL, GEMMA_MODEL, CONFIDENCE_THRESHOLD

# JSON schema for a single patient record
RECORD_SCHEMA = {
    "type": "object",
    "properties": {
        "patient_id":       {"type": "string", "description": "Patient identifier from register (or generated if absent)"},
        "date":             {"type": "string", "description": "Visit date in ISO format YYYY-MM-DD"},
        "age":              {"type": "integer", "description": "Patient age in years"},
        "sex":              {"type": "string", "enum": ["M", "F", "unknown"]},
        "chief_complaint":  {"type": "string", "description": "Main symptom(s) in English"},
        "diagnosis":        {"type": "string", "description": "Clinical diagnosis in English"},
        "icd10_code":       {"type": "string", "description": "ICD-10 code (e.g. A09, B54)"},
        "treatment":        {"type": "string", "description": "Treatment given"},
        "outcome":          {"type": "string", "enum": ["discharged", "referred", "admitted", "deceased", "unknown"]},
        "confidence":       {"type": "number", "description": "Overall confidence score 0.0–1.0"},
        "flags":            {"type": "array", "items": {"type": "string"}, "description": "List of uncertain fields"},
        "raw_text":         {"type": "string", "description": "Original OCR text for this record"},
    },
    "required": ["patient_id", "date", "age", "sex", "chief_complaint", "diagnosis", "icd10_code", "treatment", "outcome", "confidence", "flags", "raw_text"]
}

SYSTEM_PROMPT = """You are a medical data extraction assistant specialized in digitizing handwritten health center registers from rural Africa. 

Your task:
1. Parse raw OCR text from a medical register (may contain French, Arabic, or Somali text)
2. Correct OCR errors using medical context
3. Extract structured patient data
4. Assign the correct ICD-10 code based on the diagnosis
5. Translate all fields to English
6. Assign a confidence score (0.0–1.0) based on legibility
7. List any uncertain fields in the "flags" array instead of guessing

Common ICD-10 codes for this context:
- A00: Cholera | A09: Acute diarrhea | A15: Tuberculosis | A33-A37: Neonatal/pertussis
- B50-B54: Malaria | B05: Measles | J18: Pneumonia | E40-E46: Malnutrition
- O00-O99: Pregnancy complications | Z23-Z28: Vaccinations

CRITICAL: 
- Never invent data. If a field is illegible, flag it.
- Dates: convert to YYYY-MM-DD. If year is ambiguous, use current year.
- Patient IDs: if absent in record, generate one as "AUTO-{timestamp}".
- You MUST respond with a JSON array of patient record objects, one per patient found in the text.
"""


def _call_ollama(prompt: str) -> str:
    """Call Gemma 4 E4B via Ollama API with function calling for schema enforcement."""
    payload = {
        "model": GEMMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "save_medical_records",
                    "description": "Save structured medical records extracted from the register",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "records": {
                                "type": "array",
                                "items": RECORD_SCHEMA,
                                "description": "List of patient records extracted from the register page"
                            }
                        },
                        "required": ["records"]
                    }
                }
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "save_medical_records"}},
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for factual extraction
            "num_predict": 2048,
        }
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()
        return response.json()


def _parse_ollama_response(response: dict) -> list[dict]:
    """Extract structured records from Ollama tool call response."""
    message = response.get("message", {})

    # Try tool_calls first (function calling)
    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        for call in tool_calls:
            if call.get("function", {}).get("name") == "save_medical_records":
                args = call["function"].get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                return args.get("records", [])

    # Fallback: parse JSON from message content
    content = message.get("content", "")
    json_match = re.search(r'\[.*\]', content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())

    return []


def structure_records(ocr_text: str, facility_name: str = "Unknown") -> list[dict]:
    """
    Main function: takes raw OCR text, returns list of structured patient records.
    
    Args:
        ocr_text: Raw text extracted by OCR from a register page
        facility_name: Health center name for record metadata
    
    Returns:
        List of structured patient record dicts
    """
    if not ocr_text.strip():
        return []

    prompt = f"""Please extract all patient records from this medical register page.
Facility: {facility_name}
Extraction date: {datetime.now().strftime('%Y-%m-%d')}

--- RAW OCR TEXT ---
{ocr_text}
--- END ---

Extract every patient entry you can identify. Use the save_medical_records function."""

    response = _call_ollama(prompt)
    records = _parse_ollama_response(response)

    # Post-process: add facility, timestamp, flag low-confidence records
    processed = []
    for record in records:
        record["facility"] = facility_name
        record["extracted_at"] = datetime.now().isoformat()

        # Auto-flag low confidence
        if record.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
            if "low_overall_confidence" not in record.get("flags", []):
                record.setdefault("flags", []).append("low_overall_confidence")

        processed.append(record)

    return processed


def structure_single_record(text: str) -> dict | None:
    """Convenience wrapper for structuring a single patient entry."""
    records = structure_records(text)
    return records[0] if records else None
