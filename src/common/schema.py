"""Data contracts, TypedDicts, and JSON Schemas for Folk Medicine AI Agents."""

from typing import Any, Literal, TypedDict


ReviewStatus = Literal[
    "auto_accepted",
    "needs_review",
    "reviewed_ok",
    "reviewed_edited",
    "extraction_failed",
]


class ExtractedBlock(TypedDict, total=False):
    """Schema for the context-enriched 'extracted' block attached to dataset rows."""
    symptom_terms: list[str]
    canonical_ingredients: list[str]
    ingredient_aliases: dict[str, str]
    preparation: str | None
    season_climate_context: str | None
    usage_context: str | None
    topic_tags: list[str]
    mentions_pregnancy_caution: bool
    mentions_child_caution: bool
    evidence: dict[str, str]
    confidence: float
    review_status: ReviewStatus
    model: str
    prompt_version: str
    extracted_at: str


class ParsedQuery(TypedDict, total=False):
    """Schema for user query parsed by Agent 1 query parser."""
    doshas: list[str]
    symptom: str
    age: int | None
    duration: str | None
    red_flags: list[str]
    pregnant_or_breastfeeding: bool | None
    language_detected: str
    missing_fields: list[str]
    is_clarification_needed: bool


class Agent2Input(TypedDict):
    """Input contract for Agent 2 recommendation."""
    doshas: list[str]
    condition: str
    age: int
    pregnant: bool
    breastfeeding: bool


class ProvenanceInfo(TypedDict):
    """Provenance tracking for Agent 2 output."""
    knowledge_source: Literal["classical", "folk"]
    evidence_level: str
    dosage_origin: Literal["from_source", "derived"]


class Agent2Output(TypedDict):
    """Final 6-field structured output contract for Agent 2."""
    status: Literal["ok", "see_doctor", "no_match"]
    source_row_id: str | None
    remedy: str | None
    dosage_by_age: str | None
    ingredients: list[str]
    lifestyle: str | None
    precautions: str | None
    foods_to_avoid: list[str]
    escalation: str | None
    provenance: ProvenanceInfo


# JSON Schema for Gemini extract_v1 output
EXTRACTED_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "symptom_terms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Standardized clinical and lay symptom phrases mentioned in the text."
        },
        "canonical_ingredients": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Plant or mineral ingredients mentioned in the remedy."
        },
        "ingredient_aliases": {
            "type": "object",
            "description": "Local or vernacular names mapped to canonical ingredient names."
        },
        "preparation": {
            "type": ["string", "null"],
            "description": "Form of preparation (e.g. decoction, powder, paste) if mentioned, else null."
        },
        "season_climate_context": {
            "type": ["string", "null"],
            "description": "Climatic or seasonal context (e.g. winter, monsoon) if mentioned, else null."
        },
        "usage_context": {
            "type": ["string", "null"],
            "description": "Timing or manner of consumption (e.g. warm with water, after meals) if mentioned, else null."
        },
        "topic_tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "High-level domain tags such as respiratory, digestive, skin, musculoskeletal."
        },
        "mentions_pregnancy_caution": {
            "type": "boolean",
            "description": "True if the text mentions or implies caution during pregnancy, else false."
        },
        "mentions_child_caution": {
            "type": "boolean",
            "description": "True if the text mentions caution for children or infants, else false."
        },
        "evidence": {
            "type": "object",
            "description": "Verbatim quote from RECORD supporting each non-null extracted claim.",
            "additionalProperties": {"type": "string"}
        }
    },
    "required": [
        "symptom_terms",
        "canonical_ingredients",
        "preparation",
        "topic_tags",
        "mentions_pregnancy_caution",
        "mentions_child_caution",
        "evidence"
    ]
}


# JSON Schema for Gemini query_parse_v1 output
QUERY_PARSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "doshas": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Explicit doshas mentioned by user (vata, pitta, kapha). Empty if not stated."
        },
        "symptom": {
            "type": "string",
            "description": "Primary symptom or condition phrase described by the user."
        },
        "age": {
            "type": ["integer", "null"],
            "description": "Age of the patient as an integer, or null if omitted."
        },
        "duration": {
            "type": ["string", "null"],
            "description": "Duration of symptoms if stated (e.g. 3 days), else null."
        },
        "red_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Any high-risk red-flag symptoms mentioned (e.g. blood, severe chest pain, breathing difficulty)."
        },
        "pregnant_or_breastfeeding": {
            "type": ["boolean", "null"],
            "description": "True if user mentions being pregnant or breastfeeding, False if explicitly not, null if not mentioned."
        },
        "language_detected": {
            "type": "string",
            "description": "Detected language (e.g. English, Hinglish, Hindi)."
        },
        "missing_fields": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Essential fields that are missing (e.g. age, dosha)."
        }
    },
    "required": [
        "doshas",
        "symptom",
        "age",
        "red_flags",
        "language_detected",
        "missing_fields"
    ]
}
