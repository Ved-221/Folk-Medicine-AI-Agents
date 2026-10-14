"""Agent 1 Row Context Extractor.

Performs offline LLM enrichment of traditional medicine rows using versioned
extraction prompts, verifies outputs via the validator, and attaches the
validated `extracted` block to each record without modifying original fields.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent1.validator import build_record_text, validate_extracted_block
from src.common.config import ENRICHED_DATA_PATH, ROOT_DIR
from src.common.llm_client import LLMClient, default_client
from src.common.schema import EXTRACTED_OUTPUT_JSON_SCHEMA, ExtractedBlock

PROMPTS_DIR = ROOT_DIR / "src" / "agent1" / "prompts"


def load_prompt_template(version: str = "extract_v1") -> str:
    """Loads a versioned prompt template from src/agent1/prompts/."""
    file_path = PROMPTS_DIR / f"{version}.txt"
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt template {version} not found at {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def enrich_row(
    row: dict[str, Any],
    client: LLMClient | None = None,
    prompt_version: str = "extract_v1",
    use_cache: bool = True,
) -> dict[str, Any]:
    """Enriches a single dataset record with an LLM-extracted context block.

    Args:
        row: Cleaned dataset record.
        client: LLMClient instance (defaults to global default_client).
        prompt_version: Version of prompt to use.
        use_cache: Whether to read/write persistent disk cache.

    Returns:
        New dictionary copy with row['extracted'] block attached. Original
        fields are guaranteed to remain untouched.
    """
    enriched = dict(row)
    llm = client or default_client

    # 1. Build clean textual representation
    record_text = build_record_text(row)
    prompt_template = load_prompt_template(prompt_version)
    prompt = prompt_template.replace("{record_text}", record_text)

    # 2. Call LLM through unified client
    try:
        raw_extracted = llm.generate_json(
            prompt=prompt,
            schema=EXTRACTED_OUTPUT_JSON_SCHEMA,
            prompt_version=prompt_version,
            use_cache=use_cache,
        )
    except Exception as e:
        # On fatal LLM failure, mark extraction_failed without halting pipeline
        now_iso = datetime.now(timezone.utc).isoformat()
        enriched["extracted"] = {
            "symptom_terms": [],
            "canonical_ingredients": [],
            "preparation": None,
            "topic_tags": [],
            "mentions_pregnancy_caution": False,
            "mentions_child_caution": False,
            "evidence": {},
            "confidence": 0.0,
            "review_status": "extraction_failed",
            "model": llm.model_name,
            "prompt_version": prompt_version,
            "extracted_at": now_iso,
            "error": str(e),
        }
        return enriched

    # 3. Validate against schema and verbatim source text quotes
    is_valid, val_msg, validated_block = validate_extracted_block(raw_extracted, record_text)

    # 4. Stamp audit metadata
    now_iso = datetime.now(timezone.utc).isoformat()
    validated_block["model"] = llm.model_name
    validated_block["prompt_version"] = prompt_version
    validated_block["extracted_at"] = now_iso

    enriched["extracted"] = validated_block
    return enriched


def enrich_dataset(
    rows: list[dict[str, Any]],
    limit: int | None = None,
    output_path: Path = ENRICHED_DATA_PATH,
    client: LLMClient | None = None,
    prompt_version: str = "extract_v1",
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Enriches a list of rows and writes them to the enriched dataset path.

    Args:
        rows: Dataset records to enrich.
        limit: Optional maximum number of rows to process.
        output_path: Destination JSONL file.
        client: LLMClient instance.
        prompt_version: Prompt version to use.
        use_cache: Whether to use disk cache.

    Returns:
        List of enriched rows.
    """
    targets = rows[:limit] if limit is not None else rows
    enriched_rows: list[dict[str, Any]] = []

    for r in targets:
        enriched_row = enrich_row(
            r, client=client, prompt_version=prompt_version, use_cache=use_cache
        )
        enriched_rows.append(enriched_row)

    if output_path:
        import json
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for er in enriched_rows:
                f.write(json.dumps(er, ensure_ascii=False) + "\n")

    return enriched_rows
