"""Agent 1 Loader & Normalizer.

Loads the raw traditional medicine dataset, normalizes dosha casing, strips
extraneous whitespace, and saves the cleaned records while strictly preserving
the original nested record structure without mutating raw files.
"""

import copy
import json
from pathlib import Path
from typing import Any

from src.common.config import CLEAN_DATA_PATH, RAW_DATA_PATH, VALID_DOSHAS


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Loads a JSONL dataset file into a list of nested dictionaries.

    Args:
        path: Path to the JSONL dataset.

    Returns:
        List of parsed JSON objects.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    rows: list[dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rows.append(json.loads(line_str))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {line_num} in {file_path}: {e}") from e
    return rows


def clean_row(row: dict[str, Any], default_id: str | None = None) -> dict[str, Any]:
    """Cleans and normalizes a single dataset record.

    Normalization actions:
    1. Strips whitespace in `condition`.
    2. Lowercases and strips dosha tokens in `dosha_involvement`.
    3. Normalizes 'rakta' (blood/pitta-predominant tissue) to 'pitta' if present.
    4. Filters doshas strictly to valid Ayurveda doshas: {vata, pitta, kapha}.
    5. Deduplicates dosha values while preserving encounter order.
    6. Ensures a unique `row_id` identifier exists.
    7. Preserves full nested structure without flattening.

    Args:
        row: Original nested record.
        default_id: Optional ID to assign if the row has none.

    Returns:
        Cleaned copy of the row.
    """
    cleaned = copy.deepcopy(row)

    # Ensure row_id exists for unambiguous provenance
    if "row_id" not in cleaned and "id" not in cleaned:
        cleaned["row_id"] = default_id if default_id is not None else ""
    elif "row_id" not in cleaned and "id" in cleaned:
        cleaned["row_id"] = str(cleaned["id"])

    # 1. Condition whitespace stripping
    if "condition" in cleaned and isinstance(cleaned["condition"], str):
        cleaned["condition"] = cleaned["condition"].strip()

    # 2. Dosha normalization
    raw_doshas = cleaned.get("dosha_involvement", [])
    if isinstance(raw_doshas, list):
        normalized_doshas: list[str] = []
        for d in raw_doshas:
            if not isinstance(d, str):
                continue
            token = d.strip().lower()
            if token == "rakta":
                token = "pitta"
            if token in VALID_DOSHAS and token not in normalized_doshas:
                normalized_doshas.append(token)
        cleaned["dosha_involvement"] = normalized_doshas

    return cleaned


def save_dataset(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Saves records to a JSONL file.

    Args:
        rows: List of dictionary records.
        path: Destination file path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_and_save_dataset(
    raw_path: str | Path = RAW_DATA_PATH,
    clean_path: str | Path = CLEAN_DATA_PATH,
) -> list[dict[str, Any]]:
    """Loads raw dataset, cleans every row, and writes to the clean dataset path.

    Args:
        raw_path: Path to raw JSONL file.
        clean_path: Path to destination cleaned JSONL file.

    Returns:
        List of cleaned rows.
    """
    raw_rows = load_dataset(raw_path)
    cleaned_rows: list[dict[str, Any]] = []

    for i, r in enumerate(raw_rows, start=1):
        cleaned_rows.append(clean_row(r, default_id=f"row_{i:03d}"))

    save_dataset(cleaned_rows, clean_path)
    return cleaned_rows


if __name__ == "__main__":
    cleaned = process_and_save_dataset()
    print(f"Cleaned {len(cleaned)} rows and saved to {CLEAN_DATA_PATH}")
