"""Agent 1 Review Queue Manager.

Manages records routed to human review due to caution flags, low confidence,
or failed quote verification. Persists queue state in data/review/queue.jsonl.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.common.config import REVIEW_QUEUE_PATH


def sync_rows_to_review_queue(
    rows: list[dict[str, Any]],
    queue_path: Path = REVIEW_QUEUE_PATH,
) -> int:
    """Scans rows and adds any item needing human review to the review queue.

    Args:
        rows: Enriched rows containing an `extracted` block.
        queue_path: Path to queue.jsonl file.

    Returns:
        Number of items added to the queue.
    """
    queue_path = Path(queue_path)
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing queue to prevent duplicate pending entries
    existing_queue: dict[str, dict[str, Any]] = {}
    if queue_path.exists():
        with open(queue_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                item = json.loads(line_str)
                existing_queue[item["row_id"]] = item

    added_count = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for r in rows:
        row_id = r.get("row_id")
        if not row_id:
            continue
        extracted = r.get("extracted", {})
        status = extracted.get("review_status")

        if status in ("needs_review", "extraction_failed"):
            if row_id not in existing_queue or existing_queue[row_id].get("resolved"):
                reasons = []
                if extracted.get("mentions_pregnancy_caution"):
                    reasons.append("Pregnancy caution flagged")
                if extracted.get("mentions_child_caution"):
                    reasons.append("Child caution flagged")
                if extracted.get("confidence", 0) < 0.8:
                    reasons.append(f"Low confidence ({extracted.get('confidence', 0):.2f})")
                if not reasons:
                    reasons.append(f"Review status: {status}")

                queue_item = {
                    "row_id": row_id,
                    "condition": r.get("condition"),
                    "knowledge_source": r.get("knowledge_source"),
                    "review_status": status,
                    "reasons": reasons,
                    "proposed_extracted": extracted,
                    "resolved": False,
                    "resolution": None,
                    "resolved_at": None,
                    "reviewer_notes": "",
                    "created_at": now_iso,
                }
                existing_queue[row_id] = queue_item
                added_count += 1

    # Write updated queue back to disk
    with open(queue_path, "w", encoding="utf-8") as f:
        for item in existing_queue.values():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return added_count


def get_pending_queue_items(queue_path: Path = REVIEW_QUEUE_PATH) -> list[dict[str, Any]]:
    """Returns all unresolved items from the review queue."""
    queue_path = Path(queue_path)
    if not queue_path.exists():
        return []

    pending: list[dict[str, Any]] = []
    with open(queue_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            if not item.get("resolved", False):
                pending.append(item)
    return pending


def resolve_queue_item(
    row_id: str,
    resolution: str,  # "reviewed_ok" or "reviewed_edited"
    edited_extracted: dict[str, Any] | None = None,
    reviewer_notes: str = "",
    queue_path: Path = REVIEW_QUEUE_PATH,
) -> bool:
    """Marks a queue item as resolved by a human reviewer.

    Args:
        row_id: ID of the row being resolved.
        resolution: 'reviewed_ok' or 'reviewed_edited'.
        edited_extracted: Corrected extracted block if resolution is 'reviewed_edited'.
        reviewer_notes: Optional comments by the human reviewer.
        queue_path: Path to queue.jsonl.

    Returns:
        True if item was found and resolved, False otherwise.
    """
    queue_path = Path(queue_path)
    if not queue_path.exists():
        return False

    items: list[dict[str, Any]] = []
    found = False
    now_iso = datetime.now(timezone.utc).isoformat()

    with open(queue_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            if item.get("row_id") == row_id:
                item["resolved"] = True
                item["resolution"] = resolution
                item["resolved_at"] = now_iso
                item["reviewer_notes"] = reviewer_notes
                if edited_extracted:
                    item["proposed_extracted"] = edited_extracted
                found = True
            items.append(item)

    if found:
        with open(queue_path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    return found
