"""Agent 2 Deterministic Safety Gate.

MANDATORY code safety gate that executes in pure Python BEFORE any LLM is called.
Short-circuits immediately to 'see_doctor' if safe_for_home_remedy is False,
care_context is emergency or clinic, pregnancy/breastfeeding contraindications exist,
or pediatric/age contraindications apply. Cannot be bypassed.
"""

from dataclasses import dataclass
from typing import Any, Literal

from src.common.schema import Agent2Input


@dataclass
class SafetyResult:
    """Outcome of the code safety gate check."""
    is_safe: bool
    status: Literal["ok", "see_doctor"]
    escalation_reason: str | None = None
    warning_notes: str | None = None


def safety_check(row: dict[str, Any], user_context: Agent2Input) -> SafetyResult:
    """Evaluates all safety constraints deterministically on a candidate row.

    Args:
        row: Dataset row (clean or trimmed).
        user_context: Validated input containing age, pregnancy, breastfeeding, etc.

    Returns:
        SafetyResult indicating whether recommendation is permitted.
    """
    esc_obj = row.get("escalation", {})
    escalation_text = esc_obj.get("condition", "") if isinstance(esc_obj, dict) else ""

    # Rule 1: safe_for_home_remedy must be explicitly True
    safe_for_home = row.get("safe_for_home_remedy")
    if safe_for_home is False:
        reason = escalation_text or "This condition is marked unsafe for self-treatment at home. Consult a physician."
        return SafetyResult(is_safe=False, status="see_doctor", escalation_reason=reason)

    # Rule 2: care_context cannot be 'emergency' or 'clinic'
    care_context = str(row.get("care_context", "")).lower()
    if care_context in ("emergency", "clinic"):
        reason = escalation_text or f"Care context requires {care_context} medical management. Seek immediate clinical care."
        return SafetyResult(is_safe=False, status="see_doctor", escalation_reason=reason)

    gender_cons = row.get("gender_considerations", {})
    if not isinstance(gender_cons, dict):
        gender_cons = {}

    # Rule 3: Pregnancy check
    if user_context.get("pregnant"):
        preg_safe = gender_cons.get("pregnancy_safe")
        preg_note = gender_cons.get("pregnancy_note", "")
        extracted_preg = row.get("extracted", {}).get("mentions_pregnancy_caution")
        if (
            preg_safe is False
            or (preg_safe is None and preg_note and ("avoid" in preg_note.lower() or "contraindicated" in preg_note.lower()))
            or extracted_preg
        ):
            reason = preg_note or "Remedy contains herbs contraindicated or unverified for safety during pregnancy. Consult a specialist."
            return SafetyResult(is_safe=False, status="see_doctor", escalation_reason=f"Pregnancy contraindication: {reason}")

    # Rule 4: Breastfeeding check
    if user_context.get("breastfeeding"):
        bf_safe = gender_cons.get("breastfeeding_safe")
        bf_note = gender_cons.get("breastfeeding_note", "")
        if bf_safe is False or (bf_note and ("avoid" in bf_note.lower() or "contraindicated" in bf_note.lower())):
            reason = bf_note or "Remedy contains herbs contraindicated during lactation. Seek alternative medical advice."
            return SafetyResult(is_safe=False, status="see_doctor", escalation_reason=f"Breastfeeding contraindication: {reason}")

    # Rule 5: Infant safety check (< 3 years)
    age = user_context.get("age")
    if age is not None and age < 3:
        reason = "Infants under 3 years require direct paediatric specialist consultation for any herbal administration."
        return SafetyResult(is_safe=False, status="see_doctor", escalation_reason=reason)

    # Rule 6: Pending review warning
    extracted = row.get("extracted", {})
    warning = None
    if isinstance(extracted, dict) and extracted.get("review_status") in ("needs_review", "extraction_failed"):
        warning = "Notice: This remedy record is pending human review and verification. Exercise caution and verify with a qualified clinician."

    return SafetyResult(is_safe=True, status="ok", escalation_reason=None, warning_notes=warning)
