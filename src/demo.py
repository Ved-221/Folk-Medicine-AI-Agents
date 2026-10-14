"""Interactive Command-Line Demo for Folk Medicine AI Agents.

Demonstrates end-to-end query parsing, deterministic safety gating,
hybrid retrieval, age-banding, and constrained recommendation formatting.
"""

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path so 'src' can be imported when running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent2.pipeline import get_recommendation_from_text


def run_demo_case(title: str, query_text: str):
    print("\n" + "=" * 70)
    print(f"CASE: {title}")
    print(f"INPUT: \"{query_text}\"")
    print("-" * 70)

    result = get_recommendation_from_text(query_text, use_cache=True)

    status_str = result.get("status", "unknown").upper()
    print(f"STATUS: {status_str}")
    print(f"SOURCE ROW ID: {result.get('source_row_id')}")

    prov = result.get("provenance", {})
    print(f"PROVENANCE: {prov.get('knowledge_source')} | Origin: {prov.get('dosage_origin')}")

    if result.get("status") == "ok":
        print("\nRECOMMENDED REMEDY (6 Standard Fields):")
        print(f"  1. Remedy:         {result.get('remedy')}")
        print(f"  2. Dosage by Age:  {result.get('dosage_by_age')}")
        print(f"  3. Ingredients:    {', '.join(result.get('ingredients', []))}")
        print(f"  4. Lifestyle:      {result.get('lifestyle')}")
        print(f"  5. Precautions:    {result.get('precautions')}")
        print(f"  6. Foods to Avoid: {', '.join(result.get('foods_to_avoid', []))}")
    elif result.get("status") == "see_doctor":
        print("\nSAFETY GATE TRIGGERED (SHORT-CIRCUITED - ZERO LLM INVENTIONS):")
        print(f"  Precautions: {result.get('precautions')}")
        print(f"  Escalation:  {result.get('escalation')}")
    else:
        print("\nCLARIFICATION / NO MATCH:")
        print(f"  Precautions: {result.get('precautions')}")
        print(f"  Guidance:    {result.get('escalation')}")


def main():
    print("🌿 FOLK MEDICINE AI AGENTS (V2.0) - DEMONSTRATION")
    print("Grounded Indigenous Medicine Preservation & Recommendation Engine")

    # 1. Standard Clinical / Colloquial Query
    run_demo_case(
        "1. Standard User Query (Dry Cough)",
        "I am 30 years old, vata pitta dosha, suffering from a dry hacking cough."
    )

    # 2. Hinglish Query
    run_demo_case(
        "2. Hinglish / Vernacular Query (Acid Reflux)",
        "meri umar 40 saal hai, pitta dosha, khane ke baad chhati mein jalan aur acid banta hai."
    )

    # 3. Medical Emergency (Red-Flag Short-Circuit)
    run_demo_case(
        "3. Emergency Red Flag Screening (Instant Clinician Escalation)",
        "Age 45, coughing up bright red blood and severe crushing chest pain."
    )

    # 4. Missing Information (Clarification Gate)
    run_demo_case(
        "4. Incomplete Information (Clarification Gate)",
        "I have knee pain and joint swelling, please help."
    )


if __name__ == "__main__":
    main()
