# Project rules: folk-medicine-agents v2

## Stack
- Plain Python, pandas, google-genai, rapidfuzz, pytest. No agent frameworks.
- Type hints and docstrings on all functions.

## LLM boundaries
- The LLM may: extract fields from text, parse user messages into structured
  input, generate embeddings, rephrase retrieved facts.
- The LLM may NOT: decide dosha eligibility, choose a dosage, decide whether a
  remedy is safe, or add medical facts not in the source text.
- All Gemini calls go through src/common/llm_client.py only.
- Every extraction must include a supporting quote from the source text per
  field. Reject outputs with missing or non-matching quotes.
- LLM-extracted values go in the `extracted` block. Never overwrite original
  fields. Extracted caution flags may add caution but never remove it.

## Data
- Never modify data/raw/. Never edit data/gold/ from code.
- Normalize dosha values to lowercase: vata, pitta, kapha.
- Never log or print the API key.

## Safety
- src/agent2/safety.py runs before any LLM call in Agent 2 and cannot be bypassed.
- Rows with review_status needs_review or extraction_failed must not feed
  safety-related or dosage fields to Agent 2.

## Workflow
- Produce a plan and wait for approval before large changes.
- Add a pytest test for every function. Mock the LLM in unit tests.
- Run tests before declaring a task done.
- Keep tasks small. Do not refactor unrelated files.
- Ask when dataset behavior is unclear instead of guessing.
