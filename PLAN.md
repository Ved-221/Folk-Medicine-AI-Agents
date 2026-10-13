# Folk Medicine AI Agents (v2.0)

**Project:** Preserving indigenous folk medicines — research honors programme  
**Deliverables:** Agent 1 (data extraction, categorization, retrieval) and Agent 2 (dosha-based remedy recommender)  
**Environment:** Google Antigravity, Python 3.10+, pandas, google-genai, pytest, rapidfuzz  

## Architecture Overview
The system employs two deterministic pipelines:
1. **Agent 1 (Data Layer & Context Extraction):**
   - Ingests raw traditional records (`data/raw/`).
   - Normalizes and trims synthetic scaffolding into `data/clean/`.
   - Offline LLM extractor enriches context with verbatim evidence quotes and confidence scores into `data/enriched/`.
   - Embeds text representation into vector index (`data/index/`).
   - Online query parser converts natural language user text into a structured query.
   - Hybrid retrieval combines deterministic dosha overlap filtering, semantic search, and fuzzy matching.
2. **Agent 2 (Deterministic Remedy Recommender):**
   - Pure dictionary lookup for age-band dosage (`src/agent2/age_band.py`).
   - Mandatory code safety gate (`src/agent2/safety.py`) evaluates before ANY LLM call.
   - Constrained rephrasing LLM formatter (`src/agent2/formatter.py`) formats strictly from retrieved source data.
   - Outputs 6 structured fields: Remedy, Dosage (age-wise), Ingredients, Lifestyle, Precautions, Foods to Avoid.

## Key Invariants
- Code governs decisions (dosha eligibility, age banding, safety gating, dosage).
- The LLM never invents medical facts; every extraction requires a verbatim quote.
- Single unified LLM client (`src/common/llm_client.py`) with persistent disk cache to protect free-tier API quotas.
