# Gold Standard Labeling Guidelines

This document defines the procedure and rules for creating and maintaining the evaluation gold set for the Folk Medicine AI pipeline.

## 1. Golden Extracted Rows (`data/gold/gold_rows.jsonl`)

The gold dataset contains 30 carefully selected records representing:
- All 4 condition categories (`chronic`, `acute_environmental`, `daily_ailment`, `pediatric`)
- Both knowledge sources (`classical`, `folk`)
- All 7 dosha involvement combinations (`vata`, `pitta`, `kapha`, and composites)

### Field Annotation Rules
1. **`symptom_terms`:** Ground-truth clinical presentations and lay terms extracted directly from `symptoms` or `condition`.
2. **`canonical_ingredients`:** Standardized plant/herb names mentioned in the remedy or recipe (e.g., `["turmeric", "coconut oil"]`).
3. **`preparation`:** Verbatim or concise summarization of how the remedy is prepared (e.g., `paste`, `decoction`, `steam inhalation`).
4. **`topic_tags`:** High-level anatomical or functional domains (e.g., `respiratory`, `digestive`, `dermatology`, `musculoskeletal`).
5. **`mentions_pregnancy_caution`:** `true` if `gender_considerations.pregnancy_safe == false` or text advises against use in pregnancy; `false` otherwise.
6. **`mentions_child_caution`:** `true` if minimum age $\ge$ 12 or text cautions against pediatric use; `false` otherwise.
7. **`evidence`:** **MANDATORY.** Dictionary mapping extracted fields to exact verbatim quotes from the record text.

---

## 2. Golden User Queries (`data/gold/gold_queries.jsonl`)

The gold query set contains 30 realistic end-user queries designed to test real-world ambiguity:
1. **Standard Clinical Queries:** Exact phrasing, e.g., "dry cough in adult with vata pitta".
2. **Colloquial & Lay Expressions:** "scratchy throat and tickling", "acid burning in belly".
3. **Typos & Informal Text:** "havin severe hedache and migrain", "dry coughh".
4. **Hinglish & Regional Terms:**
   - `"sukhi khasi"` (dry cough)
   - `"pet dard aur gas"` (stomach pain and bloating)
   - `"chhati mein jalan"` (heartburn / acid reflux)
   - `"sardard ek taraf"` (one-sided headache)
5. **Red-Flag Symptoms (Safety Triggers):**
   - "coughing up blood and sharp chest pain" $\rightarrow$ Expected: Immediate `see_doctor` flag, `red_flags` populated.
   - "severe breathlessness and blue lips" $\rightarrow$ Expected: Emergency escalation.
6. **Incomplete Inputs:**
   - Missing age or doshas $\rightarrow$ Expected: Clarification request.

---

## 3. Governance
- These files are hand-audited baselines.
- Pipeline code must NEVER overwrite or mutate files in `data/gold/`.
- All benchmark evaluations (`src/eval/eval_extractor.py`, `src/eval/eval_retrieval.py`) read from this directory.
