# Dataset Schema Discovery: Ayurvedic & Folk Medicine Dataset

**Total Records:** 500  
**Source File:** `data/raw/ayurvedic_dataset.jsonl`  

## 1. Summary of Fields & Structural Classification

The dataset contains both **authentic traditional remedy fields** and **synthetic scaffolding fields** (e.g. `dosage_model`, `dose_optimization`, `reasoning_steps`, `agent_decision_tree`).

### Core Clinical & Remedy Fields (Retained for Agents 1 & 2)
These fields represent the traditional knowledge base and are used for retrieval, age-banding, safety checks, and recommendation formatting:
- `condition`, `condition_category`, `severity`, `onset_type`
- `knowledge_source` (`classical` vs `folk`), `evidence_level`
- `symptoms` (list of clinical presentations)
- `dosha_involvement` (e.g. `['vata', 'pitta']`)
- `remedy.name`, `remedy.type` (`oral`, `topical`)
- `dosage.amount`, `dosage.frequency`, `dosage.duration`
- `age_suitability.minimum_age`, `age_suitability.recommended_age_groups`, `age_suitability.dosage_adjustments_by_age`
- `safe_for_home_remedy`, `care_context` (`home`, `clinic`, `emergency`)
- `contraindications`, `comorbidity_cautions`, `escalation`
- `gender_considerations` (`pregnancy_safe`, `pregnancy_note`, `breastfeeding_safe`, `breastfeeding_note`)
- `lifestyle_factors` (`diet_recommendations`, `foods_to_avoid`, `physical_activity`, `sleep_guidance`)

### Synthetic Scaffolding Fields (Trimmed by Agent 1)
These fields appear synthetically generated to mimic an algorithmic decision system and must be stripped before LLM consumption to avoid hallucination loops:
- `dosage_model`: simulated mathematical dose multipliers, titration formulas, simulated GFR/hepatic tables.
- `treatment_plan.dose_optimization`: synthetic grid search and objective weights.
- `treatment_plan.agent_decision_tree`: mock decision tree nodes (`N1`, `N2`, etc.).
- `treatment_plan.variation_simulations`: pre-computed mock edge cases.
- `reasoning_steps`: template-generated chain-of-thought strings.
- `final_explanation`: boilerplate summary.

---

## 2. Exhaustive Field-by-Field Analysis

| Field Path | Data Type(s) | Frequency | Missing % | Example Values |
|---|---|---|---|---|
| `age_suitability.dosage_adjustments_by_age.adolescent_12_17_years` | `str` | 231/500 | 53.8% | `50–75% of adult dose` |
| `age_suitability.dosage_adjustments_by_age.adult_18_60_years` | `str` | 500/500 | 0.0% | `standard dose as described` |
| `age_suitability.dosage_adjustments_by_age.child_3_12_years` | `str` | 162/500 | 67.6% | `25–50% of adult dose; consult Ayurvedic or paediatric practitioner` |
| `age_suitability.dosage_adjustments_by_age.elderly_60_plus_years` | `str` | 338/500 | 32.4% | `reduce by 25–30%; monitor renal and hepatic function; start low and titrate` <br> `reduce by 25–30%; start low and titrate` |
| `age_suitability.minimum_age` | `str` | 500/500 | 0.0% | `3 years` <br> `18 years` <br> `12 years` |
| `age_suitability.notes` | `str` | 500/500 | 0.0% | `Elderly patients often have reduced renal clearance; start at lower end of dose range.` <br> `Paediatric doses must be calculated by weight or age; avoid adult-strength formulations in children.` <br> `Neonates and infants require specialist Ayurvedic or medical supervision for any remedy.` |
| `age_suitability.recommended_age_groups` | `list` | 500/500 | 0.0% | `['child (3–12 years)', 'adolescent (12–18 years)', 'adult (18–60 years)']` <br> `['adult (18–60 years)', 'elderly (60+ years)']` <br> `['adolescent (12–18 years)', 'adult (18–60 years)', 'elderly (60+ years)']` |
| `care_context` | `str` | 500/500 | 0.0% | `clinic` <br> `emergency` <br> `home` |
| `comorbidity_cautions.contraindicated_in` | `list` | 500/500 | 0.0% | `['No absolute contraindications identified beyond those listed in the main contraindications fiel...` <br> `['Pregnancy: this remedy contains herbs contraindicated in pregnancy.']` <br> `['No absolute contraindications beyond those in contraindications field.']` |
| `comorbidity_cautions.dose_reduction_recommended_in` | `list` | 500/500 | 0.0% | `['Children: always calculate dose by age/weight; avoid adult-strength bhasmas (mineral preparatio...` <br> `['Elderly patients (60+): reduce dose by 25–30%; increased sensitivity to sedating herbs (Jataman...` <br> `['CKD stage 3+: reduce dose by 25–50% and monitor renal markers quarterly.', 'Children: always ca...` |
| `comorbidity_cautions.drug_herb_interaction_risk` | `str` | 500/500 | 0.0% | `High — multiple potential interactions identified; review all concurrent medications with a pharm...` <br> `Moderate — some interaction potential; inform prescribing clinicians of all herbal use.` <br> `Low — minimal known interactions; standard precautions apply.` |
| `comorbidity_cautions.general_advice` | `str` | 500/500 | 0.0% | `Always disclose all herbal remedies to all treating clinicians. Do not substitute prescribed medi...` <br> `Disclose all herbal remedies to treating clinicians.` |
| `comorbidity_cautions.use_with_caution_in` | `list` | 500/500 | 0.0% | `['Pregnancy: safety not established — avoid unless specialist-supervised.', 'Diabetes mellitus: s...` <br> `['Pregnancy: safety not established — avoid unless specialist-supervised.', 'Peri-operative: disc...` <br> `['Chronic kidney disease: many herbs require dose reduction; monitor renal function; avoid potass...` |
| `condition` | `str` | 500/500 | 0.0% | `rheumatoid_arthritis` <br> `bronchial_asthma` <br> `tumor_cyst_neoplasm` |
| `condition_category` | `str` | 500/500 | 0.0% | `chronic` <br> `acute_environmental` <br> `daily_ailment` |
| `contraindications` | `list` | 500/500 | 0.0% | `['Avoid unsupervised oral use of Maha Yogaraja Guggulu during pregnancy, lactation, in frail elde...` <br> `['Avoid unsupervised oral use of Swasakasantaka during pregnancy, lactation, in frail elders, or ...` <br> `['Avoid unsupervised oral use of Kanchanara Guggulu during pregnancy, lactation, in frail elders,...` |
| `dosage.amount` | `str` | 500/500 | 0.0% | `500mg–1g` <br> `125–250mg` <br> `apply a thin layer locally` |
| `dosage.duration` | `str` | 500/500 | 0.0% | `Most cases require 3–6 months of consistent treatment. Please consult a qualified clinician` <br> `up to 3 days` <br> `1-2 days` |
| `dosage.frequency` | `str` | 500/500 | 0.0% | `twice daily` <br> `thrice daily` <br> `twice or thrice daily` |
| `dosage_model.adjustment_tables.age` | `list` | 500/500 | 0.0% | `[[0.0, 0.077, None, None], [0.077, 1.0, 0.1, 50.0], [1.0, 3.0, 0.2, 100.0], [3.0, 12.0, 0.35, 250...` |
| `dosage_model.adjustment_tables.age_labels` | `list` | 500/500 | 0.0% | `['neonate_0_28d', 'infant_1_12m', 'toddler_1_3y', 'child_3_12y', 'adolescent_12_18y', 'adult_18_6...` |
| `dosage_model.adjustment_tables.gender.female_breastfeeding` | `NoneType` | 500/500 | 0.0% | `None` |
| `dosage_model.adjustment_tables.gender.female_non_pregnant` | `float` | 500/500 | 0.0% | `1.0` |
| `dosage_model.adjustment_tables.gender.female_postmenopausal` | `float` | 500/500 | 0.0% | `0.8` |
| `dosage_model.adjustment_tables.gender.female_pregnant` | `NoneType` | 500/500 | 0.0% | `None` |
| `dosage_model.adjustment_tables.gender.male` | `float` | 500/500 | 0.0% | `1.0` |
| `dosage_model.adjustment_tables.hepatic` | `list` | 500/500 | 0.0% | `[[5, 6, 1.0], [7, 9, 0.75], [10, 15, 0.5]]` |
| `dosage_model.adjustment_tables.hepatic_labels` | `list` | 500/500 | 0.0% | `['child_pugh_a', 'child_pugh_b', 'child_pugh_c']` |
| `dosage_model.adjustment_tables.renal` | `list` | 500/500 | 0.0% | `[[60.0, 999.0, 1.0], [45.0, 59.9, 0.75], [30.0, 44.9, 0.5], [15.0, 29.9, 0.25], [0.0, 14.9, None]]` |
| `dosage_model.adjustment_tables.renal_labels` | `list` | 500/500 | 0.0% | `['normal', 'mild_ckd', 'moderate_ckd', 'severe_ckd', 'kidney_failure']` |
| `dosage_model.adjustment_tables.severity.mild` | `float` | 500/500 | 0.0% | `0.75` |
| `dosage_model.adjustment_tables.severity.moderate` | `float` | 500/500 | 0.0% | `1.0` |
| `dosage_model.adjustment_tables.severity.severe` | `float` | 500/500 | 0.0% | `1.25` |
| `dosage_model.adjustment_tables.weight.adult_obesity_mult` | `float` | 500/500 | 0.0% | `1.0` |
| `dosage_model.adjustment_tables.weight.adult_obesity_thresh_kg` | `float` | 500/500 | 0.0% | `100.0` |
| `dosage_model.adjustment_tables.weight.adult_underweight_mult` | `float` | 500/500 | 0.0% | `0.75` |
| `dosage_model.adjustment_tables.weight.adult_underweight_thresh_kg` | `float` | 500/500 | 0.0% | `45.0` |
| `dosage_model.adjustment_tables.weight.formula` | `str` | 500/500 | 0.0% | `adjusted = base * (patient_kg / 70.0)` |
| `dosage_model.adjustment_tables.weight.paediatric_only` | `bool` | 500/500 | 0.0% | `True` |
| `dosage_model.adjustment_tables.weight.reference_kg` | `float` | 500/500 | 0.0% | `70.0` |
| `dosage_model.components` | `list` | 500/500 | 0.0% | `[{'herb': 'guggulu', 'herb_display': '500mg–1g', 'route': 'oral', 'base_dose_min_mg': 500.0, 'bas...` <br> `[{'herb': '__generic_oral__', 'herb_display': '500mg–1g', 'route': 'oral', 'base_dose_min_mg': 50...` <br> `[{'herb': '__generic_oral__', 'herb_display': '125–250mg', 'route': 'oral', 'base_dose_min_mg': 1...` |
| `dosage_model.compute_function` | `str` | 500/500 | 0.0% | `
def compute_dose(base_dose_mid_mg, doses_per_day,
                 age_years, weight_kg, sex,
  ...` |
| `dosage_model.flags.chronic_ongoing` | `bool` | 500/500 | 0.0% | `True` <br> `False` |
| `dosage_model.flags.has_numeric_dose` | `bool` | 500/500 | 0.0% | `True` <br> `False` |
| `dosage_model.flags.is_emergency` | `bool` | 500/500 | 0.0% | `False` <br> `True` |
| `dosage_model.flags.is_topical` | `bool` | 500/500 | 0.0% | `False` <br> `True` |
| `dosage_model.flags.multi_ingredient` | `bool` | 500/500 | 0.0% | `False` <br> `True` |
| `dosage_model.flags.requires_practitioner` | `bool` | 500/500 | 0.0% | `True` <br> `False` |
| `dosage_model.numeric_examples` | `NoneType, list` | 500/500 | 0.0% | `[{'herb': 'guggulu', 'profiles': [{'profile': 'adult_standard', 'inputs': {'weight_kg': 70, 'age_...` <br> `None` <br> `[{'herb': '__generic_oral__', 'profiles': [{'profile': 'adult_standard', 'inputs': {'weight_kg': ...` |
| `dosage_model.version` | `str` | 500/500 | 0.0% | `2.0` |
| `dosha_involvement` | `list` | 500/500 | 0.0% | `['vata', 'pitta', 'kapha']` <br> `['vata', 'pitta']` <br> `['vata', 'kapha']` |
| `escalation.condition` | `str` | 500/500 | 0.0% | `Seek clinician review if symptoms progress, function declines, or there is significant pain, weig...` <br> `Seek urgent care for shortness of breath at rest, chest indrawing, bluish lips, inability to spea...` <br> `Seek prompt specialist review for a growing mass, persistent pain, ulceration, fever, weight loss...` |
| `escalation.needed` | `bool` | 500/500 | 0.0% | `True` <br> `False` |
| `evidence_level` | `str` | 500/500 | 0.0% | `classical_text` <br> `traditional_use` |
| `final_explanation` | `str` | 500/500 | 0.0% | `The selected remedy is Maha Yogaraja Guggulu, chosen to match the dosha pattern and symptom profi...` <br> `The selected remedy is Swasakasantaka, chosen to match the dosha pattern and symptom profile of t...` <br> `The selected remedy is Kanchanara Guggulu, chosen to match the dosha pattern and symptom profile ...` |
| `gender_considerations.additional_notes` | `list` | 500/500 | 0.0% | `['Primarily indicated for female health conditions.']` <br> `['No gender-specific restrictions identified beyond pregnancy and breastfeeding considerations.']` <br> `['Primarily indicated for female health conditions.', 'Contains hormonally active compounds — mon...` |
| `gender_considerations.breastfeeding_note` | `str` | 500/500 | 0.0% | `Contraindicated during breastfeeding — seek alternative management.` <br> `Breastfeeding safety not fully established; consult a practitioner before use.` <br> `Breastfeeding safety not fully established.` |
| `gender_considerations.breastfeeding_safe` | `NoneType, bool` | 500/500 | 0.0% | `False` <br> `None` |
| `gender_considerations.pregnancy_note` | `str` | 500/500 | 0.0% | `Insufficient safety data — avoid or use only under specialist supervision during pregnancy.` <br> `Safety in pregnancy not established; consult a qualified practitioner before use.` <br> `Contraindicated — do not use during pregnancy without specialist medical and Ayurvedic supervision.` |
| `gender_considerations.pregnancy_safe` | `NoneType, bool` | 500/500 | 0.0% | `None` <br> `False` <br> `True` |
| `gender_considerations.suitable_for` | `list` | 500/500 | 0.0% | `['female']` <br> `['all genders']` <br> `['male']` |
| `knowledge_source` | `str` | 500/500 | 0.0% | `classical` <br> `folk` |
| `lifestyle_factors.diet_recommendations` | `list` | 500/500 | 0.0% | `['Follow a pitta-pacifying diet: cool, sweet, bitter, and astringent foods; favour milk, ghee, co...` <br> `['Follow a vata-pacifying diet: warm, oily, nourishing, and regular meals; favour cooked grains, ...` <br> `['Follow a kapha-pacifying diet: light, dry, warm, and spiced foods; favour legumes, bitter veget...` |
| `lifestyle_factors.foods_to_avoid` | `list` | 500/500 | 0.0% | `['Avoid spicy, fermented, oily, fried, and sour foods; limit alcohol, caffeine, and red meat.', '...` <br> `['Avoid raw cold foods, carbonated drinks, prolonged fasting, and irregular meal timing.', 'Avoid...` <br> `['Avoid spicy, fermented, oily, fried, and sour foods; limit alcohol, caffeine, and red meat.']` |
| `lifestyle_factors.physical_activity` | `str` | 500/500 | 0.0% | `Minimum 150 minutes moderate aerobic exercise per week; include resistance training twice weekly.` <br> `Gentle walking and pranayama (breathing exercises) are recommended; avoid strenuous exertion in a...` <br> `Regular moderate physical activity (30 minutes daily walking or yoga) supports dosha balance and ...` |
| `lifestyle_factors.sleep_guidance` | `str` | 500/500 | 0.0% | `Aim for 7–9 hours of quality sleep; adequate rest supports immune function and dosha balance.` <br> `Maintain strict sleep hygiene: consistent bedtime and wake time, dark cool room, no screens 1 hou...` <br> `Aim for 7–9 hours of quality sleep.` |
| `lifestyle_factors.smoking` | `str` | 500/500 | 0.0% | `Avoid smoking; tobacco aggravates pitta and vata and reduces remedy efficacy.` <br> `Smoking is contraindicated — it significantly worsens this condition.` <br> `Avoid smoking; tobacco reduces remedy efficacy.` |
| `lifestyle_factors.stress_management` | `str` | 500/500 | 0.0% | `Moderate stress management through relaxation practices supports the effectiveness of the remedy.` <br> `Stress management is essential: daily pranayama, meditation, or mindfulness practice for 15–20 mi...` <br> `Daily pranayama or mindfulness supports recovery.` |
| `lifestyle_factors.substance_avoidance` | `str` | 500/500 | 0.0% | `Limit alcohol to minimal or none; alcohol aggravates most dosha imbalances.` <br> `Avoid alcohol completely — it directly antagonises the therapeutic action and may worsen the cond...` <br> `Limit alcohol; avoid smoking.` |
| `onset_type` | `str` | 500/500 | 0.0% | `sudden` <br> `gradual` <br> `acute` |
| `reasoning_steps` | `list` | 500/500 | 0.0% | `['The symptom cluster of Severe joint pain and swelling, Tenderness and redness of affected joint...` <br> `['The symptom cluster of Difficulty breathing or breathlessness, Wheezing, dry mouth in cold-type...` <br> `['The symptom cluster of Painless or painful abnormal growth or lump in breast, or white discolou...` |
| `remedy.name` | `str` | 500/500 | 0.0% | `Maha Yogaraja Guggulu` <br> `Swasakasantaka` <br> `Kanchanara Guggulu` |
| `remedy.type` | `str` | 500/500 | 0.0% | `oral` <br> `topical` <br> `lifestyle` |
| `requires_physical_intervention` | `bool` | 500/500 | 0.0% | `False` <br> `True` |
| `safe_for_home_remedy` | `bool` | 500/500 | 0.0% | `False` <br> `True` |
| `severity` | `str` | 500/500 | 0.0% | `severe` <br> `moderate` <br> `mild` |
| `symptoms` | `list` | 500/500 | 0.0% | `['Severe joint pain and swelling', 'Tenderness and redness of affected joints', 'Inflammation wit...` <br> `['Difficulty breathing or breathlessness', 'Wheezing', 'dry mouth in cold-type', 'Fever']` <br> `['Painless or painful abnormal growth or lump in breast', 'or white discolouration depending on D...` |
| `treatment_plan.agent_decision_tree.nodes` | `list` | 500/500 | 0.0% | `[{'id': 'N1', 'type': 'condition_check', 'parameter': 'gfr', 'operator': '<', 'threshold': 30.0, ...` |
| `treatment_plan.agent_decision_tree.root_condition` | `str` | 500/500 | 0.0% | `rheumatoid_arthritis` <br> `bronchial_asthma` <br> `tumor_cyst_neoplasm` |
| `treatment_plan.agent_decision_tree.version` | `str` | 500/500 | 0.0% | `1.0` |
| `treatment_plan.alternatives` | `list` | 500/500 | 0.0% | `[{'herb': 'shallaki', 'efficacy': 0.85, 'safety': 0.82, 'evidence': 0.88, 'route': 'oral', 'rank'...` <br> `[{'herb': 'sitopaladi', 'efficacy': 0.75, 'safety': 0.9, 'evidence': 0.7, 'route': 'oral', 'rank'...` <br> `[{'herb': 'shatavari', 'efficacy': 0.8, 'safety': 0.9, 'evidence': 0.75, 'route': 'oral', 'rank':...` |
| `treatment_plan.dose_optimization.constraints` | `list` | 500/500 | 0.0% | `[{'parameter': 'single_dose_mg', 'operator': '<=', 'value': 'canonical_max_single_mg'}, {'paramet...` |
| `treatment_plan.dose_optimization.convergence_target` | `float` | 500/500 | 0.0% | `0.85` |
| `treatment_plan.dose_optimization.objective` | `str` | 500/500 | 0.0% | `maximise_weighted_score` |
| `treatment_plan.dose_optimization.objective_weights.efficacy` | `float` | 500/500 | 0.0% | `0.55` <br> `0.45` <br> `0.4` |
| `treatment_plan.dose_optimization.objective_weights.safety` | `float` | 500/500 | 0.0% | `0.35` <br> `0.4` <br> `0.5` |
| `treatment_plan.dose_optimization.objective_weights.tolerability` | `float` | 500/500 | 0.0% | `0.1` <br> `0.15` |
| `treatment_plan.dose_optimization.optimization_method` | `str` | 500/500 | 0.0% | `grid_search` |
| `treatment_plan.dose_optimization.search_space` | `list` | 500/500 | 0.0% | `[{'herb': 'guggulu', 'dose_range_min': 500.0, 'dose_range_max': 500, 'step_size_mg': 25.0, 'curre...` <br> `[{'herb': '__generic_oral__', 'dose_range_min': 500.0, 'dose_range_max': 500, 'step_size_mg': 25....` <br> `[{'herb': '__generic_oral__', 'dose_range_min': 125.0, 'dose_range_max': 500, 'step_size_mg': 37....` |
| `treatment_plan.herb_interactions` | `list` | 500/500 | 0.0% | `[]` <br> `[{'herb_a': None, 'herb_b': None, 'risk_score': 0.0, 'mechanism': None, 'direction': None, 'actio...` <br> `[{'herb_a': 'brahmi', 'herb_b': 'ashwagandha', 'risk_score': 0.1, 'mechanism': 'SED', 'direction'...` |
| `treatment_plan.monitoring_checkpoints` | `list` | 500/500 | 0.0% | `[{'day': 7, 'measures': ['symptom_score_0_10', 'tolerability_score_0_10'], 'rule': {'if_symptom_g...` <br> `[{'day': 1, 'measures': ['symptom_score_0_10', 'vital_signs_if_applicable'], 'rule': {'if_worseni...` <br> `[{'day': 7, 'measures': ['symptom_score_0_10', 'tolerability_score_0_10'], 'rule': {'if_no_improv...` |
| `treatment_plan.phases` | `list` | 500/500 | 0.0% | `[{'phase': 'induction', 'dose_pct': 50, 'duration_days_min': 7, 'duration_days_max': 14, 'goal': ...` <br> `[{'phase': 'emergency', 'dose_pct': 100, 'duration_days': 1, 'goal': 'stabilise', 'titrate': False}]` <br> `[{'phase': 'treatment', 'dose_pct': 75, 'duration_days_min': 7, 'duration_days_max': 21, 'goal': ...` |
| `treatment_plan.titration.applicable` | `bool` | 500/500 | 0.0% | `True` <br> `False` |
| `treatment_plan.titration.dose_schedule` | `list` | 424/500 | 15.2% | `[{'herb': 'guggulu', 'start_dose_mg': 375.0, 'target_dose_mg': 937.5, 'step_size_mg': 187.5, 'dos...` <br> `[{'herb': '__generic_oral__', 'start_dose_mg': 93.8, 'target_dose_mg': 234.4, 'step_size_mg': 46....` <br> `[]` |
| `treatment_plan.titration.minimum_pct` | `int` | 424/500 | 15.2% | `25` <br> `50` |
| `treatment_plan.titration.reason_code` | `str` | 76/500 | 84.8% | `EMERGENCY_FIXED_DOSE` |
| `treatment_plan.titration.start_pct` | `int` | 424/500 | 15.2% | `50` <br> `75` <br> `25` |
| `treatment_plan.titration.step_interval_days` | `int` | 424/500 | 15.2% | `7` |
| `treatment_plan.titration.step_size_pct` | `int` | 424/500 | 15.2% | `25` |
| `treatment_plan.titration.taper_same_schedule` | `bool` | 424/500 | 15.2% | `True` |
| `treatment_plan.titration.target_pct` | `int` | 424/500 | 15.2% | `125` <br> `100` <br> `75` |
| `treatment_plan.variation_simulations` | `list` | 500/500 | 0.0% | `[{'scenario': 'gfr_drops_to_40', 'computable': True, 'reason_code': None, 'dose_change_pct': -50....` <br> `[]` |
| `weight_considerations.body_weight_note` | `str` | 500/500 | 0.0% | `For severe conditions, body weight may influence dose titration — involve a qualified practitioner.` <br> `Adult dosing is typically weight-independent for Ayurvedic herbs unless specifically stated.` <br> `Paediatric dosing: 1 mg per kg body weight per dose is a common Ayurvedic starting guideline; adj...` |
| `weight_considerations.obesity_adjustment` | `str` | 500/500 | 0.0% | `Standard dosing applies; no specific obesity-related adjustment required for this remedy.` <br> `Weight-based dosing may apply; heavier body weight often indicates kapha dominance — Trikatu or G...` <br> `Standard dosing applies.` |
| `weight_considerations.obesity_specific_flag` | `bool` | 500/500 | 0.0% | `False` <br> `True` |
| `weight_considerations.paediatric_weight_based_dosing` | `str` | 500/500 | 0.0% | `Not applicable for standard adult use` <br> `Yes — see age adjustments above` |
| `weight_considerations.underweight_flag` | `bool` | 500/500 | 0.0% | `True` <br> `False` |
| `weight_considerations.underweight_frail_adjustment` | `str` | 500/500 | 0.0% | `Frail or underweight patients: start at lowest dose; prioritise nutritional support (Chyawanprash...` <br> `Standard dosing applies; ensure adequate nutritional status before starting the remedy.` <br> `Start at 75% dose in frail or underweight patients.` |
