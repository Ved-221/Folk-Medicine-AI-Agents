# Product Requirements Document (PRD)

# Project: Folk Medicine AI Agents (v2.0)
**Preserving Indigenous Folk Medicines — Research Honors Programme**  
**Document Version:** 2.0.0  
**Status:** Approved & Implemented Specification  
**Architecture:** Dual Deterministic Pipeline (Agent 1: Extraction & Retrieval; Agent 2: Recommender & Safety Gate)  

---

The comprehensive, institutional-grade Product Requirements Document (PRD) is published at:
👉 **[docs/PRD.md](file:///Users/veddhapodkar/Documents/REPO/Folk-Medicine-AI-Agents/docs/PRD.md)**

### Table of Contents Summary in `docs/PRD.md`:
1. **Document Control & Revision History**: Version tracking from v1.0 heuristics to v2.0 grounded extraction & deterministic safety.
2. **Executive Summary & Strategic Context**: Ethnobotanical preservation crisis, LLM medical safety paradox, research honors objectives, and core grounding philosophy.
3. **Core Philosophy & Architectural Boundaries**: "Code Governs, LLMs Rephrase" contract, explicit inclusions/exclusions, and the Ayurvedic ontological framework (*Tridosha*, *Anupana*, *Kala*).
4. **User Personas & End-to-End User Journeys**: Ethnobotanist researcher, institutional clinical reviewer, and public seeker with complete Mermaid state transition diagrams.
5. **System Architecture & Component Specifications**: Decoupled repository layout, DAG orchestration, and directory structures.
6. **Detailed Functional Requirements (FR)**:
   - **FR-1.1**: Ingestion & scaffolding trimmer (`loader.py`, `trimmer.py`).
   - **FR-1.2**: LLM context extraction & verbatim quote verification (`extractor.py`, `validator.py`).
   - **FR-1.3**: Human review quarantine queue (`review_queue.py`).
   - **FR-1.4**: Vector embedding generation & indexing (`embedder.py`).
   - **FR-1.5**: Conversational & Hinglish vernacular query parsing (`query_parser.py`).
   - **FR-1.6**: Tri-factor hybrid retrieval engine (`query.py`).
   - **FR-2.1**: Deterministic code clinical safety gate (`safety.py`).
   - **FR-2.2**: Table-driven age-band dosage resolution (`age_band.py`).
   - **FR-2.3**: Constrained 6-field recommendation formatter (`formatter.py`).
7. **Detailed Non-Functional Requirements (NFR)**: Zero-hallucination invariants, sub-5ms safety latency, Gemini free tier call throttling (15 RPM) and persistent disk caching, offline air-gapped resilience, zero PII persistence, plain Python tech stack without agent frameworks.
8. **Data Schemas & Interface Specifications**: JSON schemas for extracted fields, review items, and standard 6-field outputs.
9. **Evaluation Methodology & Benchmark Performance**: Gold benchmark dataset results (Recall@3: 85.0%, MRR: 0.725, 100% quote match rate, 100% emergency red-flag screening catch rate).
10. **Test Strategy & Traceability Matrix**: Mapping of all 54 passing unit and integration tests across components.
11. **Future Roadmap**: Multimodal audio ingestion (elder interviews), palm-leaf manuscript OCR, pan-Indian vernacular expansion, and AYUSH pharmacopoeial alignment.

---
*For the complete specification, read the full text in [docs/PRD.md](file:///Users/veddhapodkar/Documents/REPO/Folk-Medicine-AI-Agents/docs/PRD.md).*
