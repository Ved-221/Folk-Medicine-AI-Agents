# Folk Medicine AI Agents

Preserving indigenous folk medicine and classical Ayurvedic remedies through a deterministic, verifiable AI retrieval and recommendation pipeline. Built as part of a research honors programme.

## Core Principles
1. **Deterministic Pipeline, Not an Autonomous Agent:** Code decides step order and all safety-critical checks. The LLM never chooses actions or loops on its own.
2. **Zero Medical Hallucinations:** The LLM strictly rephrases facts present in retrieved source records. Every offline extraction requires verbatim quote validation.
3. **Hard Safety Gate:** The safety gate (`src/agent2/safety.py`) runs in pure Python before any LLM is called and cannot be bypassed.
4. **Transparent Provenance:** Every output traces back to its source dataset row, knowledge source (`classical` vs `folk`), and dosage origin (`from_source` vs `derived`).

## Project Structure
```
Folk-Medicine-AI-Agents/
├── PLAN.md                    # Project brief & principles
├── AGENTS.md                  # Workspace rules for Antigravity agents
├── data/
│   ├── raw/                   # Immutable raw JSONL dataset
│   ├── clean/                 # Normalized JSONL dataset
│   ├── enriched/              # Context-enriched JSONL dataset
│   ├── index/                 # Embeddings index & id mapping
│   ├── review/                # Human review queue for flagged items
│   └── gold/                  # Hand-labelled ground-truth evaluation set
├── src/
│   ├── common/                # Shared config, schemas, unified LLM client
│   ├── agent1/                # Loader, trimmer, extractor, validator, query parser, retrieval
│   ├── agent2/                # Age-banding, deterministic safety gate, formatter, pipeline
│   └── eval/                  # Gold-set evaluation scripts (precision, recall, retrieval)
├── tests/                     # Comprehensive test suite with mocked LLM calls
└── docs/                      # Schema, evaluation results, and review samples
```

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and supply your GEMINI_API_KEY
```

### 2. Run Test Suite
```bash
pytest
```

## Disclaimer
This system is an academic research and archival preservation tool. Its outputs are drawn directly from a documented research dataset and do not constitute clinical or medical advice.
