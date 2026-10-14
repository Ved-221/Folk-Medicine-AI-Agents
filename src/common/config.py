"""Configuration constants and paths for Folk Medicine AI Agents."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Root directory of the repository
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Data Paths
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "ayurvedic_dataset.jsonl"
WORKSPACE_RAW_DATA_PATH = ROOT_DIR / "ayurvedic_dataset.jsonl"
CLEAN_DATA_PATH = ROOT_DIR / "data" / "clean" / "dataset.jsonl"
ENRICHED_DATA_PATH = ROOT_DIR / "data" / "enriched" / "dataset.jsonl"
INDEX_DIR = ROOT_DIR / "data" / "index"
REVIEW_QUEUE_PATH = ROOT_DIR / "data" / "review" / "queue.jsonl"
GOLD_DIR = ROOT_DIR / "data" / "gold"
SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample.jsonl"
CACHE_DIR = ROOT_DIR / ".cache" / "llm"

# Gemini LLM & Embeddings Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
MAX_LLM_CALLS_PER_RUN = int(os.getenv("MAX_LLM_CALLS_PER_RUN", "50"))

# Thresholds & Rules
VALID_DOSHAS = {"vata", "pitta", "kapha"}
FUZZY_MATCH_THRESHOLD = 60.0  # Rapidfuzz token score threshold (0-100)
SEMANTIC_MATCH_THRESHOLD = 0.55  # Cosine similarity threshold (0-1)
HYBRID_SEMANTIC_WEIGHT = 0.65
HYBRID_FUZZY_WEIGHT = 0.35
AUTO_ACCEPT_CONFIDENCE_THRESHOLD = 0.80

# Deterministic Red-Flag Symptoms (Mandatory Immediate see_doctor)
RED_FLAGS_LIST = [
    "coughing blood",
    "coughing up blood",
    "blood in sputum",
    "chest pain",
    "crushing chest pain",
    "stabbing chest pain",
    "radiating pain",
    "severe breathlessness",
    "shortness of breath at rest",
    "blue lips",
    "gasping for air",
    "facial paralysis",
    "sudden paralysis",
    "arm weakness",
    "slurred speech",
    "high fever 105",
    "stiff neck",
    "seizure",
    "loss of consciousness",
    "confusion",
    "anaphylaxis",
]
