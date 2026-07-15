"""
Global configuration for ResumeSense AI
"""

APP_TITLE = "ResumeSense AI"
APP_TAGLINE = "Offline AI Resume Analyzer | Privacy First | On-Device AI"

PRIMARY_COLOR = (56, 189, 248)     # Sky Blue
BG_COLOR = (15, 23, 42)            # Dark Slate
CARD_COLOR = (30, 41, 59)
TEXT_COLOR = (248, 250, 252)

ALLOWED_EXTENSIONS = ["pdf"]

MAX_UPLOAD_SIZE_MB = 10

APP_VERSION = "1.0.0"

MODEL_NAME = "Offline Skill Analyzer"

PRIVACY_MESSAGE = (
    "🔒 All resume analysis is performed locally on your device. "
    "No resume content is uploaded to cloud AI services."
)
# Skill Extractor Configuration
SKILL_TAXONOMY_PATH = "utils/models/skill_taxonomy.json"

USE_SEMANTIC_MATCHING = False

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

EMBEDDING_MODEL_CACHE_DIR = "./models"
# ATS Scoring Configuration

ATS_WEIGHTS = {
    "section_completeness": 0.20,
    "skill_match": 0.30,
    "keyword_density": 0.15,
    "formatting": 0.15,
    "readability": 0.20,
}

MIN_RESUME_WORDS = 250
MAX_RESUME_WORDS = 700
