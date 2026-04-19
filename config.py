"""
Centralized configuration for the AI Session Monitoring System.
Uses environment variables with sensible defaults.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Audio ────────────────────────────────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds per live capture chunk
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ─── Video ────────────────────────────────────────────────────────────────────
VIDEO_ANALYSIS_FPS = 2  # frames per second for processing (efficiency)
FACE_SCALE_FACTOR = 1.3
FACE_MIN_NEIGHBORS = 5
EMOTION_BACKENDS = ["opencv"]  # DeepFace backends: opencv, retinaface, mtcnn
ATTENTION_THRESHOLD = 0.6  # iris position threshold for "looking at screen"

# ─── NLP ──────────────────────────────────────────────────────────────────────
TOPIC_KEYWORDS = {
    "Projects": ["project", "built", "developed", "system", "application", "created", "designed", "implemented"],
    "Experience": ["internship", "worked", "company", "experience", "job", "role", "position", "employed"],
    "Education": ["study", "studied", "college", "university", "degree", "education", "school", "course"],
    "Career Goals": ["goal", "future", "career", "long term", "down the line", "aspire", "plan", "ambition"],
    "Skills": ["skill", "python", "machine learning", "data", "analysis", "technical", "programming", "tool"],
}

HESITATION_WORDS = ["um", "uh", "maybe", "i think", "sort of", "kind of", "probably", "like", "you know"]

SARCASM_PATTERNS = [
    "yeah right", "oh great", "sure thing", "oh wonderful", "oh perfect",
    "totally", "absolutely fantastic", "oh really", "wow amazing",
    "how nice", "brilliant", "oh joy", "what a surprise",
]

CONFUSION_PHRASES = [
    "i don't understand", "what do you mean", "could you clarify",
    "i'm confused", "can you explain", "not sure what",
    "what exactly", "sorry, i didn't get", "come again",
    "pardon", "i'm not following", "what are you saying",
]

EXPECTED_TOPICS = ["Projects", "Skills", "Experience", "Education", "Career Goals"]

# ─── Scoring Weights ─────────────────────────────────────────────────────────
CANDIDATE_SCORE_WEIGHTS = {
    "communication": 0.30,
    "positivity": 0.25,
    "confidence": 0.25,
    "response_depth": 0.20,
}

COMPOSITE_SCORE_WEIGHTS = {
    "candidate_score": 0.30,
    "engagement_score": 0.20,
    "topic_coverage": 0.15,
    "clarification_success": 0.10,
    "visual_engagement": 0.25,
}

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
